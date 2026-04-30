import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils import get_logger

logger = get_logger(__name__)

def get_hard_triplets(embeddings, labels):
    dist_matrix = torch.cdist(embeddings, embeddings, p=2.0)

    is_pit = (labels == 1)
    is_neg = (labels == 0)

    pit_idx = torch.where(is_pit)[0]
    pit_idx = torch.where(is_neg)[0]

    anchors = []
    positives = []
    negatives = []

    for i in pit_idx:
        dist_to_pits = dist_matrix[i].clone()
        dist_to_pits[~is_pit] = -1.0
        dist_to_pits[i] = -1.0

        # Hardest Positive
        hard_pos_idx = torch.argmax(dist_to_pits)

        dist_to_negs = dist_matrix[i].clone()
        dist_to_negs[is_pit] = float("inf")

        # Hardest Negative
        hard_neg_idx = torch.argmin(dist_to_negs)

        anchors.append(embeddings[i])
        positives.append(embeddings[hard_pos_idx])
        negatives.append(embeddings[hard_neg_idx])

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_sub_losses = {}

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        embeddings = model(images)

        embeddings = F.normalize(embeddings, p=2, dim=1)

        anchors, positives, negatives = get_hard_triplets(embeddings, labels)

        if len(anchors) == 0:
            continue
        loss, sub_losses = criterion(anchors, positives, negatives)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        for k, v in sub_losses.items():
            running_sub_losses[k] = running_sub_losses.get(k, 0.0) + v

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = running_loss / len(loader)
    avg_sub_loss = {k: v / len(loader) for k, v in running_sub_losses.items()}

    return avg_loss, avg_sub_loss


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    total_separation_margin = 0.0 # D(A,P) - D(A,N)
    correct_triplets = 0 # How often is D(A,P) < D(A,N)?
    total_triplets = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            anchors, positives, negatives = get_hard_triplets(embeddings, labels)
            
            if len(anchors) == 0:
                continue
                
            loss, _ = criterion(anchors, positives, negatives)
            running_loss += loss.item()

            # We want this to be smol
            dist_ap = torch.norm(anchors - positives, p=2, dim=1)
            # We want this to be big
            dist_an = torch.norm(anchors - negatives, p=2, dim=1)

            margin = dist_an - dist_ap
            total_separation_margin += margin.sum().item()

            correct_triplets += (margin > 0).sum().item()
            total_triplets += len(anchors)

    avg_loss = running_loss / len(loader)

    avg_separation = total_separation_margin / total_triplets if total_triplets > 0 else 0.0
    accuracy_triplets = (correct_triplets / total_triplets) * 100 if total_triplets > 0 else 0.0

    return avg_loss, avg_separation, accuracy_triplets

