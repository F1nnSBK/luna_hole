import torch
import torch.nn.functional as F
from tqdm import tqdm

def get_semi_hard_triplets(embeddings, labels, margin, epoch):
    """
    Mining strategy: 
    - Epochs 1-5: Random Positive (Stability)
    - Epochs > 5: Hardest Positive (Optimization)
    - All Epochs: Semi-Hard Negative
    """
    is_pit = (labels == 1)
    is_neg = (labels == 0)
    pit_idx = torch.where(is_pit)[0]
    neg_idx = torch.where(is_neg)[0]

    if len(pit_idx) < 2 or len(neg_idx) < 1:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    dist_matrix = 1.0 - torch.matmul(embeddings, embeddings.T)
    anchors, positives, negatives = [], [], []

    for i in pit_idx:
        pos_mask = is_pit.clone()
        pos_mask[i] = False
        potential_positives = torch.where(pos_mask)[0]

        # Toggle strategy based on epoch
        if epoch <= 5:
            # Random Positive for a warmer start
            p_idx = potential_positives[torch.randint(len(potential_positives), (1,))].item()
        else:
            # Hardest Positive to squeeze performance
            p_idx = potential_positives[torch.argmax(dist_matrix[i, pos_mask])]
        
        d_ap = dist_matrix[i, p_idx]

        # Semi-Hard Negative Mining
        d_an = dist_matrix[i, is_neg]
        mask = (d_an > d_ap) & (d_an < d_ap + margin)
        semi_hard = neg_idx[mask]

        if len(semi_hard) > 0:
            n_idx = semi_hard[torch.argmin(d_an[mask])]
        else:
            n_idx = neg_idx[torch.argmin(d_an)]

        anchors.append(embeddings[i])
        positives.append(embeddings[p_idx])
        negatives.append(embeddings[n_idx])

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

def train_one_epoch(model, loader, optimizer, criterion, device, margin, epoch):
    model.train()
    total_loss = 0.0
    sub_totals = {}

    for images, labels in tqdm(loader, desc=f"Training Ep {epoch}", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        embs = F.normalize(model(images), p=2, dim=1)
        a, p, n = get_semi_hard_triplets(embs, labels, margin, epoch)
        
        if len(a) == 0: 
            continue
            
        loss, subs = criterion(a, p, n)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in subs.items():
            sub_totals[k] = sub_totals.get(k, 0.0) + v

    return total_loss / len(loader), {k: v / len(loader) for k, v in sub_totals.items()}

def validate_epoch(model, loader, criterion, device, margin):
    """
    Returns (val_loss, avg_sep, accuracy, sim_pp, sim_pn)
    """
    model.eval()
    all_embs, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            embs = F.normalize(model(images), p=2, dim=1)
            all_embs.append(embs.cpu())
            all_labels.append(labels.cpu())

    embs = torch.cat(all_embs)
    labels = torch.cat(all_labels)

    embs_dev = embs.to(device)
    labels_dev = labels.to(device)
    
    val_a, val_p, val_n = get_semi_hard_triplets(embs_dev, labels_dev, margin, epoch=6)
    
    if len(val_a) > 0:
        val_loss_tensor, _ = criterion(val_a, val_p, val_n)
        val_loss = val_loss_tensor.item()
    else:
        val_loss = 0.0
    
    is_pit = (labels == 1)
    is_neg = (labels == 0)
    pit_embs = embs[is_pit]
    neg_embs = embs[is_neg]

    if len(pit_embs) < 2 or len(neg_embs) < 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Global Similarities
    sim_pp = torch.matmul(pit_embs, pit_embs.T).mean().item()
    sim_pn = torch.matmul(pit_embs, neg_embs.T).mean().item()
    avg_separation = sim_pp - sim_pn

    # Hardest-pair Accuracy
    dist_pp = 1.0 - torch.matmul(pit_embs, pit_embs.T)
    dist_pn = 1.0 - torch.matmul(pit_embs, neg_embs.T)
    dist_pp.fill_diagonal_(float('inf'))
    
    hardest_pos = dist_pp.min(dim=1)[0]
    hardest_neg = dist_pn.min(dim=1)[0]
    
    correct = (hardest_pos < hardest_neg).sum().item()
    accuracy = (correct / len(pit_embs)) * 100.0

    return val_loss / len(loader), avg_separation, accuracy, sim_pp, sim_pn