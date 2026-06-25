"""
src/trainer.py
==============
Training and validation logic for HOLE with the DIVE-inspired loss stack.

Functions
---------
mine_semi_hard_triplets
    GPU-efficient semi-hard negative mining within a batch of embeddings.
    Returns (anchor, positive, negative) tensor triples using the full-dim
    (primary) head embeddings. Mining strategy transitions from random-positive
    to hardest-positive after a warmup period.

train_one_epoch
    Runs one full training epoch. Accepts a HOLEModel + MatryoshkaDIVELoss and
    handles multi-head forward pass, triplet mining on the primary head, and
    loss computation.

validate_epoch
    Full validation pass: accumulates all embeddings, runs global semi-hard
    mining across the entire validation set, and computes separation metrics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .loss import MatryoshkaDIVELoss, MatryoshkaTripletLoss


# ── Triplet mining ────────────────────────────────────────────────────────────

def mine_semi_hard_triplets(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GPU-efficient semi-hard negative mining within a batch.

    Mining strategy:
    - Epochs 1–5:  Random positive  (stability during warmup)
    - Epochs > 5:  Hardest positive (squeeze performance)
    - All epochs:  Semi-hard negative (d_an > d_ap, d_an < d_ap + margin)
                   Falls back to easiest negative if no semi-hard found.

    Args:
        embeddings: L2-normalised embeddings (B, d), primary head only.
        labels:     Binary labels (B,) — 1=pit, 0=negative.
        margin:     Triplet margin (same as loss margin for consistency).
        epoch:      Current training epoch (controls mining strategy).

    Returns:
        (anchors, positives, negatives) — each (N, d) — or empty tensors if
        fewer than 2 pits or 1 negative in batch.
    """
    is_pit = labels == 1
    is_neg = labels == 0
    pit_idx = torch.where(is_pit)[0]
    neg_idx = torch.where(is_neg)[0]

    if len(pit_idx) < 2 or len(neg_idx) < 1:
        empty = torch.tensor([], device=embeddings.device)
        return empty, empty, empty

    # Pairwise cosine distance matrix
    dist_matrix = 1.0 - torch.matmul(embeddings, embeddings.T)  # (B, B)

    anchors, positives, negatives = [], [], []

    for i in pit_idx:
        # --- Positive selection ---
        pos_mask = is_pit.clone()
        pos_mask[i] = False
        potential_pos = torch.where(pos_mask)[0]

        if epoch <= 5:
            p_idx = potential_pos[torch.randint(len(potential_pos), (1,))].item()
        else:
            # Hardest positive: furthest pit from anchor
            p_idx = potential_pos[torch.argmax(dist_matrix[i, pos_mask])].item()

        d_ap = dist_matrix[i, p_idx]

        # --- Semi-hard negative selection ---
        d_an = dist_matrix[i, is_neg]  # distances to all negatives
        semi_hard_mask = (d_an > d_ap) & (d_an < d_ap + margin)
        semi_hard_negs = neg_idx[semi_hard_mask]

        if len(semi_hard_negs) > 0:
            # Among valid semi-hards: pick the closest (hardest semi-hard)
            n_idx = semi_hard_negs[torch.argmin(d_an[semi_hard_mask])].item()
        else:
            # Fallback: hardest available negative (closest)
            n_idx = neg_idx[torch.argmin(d_an)].item()

        anchors.append(embeddings[i])
        positives.append(embeddings[p_idx])
        negatives.append(embeddings[n_idx])

    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


# ── Training epoch ────────────────────────────────────────────────────────────

def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MatryoshkaDIVELoss | MatryoshkaTripletLoss,
    device: torch.device,
    margin: float,
    epoch: int,
) -> tuple[float, dict[str, float]]:
    """Run one training epoch.
    
    Handles both HOLEModel (multi-head) and DinoExtractor (single-head, legacy).
    
    For HOLEModel + MatryoshkaDIVELoss:
        - Runs multi-head forward pass
        - Mines triplets on primary (full-dim) head
        - Computes hinge loss on primary head triplets
        - Computes NT-Xent loss across all heads (on full batch, before mining)

    For DinoExtractor + MatryoshkaTripletLoss (legacy):
        - Runs single-head forward pass
        - Falls back to original triplet logic

    Returns:
        (mean_loss, sub_loss_averages)
    """
    model.train()
    total_loss = 0.0
    sub_totals: dict[str, float] = {}
    n_batches = 0

    for images, labels in tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        # ── Multi-head forward (HOLEModel) ──
        if isinstance(criterion, MatryoshkaDIVELoss):
            heads: list[torch.Tensor] = model(images)  # [(B,d₀)...(B,dₙ)]
            primary = heads[-1]                          # (B, d_primary)

            a, p, n = mine_semi_hard_triplets(primary, labels, margin, epoch)
            if len(a) == 0:
                continue

            loss, subs = criterion(heads, a, p, n)

        # ── Single-head forward (DinoExtractor + legacy loss) ──
        else:
            embs = F.normalize(model(images), p=2, dim=1)
            a, p, n = mine_semi_hard_triplets(embs, labels, margin, epoch)
            if len(a) == 0:
                continue
            loss, subs = criterion(a, p, n)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in subs.items():
            sub_totals[k] = sub_totals.get(k, 0.0) + v
        n_batches += 1

    if n_batches == 0:
        return 0.0, {}

    return (
        total_loss / n_batches,
        {k: v / n_batches for k, v in sub_totals.items()},
    )


# ── Validation epoch ──────────────────────────────────────────────────────────

def validate_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MatryoshkaDIVELoss | MatryoshkaTripletLoss,
    device: torch.device,
    margin: float,
) -> tuple[float, float, float, float, float]:
    """Full validation pass.

    Accumulates all embeddings across the validation set, then computes:
    - val_loss:     Triplet/DIVE loss on mined triplets from the full val set
    - avg_sep:      sim_pp - sim_pn (embedding space separation)
    - accuracy:     Hardest-pair accuracy (% pits closer to nearest pit than nearest negative)
    - sim_pp:       Mean pit-pit cosine similarity
    - sim_pn:       Mean pit-negative cosine similarity

    Returns:
        (val_loss, avg_separation, accuracy_pct, sim_pp, sim_pn)
    """
    model.eval()
    all_primary: list[torch.Tensor] = []
    all_heads_list: list[list[torch.Tensor]] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            if isinstance(criterion, MatryoshkaDIVELoss):
                heads = model(images)
                all_primary.append(heads[-1].cpu())
                all_heads_list.append([h.cpu() for h in heads])
            else:
                embs = F.normalize(model(images), p=2, dim=1)
                all_primary.append(embs.cpu())

            all_labels.append(labels)

    primary = torch.cat(all_primary)          # (N, d_primary)
    labels  = torch.cat(all_labels)           # (N,)

    primary_dev = primary.to(device)
    labels_dev  = labels.to(device)

    # --- Val loss ---
    val_a, val_p, val_n = mine_semi_hard_triplets(
        primary_dev, labels_dev, margin, epoch=6  # epoch>5 → hardest-pos strategy
    )

    if len(val_a) > 0:
        if isinstance(criterion, MatryoshkaDIVELoss):
            # For DIVE loss: reconstruct multi-head tensors for val set
            # For simplicity, use only hinge component on primary for val loss
            from .loss import SelfLimitingHingeLoss
            _hinge = SelfLimitingHingeLoss(margin=criterion.config.hinge_margin)
            val_loss_tensor, _ = _hinge(val_a, val_p, val_n)
        else:
            val_loss_tensor, _ = criterion(val_a, val_p, val_n)
        val_loss = val_loss_tensor.item()
    else:
        val_loss = 0.0

    # --- Separation metrics ---
    is_pit = labels == 1
    is_neg = labels == 0
    pit_embs = primary[is_pit]
    neg_embs = primary[is_neg]

    if len(pit_embs) < 2 or len(neg_embs) < 1:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    sim_pp = torch.matmul(pit_embs, pit_embs.T).mean().item()
    sim_pn = torch.matmul(pit_embs, neg_embs.T).mean().item()
    avg_separation = sim_pp - sim_pn

    # Hardest-pair accuracy
    dist_pp = 1.0 - torch.matmul(pit_embs, pit_embs.T)
    dist_pn = 1.0 - torch.matmul(pit_embs, neg_embs.T)
    dist_pp.fill_diagonal_(float("inf"))

    hardest_pos = dist_pp.min(dim=1).values
    hardest_neg = dist_pn.min(dim=1).values
    accuracy = ((hardest_pos < hardest_neg).float().mean() * 100.0).item()

    return val_loss, avg_separation, accuracy, sim_pp, sim_pn