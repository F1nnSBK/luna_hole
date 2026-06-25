"""
visualize_evaluation.py
========================
Evaluation script to compute metrics (similarities, separation, accuracy, AUC-ROC)
across all Matryoshka dimensions and visualize embeddings (UMAP/t-SNE) plus local
model features (Saliency Map, DINOv3 attention maps) for HOLEModel.

Usage:
------
    python visualize_evaluation.py
"""

import os
import sys
import types
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torch.utils.data import DataLoader

from src.models import HOLEModel
from src.data_loader import get_dataloaders
from src.utils import get_logger

logger = get_logger(__name__)

# Fallback for UMAP to t-SNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    HAS_UMAP = False
    logger.warning("umap-learn not installed. Falling back to t-SNE for dimension reduction.")

# ── Config ──
DATA_DIR     = "data/hf_dataset"
ADAPTER_PATH = "models/lunar_dinov3_lora/standard_lora"
NEST_DIMS    = [64, 128, 256, 384]
BATCH_SIZE   = 32
IMAGE_SIZE   = 224
PATCH_SIZE   = 16
GRID_SIZE    = IMAGE_SIZE // PATCH_SIZE  # 14


# ── Model loading ──

def load_trained_model(device: torch.device) -> HOLEModel:
    """Load base DINOv3 model, LoRA adapter weights, and MLP projection head."""
    logger.info(f"Instantiating HOLEModel on {device}...")
    model = HOLEModel(
        weights_path="models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth",
        model_size="vits16",
        nest_dims=NEST_DIMS,
    )
    logger.info(f"Loading adapter weights from: {ADAPTER_PATH}")
    model.backbone.model.load_adapter(ADAPTER_PATH, adapter_name="standard_lora")
    model.backbone.model.set_adapter("standard_lora")
    
    logger.info(f"Loading projection head weights from: {ADAPTER_PATH}")
    model.load_projection(ADAPTER_PATH)
    
    model.to(device)
    model.eval()
    return model


# ── Saliency map & Attention extraction ──

def compute_saliency_map(model: HOLEModel, img_tensor: torch.Tensor) -> np.ndarray:
    """Computes the vanilla gradient-based saliency map of the input image."""
    img_tensor = img_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass: get primary embedding (heads[-1])
    heads = model(img_tensor)
    primary = heads[-1]
    
    # Compute L2 norm of the embedding to backprop from
    score = primary.norm(p=2)
    
    model.zero_grad()
    score.backward()
    
    grads = img_tensor.grad.detach().cpu().numpy()[0]
    saliency = np.max(np.abs(grads), axis=0)
    
    # Normalize to [0, 1]
    s_min, s_max = saliency.min(), saliency.max()
    if s_max > s_min:
        saliency = (saliency - s_min) / (s_max - s_min + 1e-8)
    return saliency


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, rope: tuple) -> torch.Tensor:
    sin, cos = rope
    return x * cos + _rotate_half(x) * sin


def extract_last_block_attention(model: HOLEModel, img_tensor: torch.Tensor) -> torch.Tensor:
    """Monkeypatches last layer's self-attention to capture raw attention weights."""
    base = model.backbone.model.get_base_model()
    last_attn = base.blocks[-1].attn
    captured = {}

    original_forward = last_attn.forward

    def patched_forward(self, x: torch.Tensor, rope=None, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        head_dim = C // self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if rope is not None:
            num_spatial = rope[0].shape[-2]
            num_non_spatial = N - num_spatial
            
            q_non_spatial, q_spatial = q[:, :, :num_non_spatial, :], q[:, :, num_non_spatial:, :]
            k_non_spatial, k_spatial = k[:, :, :num_non_spatial, :], k[:, :, num_non_spatial:, :]
            
            q_spatial = _apply_rope(q_spatial, rope)
            k_spatial = _apply_rope(k_spatial, rope)
            
            q = torch.cat([q_non_spatial, q_spatial], dim=2)
            k = torch.cat([k_non_spatial, k_spatial], dim=2)

        scale = self.scale if hasattr(self, 'scale') else (head_dim ** -0.5)
        attn = (q * scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        
        captured["attn"] = attn.detach()

        if hasattr(self, 'attn_drop'):
            attn = self.attn_drop(attn)
            
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        if hasattr(self, 'proj_drop'):
            x = self.proj_drop(x)
            
        return x

    last_attn.forward = types.MethodType(patched_forward, last_attn)

    try:
        with torch.no_grad():
            model(img_tensor)
    finally:
        last_attn.forward = original_forward

    if "attn" not in captured:
        raise RuntimeError("Monkeypatch failed! PyTorch __call__ bypassed us.")

    return captured["attn"]


def cls_attention_maps(attn: torch.Tensor, grid: int) -> torch.Tensor:
    """CLS-token -> patch attention per head, reshaped to spatial grid."""
    num_spatial = grid * grid
    num_non_spatial = attn.shape[-1] - num_spatial
    return attn[0, :, 0, num_non_spatial:].reshape(-1, grid, grid)


# ── Metrics evaluation ──

def compute_metrics(embeddings: np.ndarray, labels: np.ndarray):
    """Compute similarities, separation (sim_pp - sim_pn), hardest-pair accuracy, and AUC-ROC."""
    is_pit = labels == 1
    is_neg = labels == 0
    
    pit_embs = embeddings[is_pit]
    neg_embs = embeddings[is_neg]
    
    if len(pit_embs) < 2 or len(neg_embs) < 1:
        return {"sim_pp": 0.0, "sim_pn": 0.0, "separation": 0.0, "accuracy": 0.0, "auc": 0.5}
        
    # 1. Cosine similarities (dot product since embeddings are already L2 normalised)
    sim_matrix_pp = np.matmul(pit_embs, pit_embs.T)
    np.fill_diagonal(sim_matrix_pp, np.nan)
    sim_pp = np.nanmean(sim_matrix_pp)
    
    sim_matrix_pn = np.matmul(pit_embs, neg_embs.T)
    sim_pn = np.mean(sim_matrix_pn)
    
    separation = sim_pp - sim_pn
    
    # 2. Hardest-pair Accuracy (is nearest class-mate closer than nearest negative?)
    dist_pp = 1.0 - np.matmul(pit_embs, pit_embs.T)
    np.fill_diagonal(dist_pp, np.inf)
    dist_pn = 1.0 - np.matmul(pit_embs, neg_embs.T)
    
    hardest_pos = np.min(dist_pp, axis=1)
    hardest_neg = np.min(dist_pn, axis=1)
    accuracy = np.mean(hardest_pos < hardest_neg) * 100.0
    
    # 3. AUC-ROC score using nearest-centroid prototype distance
    # Anchor score = cosine similarity to the mean validation Pit embedding
    mean_pit_emb = np.mean(pit_embs, axis=0)
    mean_pit_emb = mean_pit_emb / (np.linalg.norm(mean_pit_emb) + 1e-8)
    
    scores = np.dot(embeddings, mean_pit_emb)
    auc = roc_auc_score(labels, scores)
    
    return {
        "sim_pp": sim_pp,
        "sim_pn": sim_pn,
        "separation": separation,
        "accuracy": accuracy,
        "auc": auc
    }


def get_val_transforms() -> transforms.Compose:
    """Validation transforms matching training script."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Plotting functions ──

def plot_embeddings(embeddings_dict, labels, save_path):
    """Plot smallest (64d) and largest (384d) Matryoshka embeddings side-by-side using UMAP/t-SNE."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=150)
    dims_to_plot = [64, 384]
    
    colors = ["#e74c3c" if l == 0 else "#2ecc71" for l in labels]
    
    for idx, dim in enumerate(dims_to_plot):
        embs = embeddings_dict[dim]
        
        if HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
            proj = reducer.fit_transform(embs)
            title_prefix = "UMAP"
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels) - 1))
            proj = reducer.fit_transform(embs)
            title_prefix = "t-SNE"
            
        ax = axes[idx]
        ax.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.85, edgecolors="w", s=55, linewidth=0.5)
        
        ax.set_title(f"{title_prefix} Projection ({dim}d space)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Component 1", fontsize=10)
        ax.set_ylabel("Component 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ecc71", edgecolor="w", label="Lunar Pit"),
            Patch(facecolor="#e74c3c", edgecolor="w", label="Negative Surface")
        ]
        ax.legend(handles=legend_elements, loc="best", framealpha=0.9, fontsize=10)
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved projection figure to {save_path}")


def plot_saliency_and_attention(model, dataset, device, save_path):
    """Plot original NAC tile alongside its gradient saliency map and self-attention overlay."""
    # Find first Pit index
    sample_idx = None
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if label == 1:
            sample_idx = idx
            break
            
    if sample_idx is None:
        logger.warning("No Pit sample found in dataset for local map overlay.")
        return
        
    img_tensor, label = dataset[sample_idx]
    
    # De-normalise for visual display
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img_np = img_tensor.numpy()
    img_vis = np.clip(img_np * std + mean, 0, 1)[0]
    
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    # Compute saliency with grad enabled
    torch.set_grad_enabled(True)
    saliency = compute_saliency_map(model, img_batch)
    torch.set_grad_enabled(False)
    
    # Compute attention maps
    attn = extract_last_block_attention(model, img_batch)
    attns = cls_attention_maps(attn, GRID_SIZE)
    mean_attn = attns.mean(dim=0).cpu().numpy()
    
    # Interpolate maps to match source resolution
    saliency_res = np.array(Image.fromarray(saliency).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC))
    attn_res = np.array(Image.fromarray(mean_attn).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC))
    
    # Render figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=150)
    
    # Original image
    axes[0].imshow(img_vis, cmap="gray")
    axes[0].set_title("Input NAC Pit Tile", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    # Saliency overlay
    axes[1].imshow(img_vis, cmap="gray")
    axes[1].imshow(saliency_res, cmap="jet", alpha=0.5)
    axes[1].set_title("Gradient Saliency (Embedding Norm)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    
    # Attention overlay
    axes[2].imshow(img_vis, cmap="gray")
    axes[2].imshow(attn_res, cmap="magma", alpha=0.55)
    axes[2].set_title("DINOv3 Last Block Self-Attention Map", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved saliency and attention overlay figure to {save_path}")


def plot_bias_variance_tradeoff(train_metrics, val_metrics, save_path):
    """Plot the measured Bias-Variance trade-off proxy curves from HOLEModel."""
    dims = NEST_DIMS
    
    train_errors = [1.0 - train_metrics[d]["accuracy"] / 100.0 for d in dims]
    val_errors = [1.0 - val_metrics[d]["accuracy"] / 100.0 for d in dims]
    
    # Irreducible error represents baseline sensor/label noise (estimated at 5% for LRO NAC)
    irred_err = 0.05
    irreducible = [irred_err] * len(dims)
    
    # Bias^2 proxy is the train error minus irreducible error (non-negative)
    bias_sq = [max(0.0, err - irred_err) for err in train_errors]
    
    # Variance proxy is the generalization gap (val error - train error)
    variance = [max(0.0, val_err - train_err) for val_err, train_err in zip(val_errors, train_errors)]
    
    plt.figure(figsize=(9, 6), dpi=150)
    
    c_train = "#2980b9"  # blue
    c_val = "#e74c3c"    # red
    c_bias = "#f39c12"   # orange
    c_var = "#1abc9c"    # teal
    c_irred = "#7f8c8d"  # gray
    
    plt.plot(dims, val_errors, marker="o", label="Validation Error (Total Error)", color=c_val, linewidth=3)
    plt.plot(dims, train_errors, marker="s", label="Training Error", color=c_train, linewidth=2.5)
    plt.plot(dims, bias_sq, marker="^", label=r"Bias$^2$ Proxy (Train Error - Irred)", color=c_bias, linewidth=2, linestyle="--")
    plt.plot(dims, variance, marker="d", label="Variance Proxy (Val - Train Gap)", color=c_var, linewidth=2, linestyle="--")
    plt.plot(dims, irreducible, label="Assumed Irreducible Error (5%)", color=c_irred, linewidth=1.5, linestyle=":")
    
    plt.title("Measured Bias-Variance Trade-off across Matryoshka Dimensions", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Matryoshka Embedding Dimension", fontsize=11)
    plt.ylabel("Error Rate (0.0 to 1.0)", fontsize=11)
    plt.xticks(dims)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved measured Bias-Variance trade-off plot to {save_path}")


def plot_matryoshka_decay(metrics_dict, save_path):
    """Plot absolute performance and relative retention across Matryoshka dimensions."""
    dims = NEST_DIMS
    
    accuracies = [metrics_dict[d]["accuracy"] for d in dims]
    aucs = [metrics_dict[d]["auc"] for d in dims]
    separations = [metrics_dict[d]["separation"] for d in dims]
    
    base_acc = accuracies[-1]
    base_auc = aucs[-1]
    base_sep = separations[-1]
    
    rel_acc = [a / base_acc * 100.0 if base_acc > 0 else 0.0 for a in accuracies]
    rel_auc = [a / base_auc * 100.0 if base_auc > 0 else 0.0 for a in aucs]
    rel_sep = [s / base_sep * 100.0 if base_sep > 0 else 0.0 for s in separations]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    
    c_acc = "#2ecc71"  # vibrant green
    c_auc = "#3498db"  # soft blue
    c_sep = "#9b59b6"  # rich purple
    
    # Left subplot: Absolute Metrics
    ax1 = axes[0]
    ax1.plot(dims, accuracies, marker="o", linewidth=2.5, color=c_acc, label=f"Accuracy (384d Base: {base_acc:.1f}%)")
    ax1.plot(dims, [a * 100 for a in aucs], marker="s", linestyle="--", linewidth=2, color=c_auc, label=f"AUC-ROC x 100 (384d Base: {base_auc*100:.1f}%)")
    ax1.plot(dims, [s * 100 for s in separations], marker="^", linestyle=":", linewidth=2, color=c_sep, label=f"Separation x 100 (384d Base: {base_sep*100:.1f})")
    
    ax1.set_title("Absolute Performance vs. Dimension", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Matryoshka Embedding Dimension", fontsize=10)
    ax1.set_ylabel("Value (%) / Score x 100", fontsize=10)
    ax1.set_xticks(dims)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
    
    # Right subplot: Relative Retention
    ax2 = axes[1]
    ax2.plot(dims, rel_acc, marker="o", linewidth=2.5, color=c_acc, label="Accuracy Retention")
    ax2.plot(dims, rel_auc, marker="s", linestyle="--", linewidth=2, color=c_auc, label="AUC-ROC Retention")
    ax2.plot(dims, rel_sep, marker="^", linestyle=":", linewidth=2, color=c_sep, label="Separation Retention")
    
    ax2.axhline(100.0, color="gray", linestyle="-.", alpha=0.5)
    
    ax2.set_title("Relative Performance Retention (vs. 384d baseline)", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Matryoshka Embedding Dimension", fontsize=10)
    ax2.set_ylabel("Retention Score (%)", fontsize=10)
    ax2.set_xticks(dims)
    ax2.set_ylim(40, 105)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Matryoshka decay plot to {save_path}")


# ── Accumulating embeddings per split ──

def evaluate_split(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_embeddings = {dim: [] for dim in NEST_DIMS}
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            heads = model(images)  # returns L2 normalised head embeddings
            for dim_idx, dim in enumerate(NEST_DIMS):
                all_embeddings[dim].append(heads[dim_idx].cpu().numpy())
            all_labels.append(labels.numpy())
            
    # Concatenate
    for dim in NEST_DIMS:
        all_embeddings[dim] = np.concatenate(all_embeddings[dim], axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_embeddings, all_labels


# ── Main ──

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    
    os.makedirs("figures", exist_ok=True)
    
    # 1. Load model
    try:
        model = load_trained_model(device)
    except FileNotFoundError as e:
        logger.error(f"Could not load weights: {e}")
        sys.exit(1)
        
    # 2. Load Hugging Face dataset from disk
    logger.info(f"Loading Hugging Face dataset from disk: {DATA_DIR}")
    from datasets import load_from_disk
    ds = load_from_disk(DATA_DIR)
    
    from src.data_loader import LunarPitHFDataset
    train_dataset = LunarPitHFDataset(ds["train"], transform=get_val_transforms())
    val_dataset = LunarPitHFDataset(ds["validation"], transform=get_val_transforms())
    
    # 3. Accumulate training set embeddings
    logger.info("Computing training set embeddings...")
    train_embs, train_labels = evaluate_split(model, train_dataset, device)
    
    # 4. Accumulate validation set embeddings
    logger.info("Computing validation set embeddings...")
    val_embs, val_labels = evaluate_split(model, val_dataset, device)
    
    # 5. Compute metrics for both splits
    train_metrics_by_dim = {}
    val_metrics_by_dim = {}
    for dim in NEST_DIMS:
        train_metrics_by_dim[dim] = compute_metrics(train_embs[dim], train_labels)
        val_metrics_by_dim[dim] = compute_metrics(val_embs[dim], val_labels)
        
    # 6. Print comparative metrics table
    print("\n" + "=" * 100)
    print(f"{'MATRYOSHKA DIMENSION METRICS (TRAIN VS VALIDATION SET)':^100}")
    print("=" * 100)
    print(f"{'Dim':<5} | {'Train Acc':<10} | {'Val Acc':<10} | {'Train AUC':<10} | {'Val AUC':<10} | {'Train Sep':<10} | {'Val Sep':<10}")
    print("-" * 100)
    for dim in NEST_DIMS:
        t_m = train_metrics_by_dim[dim]
        v_m = val_metrics_by_dim[dim]
        print(
            f"{dim:<5d} | "
            f"{t_m['accuracy']:<10.1f} | "
            f"{v_m['accuracy']:<10.1f} | "
            f"{t_m['auc']:<10.4f} | "
            f"{v_m['auc']:<10.4f} | "
            f"{t_m['separation']:<10.4f} | "
            f"{v_m['separation']:<10.4f}"
        )
    print("=" * 100 + "\n")
    
    # 7. Plot Matryoshka decay curves (using validation metrics)
    logger.info("Plotting Matryoshka decay curves...")
    plot_matryoshka_decay(val_metrics_by_dim, "figures/matryoshka_decay.png")
    
    # 8. Plot measured Bias-Variance trade-off curves
    logger.info("Plotting measured Bias-Variance trade-off curves...")
    plot_bias_variance_tradeoff(train_metrics_by_dim, val_metrics_by_dim, "figures/bias_variance.png")
    
    # 9. Visualise embedding space clusters (using validation embeddings)
    logger.info("Reducing dimensions and plotting validation space clusters...")
    plot_embeddings(val_embs, val_labels, "figures/validation_umap.png")
    
    # 10. Saliency and Attention overlays (using validation dataset)
    logger.info("Computing attention maps and saliency map for Pit example...")
    plot_saliency_and_attention(model, val_dataset, device, "figures/saliency_attention.png")
    
    logger.info("Evaluation and visualization completed successfully!")


if __name__ == "__main__":
    main()
