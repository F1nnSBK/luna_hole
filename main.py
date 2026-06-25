"""
main.py — HOLE Training Entry Point
=====================================
Trains a HOLEModel (DINOv3 LoRA + Matryoshka Projection Head) using the
DIVE-inspired loss: SelfLimitingHingeLoss + HeadwiseNTXentLoss.

Usage
-----
    python main.py --epochs 20
    python main.py --epochs 5 --debug-data
    python main.py --epochs 20 --loss legacy   # use original MatryoshkaTripletLoss
"""

import argparse
import os

import mlflow
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src import (
    HOLEModel,
    DIVELossConfig,
    MatryoshkaDIVELoss,
    MatryoshkaTripletLoss,
    get_dataloaders,
    train_one_epoch,
    validate_epoch,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ── Hyperparameters ───────────────────────────────────────────────────────────

BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-4
WEIGHT_DECAY  = 1e-2
MARGIN        = 0.3          # hinge margin (cosine distance space)
WARMUP_EPOCHS = 5
NEST_DIMS     = [64, 128, 256, 384]
LORA_RANK     = 32

# DIVE loss hyperparameters
LAMBDA_NTXENT   = 0.5
NTXENT_TEMP     = 0.07

WEIGHTS_PATH    = "models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth"
ADAPTER_PATH    = "models/lunar_dinov3_lora/standard_lora"
MODEL_SIZE      = "vits16"


# ── Transforms ────────────────────────────────────────────────────────────────

def get_train_transforms() -> transforms.Compose:
    """DINOv3 training transforms with MAE-light augmentation (RandomErasing)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 360), fill=0.5),
        transforms.RandomErasing(p=0.4, scale=(0.05, 0.20), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms() -> transforms.Compose:
    """Validation transforms — deterministic, no augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── Debug helper ──────────────────────────────────────────────────────────────

def debug_model_input(dataloader, num_samples: int = 4) -> plt.Figure:
    """Visualise a batch of input tiles for sanity-checking the data pipeline."""
    images, labels = next(iter(dataloader))
    images = images.cpu()
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    for i in range(num_samples):
        img = images[i].numpy()
        img = np.clip(img * std + mean, 0, 1)[0]  # grayscale channel
        axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title("Pit" if labels[i] == 1 else "Neg")
        axes[0, i].axis("off")
        axes[1, i].hist(img.flatten(), bins=50, range=(0, 1), color="steelblue", alpha=0.8)
        axes[1, i].set_title("Pixel dist.")
        axes[1, i].set_xlim(0, 1)

    plt.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HOLE Training")
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--debug-data", action="store_true")
    parser.add_argument(
        "--loss",
        choices=["dive", "legacy"],
        default="dive",
        help="'dive' = MatryoshkaDIVELoss (new). 'legacy' = MatryoshkaTripletLoss.",
    )
    args = parser.parse_args()

    epochs     = args.epochs
    batch_size = args.batch_size

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device} | Loss: {args.loss} | Epochs: {epochs}")

    # ── Data ──
    train_loader, val_loader = get_dataloaders(
        data_dir="data/hf_dataset",
        batch_size=batch_size,
        transform=get_train_transforms(),
    )

    if args.debug_data:
        fig = debug_model_input(train_loader)
        plt.savefig("figures/debug_input.png", dpi=100)
        logger.info("Debug figure saved to figures/debug_input.png")
        return

    # ── Model ──
    model = HOLEModel(
        weights_path=WEIGHTS_PATH,
        model_size=MODEL_SIZE,
        nest_dims=NEST_DIMS,
        lora_rank=LORA_RANK,
    ).to(device)

    # ── Loss ──
    if args.loss == "dive":
        dive_cfg = DIVELossConfig(
            nest_dims=NEST_DIMS,
            hinge_margin=MARGIN,
            ntxent_temperature=NTXENT_TEMP,
            lambda_ntxent=LAMBDA_NTXENT,
        )
        criterion = MatryoshkaDIVELoss(config=dive_cfg)
    else:
        criterion = MatryoshkaTripletLoss(
            nest_dims=NEST_DIMS,
            margin=MARGIN,
            weights=[0.5, 0.75, 1.0, 1.5],
        )

    # ── Optimiser + Scheduler ──
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS),
            CosineAnnealingLR(optimizer, T_max=epochs - WARMUP_EPOCHS),
        ],
        milestones=[WARMUP_EPOCHS],
    )

    # ── MLflow ──
    mlflow.set_experiment("HOLE_Training")
    run_name = f"DIVE_{args.loss}_ep{epochs}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "loss":           args.loss,
            "batch_size":     batch_size,
            "epochs":         epochs,
            "lr":             LR,
            "margin":         MARGIN,
            "nest_dims":      str(NEST_DIMS),
            "lambda_ntxent":  LAMBDA_NTXENT,
            "ntxent_temp":    NTXENT_TEMP,
            "lora_rank":      LORA_RANK,
            "model_size":     MODEL_SIZE,
        })

        best_sep = -float("inf")
        logger.info("Training started.")

        for epoch in range(1, epochs + 1):
            train_loss, train_subs = train_one_epoch(
                model, train_loader, optimizer, criterion, device, MARGIN, epoch
            )
            val_loss, val_sep, val_acc, sim_pp, sim_pn = validate_epoch(
                model, val_loader, criterion, device, MARGIN
            )
            scheduler.step()

            # Core metrics for MLflow
            metrics = {
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_sep":    val_sep,
                "val_acc":    val_acc,
                "sim_pp":     sim_pp,
                "sim_pn":     sim_pn,
            }
            # Add DIVE sub-metrics to MLflow
            metrics.update({f"train_{k}": v for k, v in train_subs.items()})
            mlflow.log_metrics(metrics, step=epoch)

            # Active triplet ratio from train_subs (DIVE only)
            active_str = ""
            if "hinge_active_ratio" in train_subs:
                active_str = f" | ρ={train_subs['hinge_active_ratio']:.2%}"

            logger.info(
                f"Ep {epoch:02d}/{epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"sep={val_sep:.4f} | acc={val_acc:.1f}%{active_str}"
            )

            if val_sep > best_sep:
                best_sep = val_sep
                model.save_adapter(ADAPTER_PATH)
                logger.info(f"  ↑ New best separation ({best_sep:.4f}) — adapter saved.")

                adapter_dir = ADAPTER_PATH
                for fname in os.listdir(adapter_dir):
                    fpath = os.path.join(adapter_dir, fname)
                    if os.path.isfile(fpath):
                        mlflow.log_artifact(fpath, artifact_path="lora_weights")

        logger.info(f"Training complete. Best val_sep={best_sep:.4f}")


if __name__ == "__main__":
    if mlflow.active_run():
        mlflow.end_run()
    main()