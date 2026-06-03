import mlflow
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from src import (
    get_dataloaders, 
    DinoExtractor, 
    MatryoshkaTripletLoss, 
    train_one_epoch, 
    validate_epoch
)
from src.utils import get_logger

logger = get_logger(__name__)


def get_dino_transforms():
    fill_value = 0.5 

    return transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 360), fill=fill_value),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
def debug_model_input(dataloader, num_samples=4, denormalize=True):
    images, labels = next(iter(dataloader))
    images = images.cpu()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    
    # Fix: Use .reshape() for numpy arrays
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    for i in range(num_samples):
        img = images[i].numpy() # This is a numpy array (3, H, W)
        
        if denormalize:
            # Broadcast operation
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
        display_img = img[0] 
        
        axes[0, i].imshow(display_img, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Label: {labels[i].item()} (Pit)" if labels[i] == 1 else "Neg")
        axes[0, i].axis('off')
        
        axes[1, i].hist(display_img.flatten(), bins=50, range=(0, 1), color='purple', alpha=0.7)
        axes[1, i].set_title("Pixel Distribution")
        axes[1, i].set_xlim(0, 1)

    plt.tight_layout()
    return fig


def main():
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 1e-4
    MARGIN = 1.0
    MATRYOSHKA_DIMS = [384]
    LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-data", action="store_true", help="Only visualize data and then exit")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    BATCH_SIZE = 16
    EPOCHS = args.epochs

    mlflow.set_experiment("Lunar_Pit_LoRA_Training")
    with mlflow.start_run(run_name="Dataset_Check" if args.debug_data else "Production_Run"):
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LR,
            "margin": MARGIN,
            "matryoshka_dims": str(MATRYOSHKA_DIMS),
            "matroyshka_weights": str(LOSS_WEIGHTS),
            "model": "dinov3_vits16_lvd"
        })

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Starting training on device: {device}")

        train_loader, val_loader = get_dataloaders(
            data_dir="data/processed/dataset", 
            batch_size=BATCH_SIZE, 
            transform=get_dino_transforms()
        )

        if args.debug_data:
            logger.info("Running in DEBUG mode. Visualizing input batch...")
            fig = debug_model_input(dataloader=train_loader, num_samples=4, denormalize=True)
            
            mlflow.log_figure(fig, "data_check/input_samples.png")
            
            plt.show() 
            
            logger.info("Debug finish. Exiting without training.")
            return

        weights = "models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth"
        model = DinoExtractor(weights_path=weights, model_size='vits16').to(device)
        
        criterion = MatryoshkaTripletLoss(nest_dims=MATRYOSHKA_DIMS, margin=MARGIN, weights=LOSS_WEIGHTS)
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=LR, weight_decay=1e-2)
        
        warmup_epochs = 5
        scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs)

        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

        logger.info("Starting training loop...")
        best_sep = -1.0

        for epoch in range(1, EPOCHS + 1):
            train_loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, MARGIN, epoch)
            val_loss, val_sep, val_acc, sim_pp, sim_pn = validate_epoch(model, val_loader, criterion, device, MARGIN)
            
            scheduler.step()

            metrics = {
                "loss": train_loss,
                "val_loss": val_loss,
                "sep": val_sep,
                "acc": val_acc,
                "sim_pp": sim_pp,
                "sim_pn": sim_pn
            }

            logger.info(
                f"Ep {epoch} | " +
                " | ".join([
                    f"{k}: {v:.4f}" if k != "acc" else f"{k}: {v:.2f}%"
                    for k, v in metrics.items()
                ])
            )

            mlflow.log_metrics(metrics, step=epoch)
            if val_sep > best_sep:
                best_sep = val_sep
                logger.info("--> New Best Separation! Saving Adapter...")
                model.save_adapter("models/lunar_dinov3_lora/standard_lora")
                
                # Log only files, ignoring the .git directory to avoid permission errors
                import os
                adapter_dir = "models/lunar_dinov3_lora/standard_lora"
                for f in os.listdir(adapter_dir):
                    f_path = os.path.join(adapter_dir, f)
                    if os.path.isfile(f_path):
                        mlflow.log_artifact(f_path, artifact_path="lora_weights")

        logger.info("Training Finished!")

if __name__ == "__main__":
    if mlflow.active_run():
        mlflow.end_run()
    main()