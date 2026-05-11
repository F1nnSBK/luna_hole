import mlflow
import torch
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
    return transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.RandomRotation(degrees=180),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])



def main():
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 1e-4
    MARGIN = 1.0
    MATRYOSHKA_DIMS = [64, 128, 256, 384]
    LOSS_WEIGHTS = [1.0, 1.0, 0.8, 0.5]

    mlflow.set_experiment("Lunar_Pit_LoRA_Training")
    with mlflow.start_run(run_name="Dataset_Initial_Check"):
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
        logger.info(f"Starte Training auf Device: {device}")

        train_loader, val_loader = get_dataloaders(
            data_dir="data/processed/dataset", 
            batch_size=BATCH_SIZE, 
            transform=get_dino_transforms()
        )

        weights = "models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth"
        model = DinoExtractor(weights_path=weights, model_size='vits16').to(device)
        
        criterion = MatryoshkaTripletLoss(nest_dims=MATRYOSHKA_DIMS, margin=MARGIN, weights=LOSS_WEIGHTS)
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=LR, weight_decay=1e-2)
        
        warmup_epochs = 5
        scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs)

        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])

        logger.info("Starte Trainings-Loop...")
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
                model.save_adapter("models/best_lora_pit_model")
                
                mlflow.log_artifacts("models/best_lora_pit_model", artifact_path="lora_weights")

        logger.info("Training Finished!")

if __name__ == "__main__":
    if mlflow.active_run():
        mlflow.end_run()
    main()