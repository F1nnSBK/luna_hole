
import mlflow
import torch
from torch.cpu import is_available
from torchvision import transforms
from src import get_dataloaders, DinoExtractor
from src.utils import get_logger

logger = get_logger(__name__)


def get_dino_transforms():
    return transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])



def main():
    mlflow.set_experiment("Lunar_Pit_Dataset_Preparation")
    with mlflow.start_run(run_name="Dataset_Initial_Check"):
        train_loader, val_loader = get_dataloaders(transform=get_dino_transforms())

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)

        mlflow.log_param("train_size", n_train)
        mlflow.log_param("val_size", n_val)
        mlflow.log_param("img_size", 256)

        images, labels = next(iter(train_loader))
        logger.info(f"Batch shape: {images.shape}")

        weights = "models/meta/dinov3_vits16_pretrain_lvd.pth"
        model = DinoExtractor(weights_path=weights, model_size='vits16')

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        model = model.to(device)

        with torch.no_grad():
            embeddings = model(images.to(device))
    

if __name__ == "__main__":
    main()