import mlflow
from src import get_dataloaders
from src.utils import get_logger

logger = get_logger(__name__)

def main():
    mlflow.set_experiment("Lunar_Pit_Dataset_Preparation")
    with mlflow.start_run(run_name="Dataset_Initial_Check"):
        train_loader, val_loader = get_dataloaders()

        n_train = len(train_loader.dataset)
        n_val = len(val_loader.dataset)

        mlflow.log_param("train_size", n_train)
        mlflow.log_param("val_size", n_val)
        mlflow.log_param("img_size", 256)

        images, labels = next(iter(train_loader))
        logger.info(f"Batch shape: {images.shape}")

if __name__ == "__main__":
    main()