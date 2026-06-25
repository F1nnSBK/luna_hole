from huggingface_hub import HfApi
import os
import sys

def upload_dataset():
    dataset_path = "data/processed/dataset"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found!")
        sys.exit(1)

    repo_id = "F1nnSBK/lunar-pits-dataset"
    print(f"Starting upload of '{dataset_path}' to HF Dataset repo '{repo_id}'...")

    api = HfApi()

    # Create the repository on HF if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True
        )
        print(f"Repository '{repo_id}' checked/created successfully.")
    except Exception as e:
        print(f"Warning during repo creation: {e}")
        print("Continuing with upload...")

    # Upload folder
    try:
        api.upload_folder(
            folder_path=dataset_path,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("\nDataset uploaded successfully to Hugging Face!")
        print(f"View it here: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"\nError during upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload_dataset()
