---
library_name: peft
tags:
- lora
- safetensors
- dinov3
- lunar-pits
- computer-vision
- image-feature-extraction
base_model: facebookresearch/dinov3
license: apache-2.0
pipeline_tag: image-feature-extraction
---

# Lunar DINOv3 LoRA (F1nnSBK/lunar-dinov3-lora)

This model card describes the LoRA adapter fine-tuned on top of the **DINOv3 (ViT-S/16)** vision backbone. The model maps lunar surface terrain tiles (represented as normalized Narrow Angle Camera (NAC) heights/pixels) to a metric embedding space optimized for distinguishing lunar pits from surrounding volcanic plains and negative control features.

## Model Details

- **Developed by:** F1nnSBK
- **Model type:** PEFT (LoRA) Adapter for DINOv3 Vision Transformer
- **Language(s):** English
- **License:** Apache 2.0
- **Base model:** `facebookresearch/dinov3_vits16` (fine-tuned from the pretrained `dinov3_vits16_pretrain_lvd.pth` weights)
- **PEFT Configuration:**
  - `peft_type`: `LORA`
  - `r (rank)`: `32`
  - `lora_alpha`: `32`
  - `target_modules`: `["qkv", "proj", "fc2", "fc1"]`
  - `lora_dropout`: `0.1`
  - `bias`: `"none"`

---

## Uses

### Direct Use
This model is designed to be loaded onto a pretrained DINOv3 backbone to extract 384-dimensional embeddings of lunar surface images. These embeddings are structurally grouped using triplet distance, allowing:
- Automatic clustering and cataloging of lunar pits and volcanic depressions.
- Feature similarity searches across newly acquired lunar NAC images.

### Out-of-Scope Use
- Not intended for general-purpose terrestrial image classification.
- Not tested for real-time hazard detection during automated spacecraft landings.

---

## How to Get Started

Use the code snippet below to load the base DINOv3 backbone and apply the PEFT/LoRA adapter from Hugging Face:

```python
import torch
import torch.nn as nn
from peft import PeftModel

# 1. Initialize base DINOv3 ViT-S/16 backbone
base_model = torch.hub.load("facebookresearch/dinov3", "dinov3_vits16", pretrained=False)

# 2. Load the base pre-trained weights (pretrain_lvd)
# (Ensure you download 'dinov3_vits16_pretrain_lvd.pth' to your local models directory)
state_dict = torch.load("models/meta/dinov3/dinov3_vits16_pretrain_lvd.pth", map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
base_model.load_state_dict(state_dict, strict=True)

# 3. Load the LoRA adapter from Hugging Face
model = PeftModel.from_pretrained(
    base_model, 
    "F1nnSBK/lunar-dinov3-lora", 
    adapter_name="pit_adapter"
)
model.eval()

# 4. Extract embeddings
# Input tensor shape: [batch_size, 3, 224, 224]
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    embeddings = model(dummy_input)
print("Embedding shape:", embeddings.shape)  # Should output: [1, 384]
```

---

## Training Details

### Training Data
The model was trained on a dataset of curated Narrow Angle Camera (NAC) tiles, containing:
- **Pits**: Volcanic pit crater coordinates.
- **Negatives**: Control volcanic regions, shallow craters, and shadow features.
- Data splits are partitioned dynamically using a **Group-Split based on NAC strip IDs** to prevent any data leakage between training and validation groups.

### Training Procedure
- **Loss Function:** Matryoshka Triplet Loss (dimensions optimized at `384`).
- **Mining Strategy:** Semi-Hard Triplet Mining.
- **Optimizer:** `AdamW` with learning rate of `1e-4` and weight decay `1e-2`.
- **Scheduler:** Sequential linear warmup (5 epochs) followed by cosine annealing.

---

## Evaluation & Saliency Maps

The fine-tuned model exhibits strong translation equivariance and distinct feature localization compared to the zero-shot DINOv3 baseline.

### 1. Lunar Pit Saliency Comparison
The LoRA fine-tuning refocuses the backbone features directly onto the structural elements of lunar pits.

![Pit Saliency](luna_fig1_pits_lora_r32a32_qkv-proj-fc2-fc1.svg)

### 2. Control (Non-Pit) Saliency Comparison
For non-pit control regions, the saliency maps show negligible drift and focus on background patterns.

![Non-Pit Saliency](luna_fig2_nonpits_lora_r32a32_qkv-proj-fc2-fc1.svg)

### 3. Translation Equivariance Shift Test
To ensure stability, the model was tested under a diagonal pixel shift (+$50\text{px}$). The observed activation peak shift matches the expected shift of $\approx 70.7\text{px}$ with minimal error.

![Equivariance Qual](luna_fig_equivariance_qual_lora_r32a32_qkv-proj-fc2-fc1.svg)

---

## Local Repository Setup & Scripts

If you have cloned this repository locally, you can use the following scripts to reproduce training, split datasets, and generate evaluation figures.

### Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Scripts
- **Dataset Generation**: Splits raw PNG/NPY files under `data/processed/dataset/` into `train` and `test` subfolders:
  ```bash
  python create_dataset.py
  ```
- **Stats Builder**: Generates normalization percentiles:
  ```bash
  python build_stats.py
  ```
- **Training**: Runs training and logs to MLflow:
  ```bash
  python main.py --epochs 20
  ```
- **Equivariance Evaluation**: Generates `luna_fig_equivariance_qual_lora...` and `luna_fig_equivariance_quant_lora...` figures:
  ```bash
  python luna_acid.py
  ```
- **Saliency Differences**: Generates the comparison figures showing fine-tuned vs. zero-shot saliency differences:
  ```bash
  python luna_diff.py
  ```
