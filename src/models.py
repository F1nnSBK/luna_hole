import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from .utils import get_logger

logger = get_logger(__name__)
L_RANK = 8

class DinoExtractor(nn.Module):
    def __init__(self, weights_path="models/meta/dinov3_vits16_pretrain_lvd.pth", model_name="vits16_lvd", model_size="vits16", freeze=True):
        super().__init__()
        self.model_size = model_size
        self.model_name = model_name
        logger.info(f"Load DINO model: dino_{model_name}_{model_size}...")

        try:
            self.backbone = torch.hub.load("facebookresearch/dinov3", f"dinov3_{model_size}", pretrained=False)
            state_dict = torch.load(weights_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
                
            self.backbone.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.error(f"Error while loading Model: {e}")
            raise
        
        lora_config = LoraConfig(
            r=L_RANK,
            lora_alpha=L_RANK * 2,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none"
        )

        self.backbone = get_peft_model(self.backbone, lora_config)
        self.backbone.print_trainable_parameters()

    def forward(self, x):
        return self.backbone(x)

    def get_embedding_dim(self):
        dims = {
            'vits16': 384,
            'vitb16': 768,
            'vitl16': 1024,
            'vitg16': 1536
        }
        return dims.get(self.model_size, 0)