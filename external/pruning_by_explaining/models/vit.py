from torchvision.models.vision_transformer import (
    vit_b_16,
    ViT_B_16_Weights,
)
import torch
import torch.nn as nn


def load_vit_b_16(checkpoint_path, num_classes=1000):
    """
    Loading ViT-B-16 from torchvision models

    Args:
        checkpoint_path (str): checkpoint-path of the model if given

    Returns:
        model (torch.nn.module): the model
    """
    weights = None
#    if checkpoint_path is None:
#        weights = ViT_B_16_Weights.IMAGENET1K_V1
#    else:
#        weights = torch.load(checkpoint_path)
#    model = vit_b_16(weights=weights)
#    del weights
#    model.eval()

    if checkpoint_path is None:
        # Load pretrained weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        model.heads.head = nn.Linear(768, num_classes)  # Replace head for binary classification
    else:
        # Initialize model and load checkpoint
        model = vit_b_16()
        model.heads.head = nn.Linear(768, num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(checkpoint, strict=False)
        print("vit checkpoint loaded")

    return model
