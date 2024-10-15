import sys, os

from torchvision.models import vit_b_16
import torch.nn as nn
import torch

from surgeon_pytorch import Extract, get_nodes

from PIL import Image
import torchvision.transforms as transforms
import os

import numpy as np

from matplotlib import pyplot as plt
import cv2

from captum.attr import LayerGradCam, LayerAttribution

external_repo_path = os.path.abspath('external/transformer_explainability')
if external_repo_path not in sys.path:
    sys.path.append(external_repo_path)

from external.transformer_explainability.baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from external.transformer_explainability.baselines.ViT.ViT_explanation_generator import LRP





checkpoint = torch.load("logs/vit_waterbirds.pth", map_location=torch.device('cpu'), weights_only=False)
model2 = vit_LRP(pretrained=True, num_classes=2)

new_checkpoint = {}

for k, v in checkpoint.items():
    new_key = k
    new_key = new_key.replace('class_token', 'cls_token')
    new_key = new_key.replace('conv_proj', 'patch_embed.proj')
    new_key = new_key.replace('encoder.pos_embedding', 'pos_embed')
    new_key = new_key.replace('encoder.ln.weight', 'norm.weight')
    new_key = new_key.replace('encoder.ln.bias', 'norm.bias')
    new_key = new_key.replace('heads.head.weight', 'head.weight')
    new_key = new_key.replace('heads.head.bias', 'head.bias')

    new_key = new_key.replace('encoder.layers.encoder_layer_', 'blocks.')
    new_key = new_key.replace('.ln_1', '.norm1')
    new_key = new_key.replace('.ln_2', '.norm2')
    new_key = new_key.replace('.self_attention.in_proj_weight', '.attn.qkv.weight')
    new_key = new_key.replace('.self_attention.in_proj_bias', '.attn.qkv.bias')
    new_key = new_key.replace('.self_attention.out_proj.weight', '.attn.proj.weight')
    new_key = new_key.replace('.self_attention.out_proj.bias', '.attn.proj.bias')
    new_key = new_key.replace('.mlp.0', '.mlp.fc1')
    new_key = new_key.replace('.mlp.3', '.mlp.fc2')

    new_checkpoint[new_key] = v

model2.load_state_dict(new_checkpoint)
model2 = model2.cuda()

attribution_generator = LRP(model2)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def tensorize(img_path):
    image = Image.open(img_path).convert('RGB')
    return preprocess(image)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization(original_image, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())



    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis =  np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

img = tensorize('notebooks/data/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg').cuda()
img2 = tensorize('notebooks/data/054.Blue_Grosbeak/Blue_Grosbeak_0004_14988.jpg').cuda()
img3 = tensorize('notebooks/data/136.Barn_Swallow/Barn_Swallow_0014_130403.jpg').cuda()
print(img.shape)

out = model2(img.unsqueeze(0))
print(out)

image_np = img.permute(1, 2, 0).cpu().numpy()
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

# Define output directory
output_dir = "logs"
os.makedirs(output_dir, exist_ok=True)

# Generate visualization for the class predicted by the model (e.g., 'cat')
cat = generate_visualization(img3)
cat_output_path = os.path.join(output_dir, "cat_visualization.png")
cv2.imwrite(cat_output_path, cat)

# Generate visualization for class index 1 (e.g., 'dog')
dog = generate_visualization(img3, class_index=1)
dog_output_path = os.path.join(output_dir, "dog_visualization.png")
cv2.imwrite(dog_output_path, dog)

print(f"Visualization for 'cat' saved at: {cat_output_path}")
print(f"Visualization for 'dog' saved at: {dog_output_path}")
