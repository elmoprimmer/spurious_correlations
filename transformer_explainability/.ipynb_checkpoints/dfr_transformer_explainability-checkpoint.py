import sys, os

import torch

from PIL import Image
import torchvision.transforms as transforms
import os

import numpy as np
import cv2

import argparse

external_repo_path = os.path.abspath('../../deep_feature_reweighting/external/transformer_explainability')

if external_repo_path not in sys.path:
    sys.path.append(external_repo_path)

from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from baselines.ViT.ViT_explanation_generator import LRP

parser = argparse.ArgumentParser(description="Visualize 2 class transformer")
parser.add_argument("--img_path", type=str, default=None, help="Path to image to visualize")
parser.add_argument("--output_dir", type=str, default=None, help="Dir to save visualizations in. Will generate 2 "
                                                                 "images by default")
parser.add_argument("--chkpt_path", type=str, default=None, help="Chkpoint to use for model")
parser.add_argument("--only_idx0", action='store_true', help="Only make img for class 0")
parser.add_argument("--only_idx1", action='store_true', help="Only make img for class 1")
parser.add_argument("--output_filename_prefix", type=str, default=None, help="What to add to start of output file")

args = parser.parse_args()

checkpoint = torch.load(args.chkpt_path, map_location=torch.device('cpu'), weights_only=False)
model = vit_LRP(pretrained=True, num_classes=2)

new_checkpoint = {}

# fix ckpt to be suitable for their model
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

model.load_state_dict(new_checkpoint)
model = model.cuda()

attribution_generator = LRP(model)

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
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(),
                                                                 method="transformer_attribution",
                                                                 index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (
            transformer_attribution.max() - transformer_attribution.min())
    # flip values
    transformer_attribution = 1 - transformer_attribution

    image_transformer_attribution = original_image.permute(1, 2, 0).data.cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (
            image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


img = tensorize(args.img_path).cuda()

os.makedirs(args.output_dir, exist_ok=True)

img_name = os.path.basename(args.img_path).split('.')[0]
# Generate visualization for the class predicted by the model (e.g., 'land')
cat = generate_visualization(img, class_index=0)
land_filename = img_name + "_0.png"
cat_output_path = os.path.join(args.output_dir, land_filename)
cv2.imwrite(cat_output_path, cat)

# Generate visualization for class index 1 (e.g., 'water')
dog = generate_visualization(img, class_index=1)
water_filename = img_name + "_1.png"
dog_output_path = os.path.join(args.output_dir, water_filename)
cv2.imwrite(dog_output_path, dog)

print(f"Visualization for '0' saved at: {cat_output_path}")  # cat
print(f"Visualization for '1' saved at: {dog_output_path}")  # dog
