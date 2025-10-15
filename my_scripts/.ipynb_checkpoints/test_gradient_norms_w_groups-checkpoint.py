import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_utils import evaluate_gradients
from ISIC_ViT.isic_data import ISICDataset
from external.dfr.wb_data import WaterBirdsDataset

import argparse

import torchvision
import torch
from torchvision.models import vit_b_16
import torch.nn as nn

parser = argparse.ArgumentParser(description="Avg norms of gradients accross all neurons "
                                             "(only weights) for a given group")
parser.add_argument(
    "--data_dir", type=str,
    default=None,
    help="Data directory")
parser.add_argument(
    "--metadata_csv", type=str,
    default=None,
    help="Metadata csv")
parser.add_argument(
    "--split", type=str,
    default=None,
    help="train, test or val")
parser.add_argument(
    "--batch_size", type=int,
    default=32,
    help="batch size for dataloader")
parser.add_argument(
    "--num_workers", type=int,
    default=4,
    help="num of workers for dataloader")
parser.add_argument(
    "--model_path", type=str,
    default=None,
    help="path to model checkpoint")
parser.add_argument(
    "--dataset", type=str,
    default="isic",
    help="are we using isic or waterbirds")
parser.add_argument(
    "--group", type=int,
    default=0,
    help="which group to use")

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if args.dataset == "isic":
    dataset = ISICDataset(basedir=args.data_dir,
                          csv_file=args.metadata_csv,
                          transform=transform,
                          split=args.split)
if args.dataset == "waterbirds":
    dataset = WaterBirdsDataset(basedir=args.data_dir,
                                transform=transform,
                                split=args.split)

loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.batch_size, 'pin_memory': True}
loader = torch.utils.data.DataLoader(dataset, shuffle=False, **loader_kwargs)

model = vit_b_16(weights=None)
model.heads.head = nn.Linear(model.heads.head.in_features, 2)
model.load_state_dict(torch.load(args.model_path, weights_only=False, map_location=device))
model.to(device)

metric = evaluate_gradients.AvgGradientByGroup(args.group)

results = evaluate_gradients.evaluate(model, loader, metric, args.group, device)
print("Group: ", results["group"])
print("N: ", results["n_total"])
top3 = []
for k, v in results["gradients"].items():
    if "head" in k:
        continue
    norms = v.cpu().numpy()
    for idx, norm in enumerate(norms):
        if len(top3) < 3:
            top3.append((k, idx, norm))
            top3.sort(key=lambda x: x[2], reverse=True)
        else:
            if norm > top3[-1][2]:
                top3[-1] = (k, idx, norm)
                top3.sort(key=lambda x: x[2], reverse=True)
print("Top 3 neurons (layer, index, norm):")
for layer, idx, norm in top3:
    print(f"Layer: {layer}, Index: {idx}, Norm: {norm}", results["gradients"][layer].size())
