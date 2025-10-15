import math
import sys
import os

from functools import reduce

import re

import numpy as np
from matplotlib import pyplot as plt

from my_scripts.utils import visualize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, 'external', 'wdpruning'))
sys.path.append(os.path.join(ROOT, 'external', 'pruning_by_explaining'))

from external.wdpruning.vit_wdpruning import VisionTransformerWithWDPruning
from external.pruning_by_explaining.pxp import GlobalPruningOperations, ComponentAttibution, get_vit_composite, \
    ModelLayerUtils
from external.pruning_by_explaining.models import ModelLoader
from external.pruning_by_explaining.my_datasets import WaterBirds, WaterBirdDataset

from my_utils import evaluate, evaluate_gradients
from ISIC_ViT.isic_data import ISICDataset
from external.dfr.wb_data import WaterBirdsDataset

import argparse

import torchvision
import torch
from torchvision.models import vit_b_16
import torch.nn as nn

parser = argparse.ArgumentParser(description="Accuracy for given set for vit_b_16 w binary classification")
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
    "--gradient_norm_pruning", type=bool,
    default=False,
    help="are we pruning based on gradient norms")
parser.add_argument(
    "--n_groups", type=int,
    default=4,
    help="number of groups to prune based on")
parser.add_argument(
    "--n_prunable_neurons", type=int,
    default=4,
    help="number of neurons to prune (with largest gradient norms)")
parser.add_argument(
    "--wdpruning", type=bool, default=False, help="are we pruning a wdpruning model"
)
parser.add_argument(
    "--pruning_mask", type=str, default=None, help="are we applying a pruning mask? if so this is the path to it"
)
parser.add_argument(
    "--merge", type=str, default=None, help="what mask to merge the current mask with?"
)

args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if args.dataset == "isic":
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if args.dataset == "waterbirds":
    transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


if args.dataset == "isic":
    test_dataset = ISICDataset(basedir=args.data_dir,
                               csv_file=args.metadata_csv,
                               transform=transform,
                               split="test")
    val_dataset = ISICDataset(basedir=args.data_dir,
                              csv_file=args.metadata_csv,
                              transform=transform,
                              split="val")
    train_dataset = ISICDataset(basedir=args.data_dir,
                                csv_file=args.metadata_csv,
                                transform=transform,
                                split="train")
if args.dataset == "waterbirds":
    test_dataset = WaterBirdsDataset(basedir=args.data_dir,
                                     transform=transform,
                                     split="test")
    val_dataset = WaterBirdsDataset(basedir=args.data_dir,
                                    transform=transform,
                                    split="val")
    train_dataset = WaterBirdsDataset(basedir=args.data_dir,
                                      transform=transform,
                                      split="train")

loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': True}
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, **loader_kwargs)





def n_params(m): return sum(p.numel() for p in m.parameters())


if not args.wdpruning:
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    model.load_state_dict(torch.load(args.model_path, weights_only=False, map_location=device))
    model.to(device)
else:
    model = VisionTransformerWithWDPruning(num_classes=2,
                                           patch_size=16, embed_dim=768,
                                           depth=12, num_heads=12, mlp_ratio=4,
                                           head_pruning=True, fc_pruning=True)
    ckpt = torch.load(args.model_path, map_location=device)['model']
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    visualize(model, args.model_path)
    before = n_params(model)
    model._make_structural_pruning()
    after = n_params(model)
    print("true pruning ratio = ", after / before)


# ------------------------------------------------------------
# PXP
# ------------------------------------------------------------
def parse_head_mask(head_mask):
    """
    head_mask: OrderedDict, output of pxp
    returns   : dict {int(layer_index) -> list[int(head_idx)]}
    """
    pat = re.compile(r'encoder_layer_(\d+)')
    pruned = {}
    for name, heads in head_mask.items():
        m = pat.search(name)
        if m is None or len(heads) == 0:
            continue
        layer_idx = int(m.group(1))
        pruned[layer_idx] = heads.tolist()
    return pruned

@torch.no_grad()
def zero_vit_heads(model, prunable_head_dict):
    """
    Apply pxp pruning
    Zero Q/K/V and out-proj slices of given heads

    prunable_head_dict: dict {layer_idx -> list[int(head_id)]}
    """
    for idx, block in enumerate(model.encoder.layers):
        if idx not in prunable_head_dict:
            continue

        attn = getattr(block, "self_attention", None)

        if attn is None:
            raise RuntimeError(f"Could not find attention module in layer {idx}")

        hdim = attn.head_dim  # 64
        nheads = attn.num_heads  # 12
        edim = attn.embed_dim  # 768

        in_w = attn.in_proj_weight  # (3*edim, edim)
        in_b = getattr(attn, "in_proj_bias", None)
        out_w = attn.out_proj.weight  # (edim, edim)
        out_b = attn.out_proj.bias

        for h in prunable_head_dict[idx]:
            rows = slice(h * hdim, (h + 1) * hdim)

            # Q, K, V are stacked: [0:edim]   [edim:2*edim]   [2*edim:3*edim]
            for blk in range(3):
                r = slice(rows.start + blk * edim, rows.stop + blk * edim)
                in_w[r].zero_()
                if in_b is not None:
                    in_b[r].zero_()

            out_w[:, rows].zero_()

        if out_b is not None and len(prunable_head_dict[idx]) == nheads:
            out_b.zero_()  # whole attention output is zero

        print(f"layer {idx:2d}: pruned heads {prunable_head_dict[idx]}")

    print("all requested heads have been zero-ed")

def merge_dicts(d1, d2):
    merged = {}
    keys = set(d1) | set(d2)  # union of all keys
    for key in keys:
        merged[key] = d1.get(key, []) + d2.get(key, [])
    return merged


if args.pruning_mask is not None:
    print("pxp")
    print(args.pruning_mask)

    mask = torch.load(args.pruning_mask, map_location=device)
    print(mask)
    heads = parse_head_mask(mask)

    if args.merge is not None:
        mask2 = torch.load(args.merge, map_location=device)
        heads2 = parse_head_mask(mask2)
        heads = merge_dicts(heads, heads2)

    zero_vit_heads(model, heads)

# ------------------------------------------------------------
# making sure the right atn heads are pruned
# ------------------------------------------------------------
"""
for i, block in enumerate(model.encoder.layers):
    print("---------------------------------------------------------")
    print("---------------------------", i, "---------------------------")
    print("---------------------------------------------------------")
    print('in weights')
    print(block.self_attention.in_proj_weight.shape)
    print('in bias')
    print(block.self_attention.in_proj_bias.shape)

    for head in range(12):
        print("-----", head, "-----")
        for part in range(3):
            i0 = part * 768 + head * 64
            i1 = i0 + 64
            print('in weights and biases:', part, i0, ":", i1, block.self_attention.in_proj_weight.shape)
            print(block.self_attention.in_proj_weight[i0:i1])
            print(block.self_attention.in_proj_bias[i0:i1])

        c0 = head * 64
        c1 = c0 + 64
        print('out weights:', block.self_attention.out_proj.weight.shape)
        print(block.self_attention.out_proj.weight[:, c0:c1])
"""

# ------------------------------------------------------------
# Pruning n neurons w largest gradient l2 norms
# ------------------------------------------------------------
def get_param_by_name(model, name):
    return reduce(getattr, name.replace('[', '.').replace(']', '').split('.'), model)


all_neurons_to_be_zeord = []
if args.gradient_norm_pruning:
    for group in range(args.n_groups):
        metric_gradients = evaluate_gradients.AvgGradientByGroup(group)
        results = evaluate_gradients.evaluate(model, train_loader, metric_gradients, group, device)
        print("Group: ", results["group"])
        print("N: ", results["n_total"])
        top3 = []
        for k, v in results["gradients"].items():
            if "head" in k:
                continue
            norms = v.cpu().numpy()
            for idx, norm in enumerate(norms):
                if len(top3) < args.n_prunable_neurons:
                    top3.append((k, idx, norm))
                    top3.sort(key=lambda x: x[2], reverse=True)
                else:
                    if norm > top3[-1][2]:
                        top3[-1] = (k, idx, norm)
                        top3.sort(key=lambda x: x[2], reverse=True)
        print("Top 3 neurons (layer, index, norm):")
        for layer, idx, norm in top3:
            print(f"Layer: {layer}, Index: {idx}, Norm: {norm}", results["gradients"][layer].size())

        all_neurons_to_be_zeord += top3

    with torch.no_grad():
        for layer_name, neuron_idx, _ in all_neurons_to_be_zeord:
            weight_tensor = get_param_by_name(model, layer_name)
            weight_tensor[neuron_idx, :].zero_()

## Printing results
metric = evaluate.AccuracyWithGroups()
train_results = evaluate.evaluate(model, train_loader, metric, device)
val_results = evaluate.evaluate(model, val_loader, metric, device)
test_results = evaluate.evaluate(model, test_loader, metric, device)
print("path = ", args.model_path)
print("split = train")
print(train_results)
print("split = val")
print(val_results)
print("split = test")
print(test_results)
print("fin")
