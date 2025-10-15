import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt


def visualize(model, model_path, prune_framework="wd pruning"):
    if prune_framework=="wd_pruning":
        ratios = np.zeros((12, 4), dtype=np.float64)
    
        for i in range(12):
            proj_thr = torch.sigmoid(model.blocks[i].attn.proj.threshold_fc).item()
            proj_numel = model.blocks[i].attn.proj.saliency_scores.numel()
            attn_proj_r = 1 - math.ceil(proj_thr * proj_numel) / proj_numel
    
            fc1_thr = torch.sigmoid(model.blocks[i].mlp.fc1.threshold_fc).item()
            fc1_numel = model.blocks[i].mlp.fc1.saliency_scores.numel()
            fc1_r = 1 - math.ceil(fc1_thr * fc1_numel) / fc1_numel
    
            fc2_thr = torch.sigmoid(model.blocks[i].mlp.fc2.threshold_fc).item()
            fc2_numel = model.blocks[i].mlp.fc2.saliency_scores.numel()
            fc2_r = 1 - math.ceil(fc2_thr * fc2_numel) / fc2_numel
    
            attn_sal = model.blocks[i].attn.qkv.head_saliency_scores
            attn_t = torch.sigmoid(model.blocks[i].attn.qkv.threshold_head).item()
            attn_r = 1 - math.ceil(attn_t * attn_sal.numel()) / attn_sal.numel()
    
            ratios[i] = [attn_r, attn_proj_r, fc1_r, fc2_r, ]
        print(ratios)

    if prune_framework=="pxp":
        zero_stats = count_zero_weights_mask(model)

        ratios = np.zeros((12, 4), dtype=np.float64)
        for i in range(12):
            fc1_key = f"encoder.layers.encoder_layer_{i}.mlp.0"
            fc2_key = f"encoder.layers.encoder_layer_{i}.mlp.3"

            attn_r = 0
            attn_proj_r = 0
            fc1_r = zero_stats[fc1_key][2]
            fc2_r = zero_stats[fc2_key][2]

            ratios[i] = [attn_r, attn_proj_r, fc1_r, fc2_r]
        print("PXP zero-weight ratios:\n", ratios)

    n_layers = ratios.shape[0]
    avgs = np.sum(ratios, axis=0)/n_layers


    bar_width = 0.23
    x = np.arange(n_layers)

    labels = [
        f"attn_head r={avgs[0]:.4f}",
        f"attn_proj r={avgs[1]:.4f}",
        f"fc1 r={avgs[2]:.4f}",
        f"fc2 r={avgs[3]:.4f}",
    ]
    offsets = (np.arange(ratios.shape[1]) - (ratios.shape[1] - 1) / 2) * bar_width

    for j, offset in enumerate(offsets):
        plt.bar(x + offset, ratios[:, j], width=bar_width, label=labels[j])

    plt.xlabel("Transformer block idx")
    plt.ylabel("Prune ratio")
    plt.title("")
    plt.xticks(x, [str(i) for i in range(n_layers)])
    plt.legend()
    plt.tight_layout()

    save_dir = os.path.dirname(model_path)
    save_path = os.path.join(save_dir, "prune_ratios.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()



def count_zero_weights(model):
    """
    return a dict mapping layer names → (zero_count, total_params, zero_ratio).
    """
    zero_stats = {}
    for name, param in model.named_parameters():
        if 'weight' not in name:
            continue  # skip biases, buffers, etc.
        tensor = param.detach()
        total = tensor.numel()
        zeros = int((tensor == 0).sum().item())
        ratio = zeros / total
        zero_stats[name] = (zeros, total, ratio)
    return zero_stats


def count_zero_weights_mask(mask):
    """
    return a dict mapping layer names → (zero_count, total_params, zero_ratio).
    """
    zero_stats = {}
    for k in mask.keys():
        tensor = mask[k]['Linear']['weight']
        tensor = tensor.detach()
        total = tensor.numel()
        zeros = int((tensor == 0).sum().item())
        ratio = zeros / total
        zero_stats[k] = (zeros, total, ratio)
    return zero_stats