import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt


def visualize(model, model_path):
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