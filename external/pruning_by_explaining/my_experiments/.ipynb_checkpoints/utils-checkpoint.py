import yaml
import click
import torch
import tqdm.auto
import numpy as np
from torchvision.models import vit_b_16

from matplotlib import pyplot as plt


import sys
import os
import re
import math
#project_root = "C:/Users/elmop/deep_feature_reweighting/deep_feature_reweighting/external/pruning_by_explaining"
project_root = "/home/primmere/ide/external/pruning_by_explaining"
sys.path.insert(0, project_root)                 
sys.path.insert(0, os.path.dirname(project_root))

from pruning_by_explaining.models import ModelLoader
from pruning_by_explaining.metrics import compute_accuracy
from pruning_by_explaining.my_metrics import compute_worst_accuracy
from pruning_by_explaining.my_datasets import WaterBirds, get_sample_indices_for_group, WaterBirdSubset, ISIC, ISICSubset
from pruning_by_explaining.utils import (
    initialize_random_seed,
    initialize_wandb_logger,
)


from pruning_by_explaining.pxp import (
    ModelLayerUtils,
    get_cnn_composite,
    get_vit_composite,
)

from pruning_by_explaining.pxp import GlobalPruningOperations
from pruning_by_explaining.pxp import ComponentAttibution


def plot_layer_head_pruned(mask, title="Self-Attention Softmax", cmap="magma",
                            save_path=None, show=True,):

    
    
    mat = np.zeros((12,12))
    for v, i in zip(mask.values(),range(len(mat))):
        for v2 in v:
            mat[i][v2] = 1.0

    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"{i}" for i in range(mat.shape[0])])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relevance")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_layer_head_heatmap(d, title="Self-Attention Softmax", cmap="magma",
                            save_path=None, show=True, normalise = 1):
    # d = relevance scores
    
    mat = np.zeros((12,12))
    for v, i in zip(d.values(),range(len(mat))):
        mat[i] = v/normalise

    
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    im = ax.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    #ax.set_title(title)
    ax.set_xticks(range(mat.shape[1]))
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([f"{i}" for i in range(mat.shape[0])])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relevance")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax



def visualise(global_pruning_mask_combined, prune_r, layertype = "Linear", save = False):
    if layertype == "Linear":
        count=0
        ratios = np.zeros((12,2))
        i = 0
        for n, t in global_pruning_mask_combined.items():
            param_total = t['Linear']['weight'].numel()
            param_nonzero = t['Linear']['weight'].nonzero().size(0)
            param_shape = t['Linear']['weight'].shape
        
            pruned = (param_total-param_nonzero)/param_shape[1]
            total = param_total/param_shape[1]
        
            if 'mlp.0' in n:
                ratios[i][0] = pruned/total
            if 'mlp.3' in n:
                ratios[i][1] = pruned/total
                i += 1
        
        #print(array)
        avgs = np.sum(ratios, axis=0)/12
        
        bar_width = 0.23
        x = np.arange(12)
        
        labels = [
            f"fc1 r={avgs[0]:.4f}",
            f"fc2 r={avgs[1]:.4f}",
        ]
        offsets = (np.arange(ratios.shape[1]) - (ratios.shape[1] - 1) / 2) * bar_width
        
        for j, offset in enumerate(offsets):
            plt.bar(x + offset, ratios[:, j], width=bar_width, label=labels[j])
        
        plt.xlabel("Transformer block idx")
        plt.ylabel("Prune ratio")
        plt.title(f'r = {prune_r}')
        plt.xticks(x, [str(i) for i in range(12)])
        plt.ylim(0, 1)   
        plt.legend()
        plt.tight_layout()
        
        save_dir = ""
        save_path = os.path.join(save_dir, f"prune_ratios{prune_r}.png")
        if save:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
    if layertype == "Softmax":
        ratios = np.zeros((12,1))
        i = 0
        for v in global_pruning_mask_combined.values():
            ratios[i][0] = len(v.detach().numpy())/12
            i += 1

        bar_width = 0.23
        x = np.arange(12)
        
        labels = [
            f"Attention head",
        ]
        offsets = (np.arange(ratios.shape[1]) - (ratios.shape[1] - 1) / 2) * bar_width
        
        for j, offset in enumerate(offsets):
            plt.bar(x + offset, ratios[:, j], width=bar_width, label=labels[j])
        
        plt.xlabel("Transformer block idx")
        plt.ylabel("Prune ratio")
        plt.title(f'r = {prune_r}')
        plt.xticks(x, [str(i) for i in range(12)])
        plt.ylim(0, 1)   
        plt.legend()
        plt.tight_layout()
        save_dir = ""
        save_path = os.path.join(save_dir, f"prune_ratios{prune_r}.png")
        if save:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()



def plot_r_accuracy_lines(arr, title="Accuracy vs r",
                          save_path=None, show=True):
    """
    make a line chart of acc progress
    arr: np array with r, acc and group accs
    """

    
    arr = np.asarray(arr, dtype=float)

    order = np.argsort(arr[:, 0])
    r = arr[order, 0]
    acc_cols = arr[order, 1:]
    labels = ["Total accuracy", "Group 0", "Group 1", "Group 2", "Group 3"]

    
    

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)

    for i in range(acc_cols.shape[1]):
        mask = ~np.isnan(acc_cols[:, i]) & ~np.isnan(r)
        if i == 0:
            ax.plot(r[mask], acc_cols[mask, i], marker="o", color="black", linewidth=1.8, label=labels[i])
        else:
            ax.plot(r[mask], acc_cols[mask, i], marker="o", linewidth=1.8, label=labels[i])

    ax.set_xlabel("r")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax

