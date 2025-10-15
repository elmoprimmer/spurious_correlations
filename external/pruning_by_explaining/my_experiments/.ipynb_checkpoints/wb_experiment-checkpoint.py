

import os
import re
import sys
import json
import math
import argparse
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.models import vit_b_16

# Add local project roots used in the notebook (adjust if needed)
DEFAULT_PROJECT_ROOT = "/home/primmere/ide/external/pruning_by_explaining"
if DEFAULT_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, DEFAULT_PROJECT_ROOT)
    sys.path.insert(0, os.path.dirname(DEFAULT_PROJECT_ROOT))

# --- Imports from the notebook's package ---
from pruning_by_explaining.models import ModelLoader
from pruning_by_explaining.metrics import compute_accuracy
from pruning_by_explaining.my_metrics import compute_worst_accuracy
from pruning_by_explaining.my_datasets import (
    WaterBirds, get_sample_indices_for_group, WaterBirdSubset, ISIC, ISICSubset
)
from pruning_by_explaining.utils import (
    initialize_random_seed,
    initialize_wandb_logger,
)

from pruning_by_explaining.pxp import (
    ModelLayerUtils,
    get_cnn_composite,
    get_vit_composite,
    GlobalPruningOperations,
    ComponentAttibution,
)

# Helpful plotting utils referenced in the notebook.
try:
    from pruning_by_explaining.my_experiments.utils import (
        visualise,
        plot_layer_head_heatmap,
        plot_layer_head_pruned,
        plot_r_accuracy_lines,
    )
except Exception:
    visualise = None
    plot_layer_head_heatmap = None
    plot_layer_head_pruned = None
    plot_r_accuracy_lines = None

# Head pruning helpers (locations may vary; try multiple import paths)
parse_head_mask = None
zero_vit_heads = None
for _mod in [
    "pruning_by_explaining.my_experiments.utils",
    "pruning_by_explaining.pxp",
    "pruning_by_explaining.utils",
]:
    try:
        m = __import__(_mod, fromlist=["parse_head_mask", "zero_vit_heads"])
        if parse_head_mask is None and hasattr(m, "parse_head_mask"):
            parse_head_mask = getattr(m, "parse_head_mask")
        if zero_vit_heads is None and hasattr(m, "zero_vit_heads"):
            zero_vit_heads = getattr(m, "zero_vit_heads")
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Waterbirds pruning experiment (SLURM-ready)")

    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument("--num-workers", type=int, default=12, help="DataLoader workers (default: 12)")

    parser.add_argument("--not_wb", action="store_true", help="Disable saving artifacts")

    # Saving enabled by default; --no-save turns it off.
    parser.add_argument("--no-save", action="store_true", help="Disable saving artifacts")

    parser.add_argument("--prune_w_train_set", action="store_true", help="prune w samples from train set and groups = None")

    parser.add_argument("--prune-r", type=str, required=True,
                        help="Comma or space separated list of prune ratios, e.g. '0.0,0.1,0.2' or '0.0 0.1 0.2'")

    parser.add_argument("--layer-type", type=str, choices=["Linear", "Softmax"], required=True,
                        help="Layer type to prune ('Linear' or 'Softmax')")

    parser.add_argument("--n-indices", type=int, required=True,
                        help="Number of indices per selected groups for pruning set")

    parser.add_argument("--groups", type=str, default=None,
                        help="Comma/space separated group IDs, e.g. '1,2'. Use 'None' to disable group filtering")
    
    parser.add_argument("--combining_relevances", type=str, default = None, help="Do we minus a certain groups importance? if so which?")

    # Practical extras with sensible defaults based on the notebook (can be overridden if needed)
    parser.add_argument("--data-root", type=str, default="/scratch_shared/primmere/waterbird",
                        help="WaterBirds dataset root (default from notebook)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to ViT checkpoint (default from notebook)")
    parser.add_argument("--batch-size-val", type=int, default=32, help="Batch size for validation/test loader")
    parser.add_argument("--batch-size-prune", type=int, default=1, help="Batch size for pruning attribution loader")

    return parser.parse_args()


def _parse_float_list(s: str):
    if s is None:
        return []
    tokens = re.split(r"[,\s]+", s.strip())
    out = []
    for t in tokens:
        if t == "":
            continue
        out.append(float(t))
    return out


def _parse_groups(s: str):
    if s is None or s.strip().lower() == "none":
        return None
    toks = re.split(r"[,\s]+", s.strip())
    groups = []
    for t in toks:
        if t == "":
            continue
        groups.append(int(t))
    return groups


def _make_experiment_folder(args, dataset_name_short = 'wb'):
    g = _parse_groups(args.groups)
    g2 = _parse_groups(args.combining_relevances)
    g_str = "None" if g is None else "_".join(map(str, g))
    
    name = f"f_experiments/{dataset_name_short}_{args.layer_type}_g-{g_str}_n-{args.n_indices}_seed-{args.seed}"
    if g2 is not None:
        g_str2 = "None" if g is None else "_".join(map(str, g2))
        name = f"f_experiments/{dataset_name_short}_{args.layer_type}_g-{g_str}_minus-{g_str2}_n-{args.n_indices}_seed-{args.seed}"

    if args.prune_w_train_set:
        name = f"f_experiments/{dataset_name_short}_{args.layer_type}_g-{g_str}_n-{args.n_indices}_seed-{args.seed}_train"
    exp_dir = os.path.abspath(name)
    os.makedirs(exp_dir, exist_ok=True)
        
    return exp_dir


def main():
    args = parse_args()

    least_rel_first2 = False
    abs_flag2 = False
    Zplus_flag = True
    
    scale_bool = True #TODO make int and assignable

    save_artifacts = not args.no_save
    wb_bool = not args.not_wb
    prune_r_list = _parse_float_list(args.prune_r)
    if len(prune_r_list) == 0:
        raise ValueError("--prune-r must include at least one value")

    
    sel_groups = _parse_groups(args.groups)  # can be None
    combining_relevances_groups = _parse_groups(args.combining_relevances)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_string = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # Seed & workers
    initialize_random_seed(args.seed)

    # Dataset
    if wb_bool:
        dataset_name_short = 'wb'
        dataset_name = 'Waterbirds'
        args.checkpoint = "/home/primmere/ide/dfr/logs/vit_waterbirds.pth"
        print('wb')
        waterbirds = WaterBirds(args.data_root, seed=args.seed, num_workers=args.num_workers)
        train_set = waterbirds.get_train_set()
        val_set = waterbirds.get_valid_set()
        test_set = waterbirds.get_test_set()

            
        
    if not wb_bool:
        dataset_name_short = 'isic'
        dataset_name = 'ISIC'
        args.checkpoint = '/home/primmere/logs/isic_logs_4/vit_isic_v2.pt'
        print('isic')
        isic = ISIC(
            "/scratch_shared/primmere/isic/isic_224/raw_224_with_selected", 
            metadata_path='/scratch_shared/primmere/isic/metadata_w_split_v2_w_elmos_modifications.csv', 
            seed=args.seed, 
            num_workers=args.num_workers
                )
        train_set = isic.get_train_set()
        val_set = isic.get_valid_set()
        test_set = isic.get_test_set()
    print(dataset_name)
    print(args.checkpoint)
    print("aaaaaa")
    # Pruning subset indices: pass groups (possibly None) directly
    pruning_indices = get_sample_indices_for_group(val_set, args.n_indices, device_string, sel_groups)
    if args.prune_w_train_set: pruning_indices = get_sample_indices_for_group(val_set, args.n_indices, device_string, None)
        


    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=args.num_workers)
    prune_dataloader = torch.utils.data.DataLoader(WaterBirdSubset(val_set, pruning_indices),
                                                   batch_size=args.batch_size_prune, shuffle=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers)

    if combining_relevances_groups is not None:
        print('comb groups ', combining_relevances_groups)
        pruning_indices2 = get_sample_indices_for_group(val_set, args.n_indices, device_string, combining_relevances_groups)
        custom_pruning_set2 = WaterBirdSubset(val_set, pruning_indices2)
        prune_dataloader2 = torch.utils.data.DataLoader(custom_pruning_set2, args.batch_size_prune, shuffle=True, num_workers=args.num_workers)

    # Composite / layer mapping
    suggested_composite = {
        "low_level_hidden_layer_rule": "Epsilon",
        "mid_level_hidden_layer_rule":"Epsilon",
        "high_level_hidden_layer_rule": "Epsilon",
        "fully_connected_layers_rule": "Epsilon",
        "softmax_rule": "Epsilon",
    }
    composite = get_vit_composite("vit_b_16", suggested_composite)
    layer_types = {
        "Softmax": torch.nn.Softmax,
        "Linear": torch.nn.Linear,
        "Conv2d": torch.nn.Conv2d,
    }

    # Base model (for attribution only; we reload per-r for saving/eval parity)
    model = ModelLoader.get_basic_model("vit_b_16", args.checkpoint, device, num_classes=2)


    # Attribution on base model
    least_rel_first = True
    abs_flag = True
    component_attributor = ComponentAttibution(
        "Relevance",
        "ViT",
        layer_types[args.layer_type],
        least_rel_first
    )
    components_relevances = component_attributor.attribute(
        model,
        prune_dataloader,
        composite,
        abs_flag=abs_flag,
        Zplus_flag=False,
        device=device,
    )
    layer_names = component_attributor.layer_names

    
    if combining_relevances_groups is not None:
        model2 = ModelLoader.get_basic_model("vit_b_16", args.checkpoint, device, num_classes=2)
        component_attributor2 = ComponentAttibution(
                "Relevance",
                "ViT",
                layer_types[args.layer_type],
                least_rel_first2
            )
        
        components_relevances2 = component_attributor2.attribute(
                model2,
                prune_dataloader2,
                composite,
                abs_flag=abs_flag2,
                Zplus_flag=Zplus_flag,
                device=device,
            )
        layer_names2 = component_attributor.layer_names
        pruner2 = GlobalPruningOperations(
                layer_types[args.layer_type],
                layer_names2,
            )

        global_pruning_mask2 = pruner2.generate_global_pruning_mask(
                model2,
                components_relevances2,
                0.1,
                subsequent_layer_pruning=args.layer_type,
                least_relevant_first=least_rel_first,
                device=device,
            )
        print(pruning_indices)
        print(pruning_indices2)
        scale = len(pruning_indices)
        scale2 = len(pruning_indices2)
        if scale_bool:
            for t in components_relevances.values():
                t.div_(scale)
            for t in components_relevances2.values():
                t.div_(scale2*2) #TODO this should be controllable
        check = True
#        print(components_relevances2)
        for v, v2 in zip(components_relevances.values(), components_relevances2.values()):
            print(v[0:8])
            print(v2[0:8])
            print('---')
        combined_relevances = {}
        for (k1, v1), (k2, v2) in zip(
                components_relevances.items(), components_relevances2.items()
        ):
            check = check & (k1==k2)
            combined_relevances[k1] = v1 - v2
        print("all keys match = ",check)
        components_relevances = combined_relevances
        
    # Global pruner
    pruner = GlobalPruningOperations(layer_types[args.layer_type], layer_names)

    # Results array: columns -> [r, acc_overall, g0, g1, g2, g3]
    accs = np.zeros((len(prune_r_list), 6), dtype=float)

    # Experiment folder
    exp_dir = _make_experiment_folder(args, dataset_name_short = dataset_name_short)
    print(f"[INFO] Saving to: {exp_dir}" if save_artifacts else "[INFO] Saving disabled (--no-save)")

    saved_paths = []


    # Run across prune ratios (reload fresh base model each time so each r is independent)
    for i, r in enumerate(prune_r_list):
        print(f"[INFO] Pruning ratio r={r}")

        # Fresh model per-r
        model_r = ModelLoader.get_basic_model("vit_b_16", args.checkpoint, device, num_classes=2)

        # Turn new model into correct format
        _ = component_attributor.attribute(
            model_r,
            prune_dataloader,
            composite,
            abs_flag=abs_flag,
            Zplus_flag=False,
            device=device,
        )

        global_pruning_mask = pruner.generate_global_pruning_mask(
            model_r,
            components_relevances,
            r,
            subsequent_layer_pruning=args.layer_type,
            least_relevant_first=True,
            device=device,
        )

        # Apply pruning for evaluation
        hook_handles = pruner.fit_pruning_mask(model_r, global_pruning_mask)

        # Evaluate
        acc, acc_groups = compute_worst_accuracy(
            model_r,
            val_dataloader,
            device,
        )
        accs[i] = np.array([r, acc, acc_groups[0], acc_groups[1], acc_groups[2], acc_groups[3]])
        print(accs[i])
        # Remove hooks (if any)
        try:
            for h in hook_handles or []:
                try:
                    h.remove()
                except Exception:
                    pass
        except Exception:
            pass

        # Save model for this r
        if save_artifacts:
            if args.layer_type == "Softmax":
                if parse_head_mask is None or zero_vit_heads is None:
                    raise RuntimeError("Softmax pruning requires parse_head_mask and zero_vit_heads to be importable.")
                # Bake pruning into a copy so evaluation path remains hook-based
                import copy as _copy
                if wb_bool:
                    chkpt_path = "/home/primmere/ide/dfr/logs/vit_waterbirds.pth"
                if not wb_bool:
                    chkpt_path = '/home/primmere/logs/isic_logs_4/vit_isic_v2.pt'
                checkpoint = torch.load(chkpt_path, map_location=torch.device('cuda'), weights_only=True)
                model_to_save = vit_b_16(num_classes = 2)
                model_to_save.load_state_dict(checkpoint)
                try:
                    head_mask = parse_head_mask(global_pruning_mask)
                except TypeError:
                    head_mask = parse_head_mask(global_pruning_mask, layer_names)
                zero_vit_heads(model_to_save, head_mask)
                ckpt_path = os.path.join(exp_dir, f"model_pruned_softmax_r{r:g}.pth")
                torch.save(model_to_save.state_dict(), ckpt_path)
            else:
                # Linear: pruner zeroes weights in-place on model_r
                ckpt_path = os.path.join(exp_dir, f"model_pruned_linear_r{r:g}.pth")
                torch.save(model_r.state_dict(), ckpt_path)
            saved_paths.append(ckpt_path)
            print(f"[INFO] Saved model for r={r} -> {ckpt_path}")

        # Per-r visualisation of pruning mask (optional)
        if save_artifacts and args.layer_type != "Linear":
            try:
                if plot_layer_head_pruned is not None:
                    plt.figure()
                    out = plot_layer_head_pruned(global_pruning_mask)
                    plt.tight_layout()
                    plt.savefig(os.path.join(exp_dir, f"mask_pruned_r{r:g}.png"), dpi=200, bbox_inches="tight")
                    plt.close()
            except Exception as e:
                print(f"[WARN] plot_layer_head_pruned failed at r={r}: {e}")

    # Save the pretty-printed LaTeX table rows as in the notebook
    lines = []
    for i in range(len(accs)):
        lines.append(f"{accs[i][0]} & {accs[i][1]:.3f} & {accs[i][2]:.3f} & {accs[i][3]:.3f} & {accs[i][4]:.3f} & {accs[i][5]:.3f} \\\\")
    table_txt = "\n".join(lines)

    if save_artifacts:
        with open(os.path.join(exp_dir, "acc_table.txt"), "w") as f:
            f.write(table_txt)

    # Visualisations (Linear: visualise + r-accuracy; Softmax: all four if available)
    if save_artifacts:
        # Accuracy lines (always attempt)
        try:
            if plot_r_accuracy_lines is not None:
                plt.figure()
                plot_r_accuracy_lines(accs, save_path=os.path.join(exp_dir, f"acc_{args.layer_type.lower()}.png"),
                                      title=f"{args.layer_type} Layers of {dataset_name} Model")
                plt.close()
            else:
                import matplotlib.pyplot as _plt
                _plt.figure()
                _plt.plot(accs[:,0], accs[:,1], marker="o")
                _plt.xlabel("r")
                _plt.ylabel("Accuracy")
                _plt.title(f"{args.layer_type} Layers - Accuracy vs r")
                _plt.grid(True, linestyle=":")
                _plt.savefig(os.path.join(exp_dir, f"acc_{args.layer_type.lower()}_basic.png"), dpi=200, bbox_inches="tight")
                _plt.close()
        except Exception as e:
            print(f"[WARN] plot_r_accuracy_lines failed: {e}")

        if args.layer_type == "Linear":
            try:
                if visualise is not None:
                    saved = False
                    try:
                        visualise(components_relevances, save_path=os.path.join(exp_dir, "visualise.png"))
                        saved = True
                    except Exception:
                        pass
                    if not saved:
                        try:
                            visualise(components_relevances)
                            plt.savefig(os.path.join(exp_dir, "visualise.png"), dpi=200, bbox_inches="tight")
                            plt.close()
                            saved = True
                        except Exception:
                            pass
                    if not saved:
                        print("[WARN] Could not run visualise() with guessed signatures; skipping")
                else:
                    print("[INFO] visualise() not available; skipping")
            except Exception as e:
                print(f"[WARN] visualise() failed: {e}")
        else:
            try:
                if plot_layer_head_heatmap is not None:
                    plt.figure()
                    plot_layer_head_heatmap(components_relevances, save_path=os.path.join(exp_dir, "pxp_attn_heatmap.png"))
                    plt.close()
            except Exception as e:
                print(f"[WARN] plot_layer_head_heatmap failed: {e}")

            try:
                if visualise is not None:
                    saved = False
                    try:
                        visualise(components_relevances, save_path=os.path.join(exp_dir, "visualise.png"))
                        saved = True
                    except Exception:
                        pass
                    if not saved:
                        try:
                            visualise(components_relevances)
                            plt.savefig(os.path.join(exp_dir, "visualise.png"), dpi=200, bbox_inches="tight")
                            plt.close()
                            saved = True
                        except Exception:
                            pass
                    if not saved:
                        print("[WARN] Could not run visualise() with guessed signatures; skipping")
                else:
                    print("[INFO] visualise() not available; skipping")
            except Exception as e:
                print(f"[WARN] visualise() failed: {e}")

    # Also store the raw accs array
    if save_artifacts:
        np.save(os.path.join(exp_dir, "accs.npy"), accs)

    # Echo the table to stdout for SLURM logs
    print("\n=== ACC TABLE (LaTeX rows) ===")
    print(table_txt)
    print("==============================\n")

    if save_artifacts:
        print("[INFO] Saved model files:")
        for p in saved_paths:
            print(p)

if __name__ == "__main__":
    main()