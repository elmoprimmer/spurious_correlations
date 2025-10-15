import os

import yaml
import click
import torch
import tqdm.auto
import numpy as np
import wandb

from models import ModelLoader
from metrics import compute_accuracy
from my_metrics import compute_worst_accuracy
from datasets import ImageNet, ImageNetSubset, get_sample_indices_for_class
from my_datasets import WaterBirds, get_sample_indices_for_group, WaterBirdSubset, ISIC, ISICSubset


from utils import (
    initialize_random_seed,
    initialize_wandb_logger,
)

from pxp import (
    ModelLayerUtils,
    get_cnn_composite,
    get_vit_composite,
)

from pxp import GlobalPruningOperations
from pxp import ComponentAttibution


# generate some input reciever using click
@click.command()
@click.option("--configs_path", type=str)
@click.option("--output_path", type=str)
@click.option("--checkpoint_path", type=str)
@click.option("--dataset_path", type=str)
@click.option("--num_workers", type=int, default=4)
@click.option("--dataset_name", type=str, default="waterbirds")
def start(
        configs_path,
        output_path,
        checkpoint_path,
        dataset_path,
        num_workers,
        dataset_name
):




    # Parse the configs file
    with open(configs_path, "r") as stream:
        configs = yaml.safe_load(stream)

    num_workers = num_workers

    configs["output_path"] = output_path
    configs["dataset_path"] = dataset_path
    configs["dataset_name"] = dataset_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    initialize_random_seed(configs["random_seed"])

    results_file_name = (
        f"accuracy_{configs['model_architecture']}_{configs['mode']}_{configs['subsequent_layer_pruning']}_sort{configs['abs_sort']}_leastvalue{configs['least_relevant_first']}_{configs['random_seed']}"
        if configs["mode"] == "Relevance"
        else f"accuracy_{configs['mode']}_{configs['subsequent_layer_pruning']}_sort{configs['abs_sort']}_leastvalue{configs['least_relevant_first']}_{configs['random_seed']}"
    )

    configs["results_file_name"] = results_file_name
    suggested_composite = {
        "low_level_hidden_layer_rule": configs["low_level_hidden_layer_rule"],
        "mid_level_hidden_layer_rule": configs["mid_level_hidden_layer_rule"],
        "high_level_hidden_layer_rule": configs["high_level_hidden_layer_rule"],
        "fully_connected_layers_rule": configs["fully_connected_layers_rule"],
        "softmax_rule": configs["softmax_rule"],
    }

    """
    Initialize WANDB run
    """
    if configs["wandb"]:
        initialize_wandb_logger(**configs)

    """
    Load the dataset
    """

    print("load the dataset")
    if configs["dataset_name"] == "waterbirds":
        print("waterbirds")
        waterbirds = WaterBirds(
            configs["dataset_path"], seed=configs["random_seed"], num_workers=num_workers
        )
        print("get train set")
        train_set = waterbirds.get_train_set()
        print("get valid set")
        val_set = waterbirds.get_valid_set()
        print("get test set")
        test_set = waterbirds.get_test_set()

    if configs["dataset_name"] == "isic":
        print("isic")
        isic = ISIC(
            configs["dataset_path"], metadata_path=configs["metadata_path"], seed=configs["random_seed"], num_workers=num_workers
        )
        print("get train set")
        train_set = isic.get_train_set()
        print("get valid set")
        val_set = isic.get_valid_set()
        print("get test set")
        test_set = isic.get_test_set()
    if configs["dataset_name"] != "isic" and configs["dataset_name"] != "waterbirds":
        print("no / wrong dataset")


    pruning_set = val_set #we prune based on samples from he val set
    pruning_indices = get_sample_indices_for_group(
        pruning_set,
        configs["reference_samples_per_class"],
        "cuda",
        configs["groups"]
    )
    print("pruning indices ", len(pruning_indices))

    validation_indices = get_sample_indices_for_group( #these are just used for printing acc
        test_set, 'all', "cuda"
    )
    print("pruning indices (val)", len(validation_indices))

    if configs["dataset_name"] == "waterbirds":
        custom_pruning_set = WaterBirdSubset(pruning_set, pruning_indices)
        custom_validation_set = WaterBirdSubset(test_set, validation_indices)

    if configs["dataset_name"] == "isic":
        custom_pruning_set = ISICSubset(pruning_set, pruning_indices)
        custom_validation_set = ISICSubset(test_set, validation_indices)

    custom_pruning_dataloader = torch.utils.data.DataLoader(
        custom_pruning_set,
        batch_size=configs["pruning_dataloader_batchsize"],
        shuffle=True,
        num_workers=num_workers,
    )
    custom_validation_dataloader = torch.utils.data.DataLoader(
        custom_validation_set,
        batch_size=configs["validation_dataloader_batchsize"],
        shuffle=False,
        num_workers=num_workers,
    )


    del custom_pruning_set
    del custom_validation_set
    del validation_indices
    del pruning_indices
    del train_set
    del pruning_set

    """
    Load the model and applying the Composites/Canonizers
    """
    if configs["model_architecture"] == "vit_b_16":
        composite = get_vit_composite(
            configs["model_architecture"], suggested_composite
        )
    else:
        composite = get_cnn_composite(
            configs["model_architecture"], suggested_composite
        )

    model = ModelLoader.get_basic_model(
        configs["model_architecture"], configs["checkpoint_path"], device, num_classes=2
    )



    layer_types = {
        "Softmax": torch.nn.Softmax,
        "Linear": torch.nn.Linear,
        "Conv2d": torch.nn.Conv2d,
    }

    # modify model's last linear layer for domain
    # restriction (e.g., 3 classes classification)

    """
    Setting up extra configurations for the pruning
    """
    pruning_rates = configs["pruning_rates"]
    print("pruning_rates")
    print(pruning_rates)



    component_attributor = ComponentAttibution(
        "Relevance",
        "ViT" if configs["model_architecture"] == "vit_b_16" else "CNN",
        layer_types[configs["pruning_layer_type"]],
        configs["least_relevant_first"]
    )

    components_relevances = component_attributor.attribute(
        model,
        custom_pruning_dataloader,
        composite,
        abs_flag=configs["abs_sort"],
        device=device,
    )



    acc_top1 = compute_accuracy(
        model,
        custom_validation_dataloader,
        device,
    )

    print(f"Initial accuracy (val set): top1={acc_top1}")

    worst_acc, group_acc = compute_worst_accuracy(
        model,
        custom_validation_dataloader,
        device,
    )
    print(f"Initial worst accuracy (val set)={worst_acc}")
    print(f"Initial group accuracies (val set)={group_acc}")

    print("0:", group_acc[0])
    print("1:", group_acc[1])
    print("2:", group_acc[2])
    print("3:", group_acc[3])
    
    """
    Experiment's main loop
    """
    layer_names = component_attributor.layer_names
    pruner = GlobalPruningOperations(
        layer_types[configs["pruning_layer_type"]],
        layer_names,
    )
    print(pruner)
    top1_acc_list = []
    worst_acc_list = []
    progress_bar = tqdm.tqdm(total=len(pruning_rates))
    for pruning_rate in pruning_rates:
        progress_bar.set_description(f"Processing {int((pruning_rate) * 100)}% Pruning")
        # skip pruning if compression rate is 0.00 as we
        # have computed few lines above, otherwise prune
        if pruning_rate != 0.0:
            # prune the model based on the
            # pre-computed attibution flow
            # (relevance values)
            global_pruning_mask = pruner.generate_global_pruning_mask(
                model,
                components_relevances,
                pruning_rate,
                subsequent_layer_pruning=configs["subsequent_layer_pruning"],
                least_relevant_first=configs["least_relevant_first"],
                device=device,
            )
            print("global_pruning_mask")
            print(configs["output_path"] + f"{pruning_rate * 100}.pth")
            os.makedirs(configs["output_path"], exist_ok=True)
            torch.save(global_pruning_mask, configs["output_path"] + f"{pruning_rate}.pth")
            # Our pruning gets applied by masking the
            # activation of layers via forward hooks.
            # Therefore hooks are returned for later
            # removal
            hook_handles = pruner.fit_pruning_mask(
                model,
                global_pruning_mask,
            )
            print(hook_handles)

        progress_bar.set_description(
            f"Computing accuracy for model pruned with {int((pruning_rate) * 100)}%"
        )
        acc_top1_train = compute_accuracy(
            model,
            custom_pruning_dataloader,
            device,
        )
        print(f"acc top1 (train) {acc_top1_train}")
        worst_acc_train, worst_groups_train = compute_worst_accuracy(
            model,
            custom_pruning_dataloader,
            device
        )
        print(f"group accs (train) {worst_groups_train}")

        
        acc_top1 = compute_accuracy(
            model,
            custom_validation_dataloader,
            device,
        )
        worst_acc, worst_groups = compute_worst_accuracy(
            model,
            custom_validation_dataloader,
            device
        )

        # Remove/Deactivate hooks (except
        # when the pruning rate is 0.00)
        if pruning_rate != 0.0:
            if layer_types[configs["pruning_layer_type"]] == torch.nn.Softmax:
                for hook in hook_handles:
                    hook.remove()
        top1_acc_list.append(acc_top1)
        worst_acc_list.append(worst_acc)
        print(f"Accuracy-Flow list (val): {top1_acc_list}")
        print(f"WORST Accuracy-Flow list (val): {worst_acc_list}")
        print(f"Group accuracy (val): {worst_groups}")
        print("0:", worst_groups[0])
        print("1:", worst_groups[1])
        print("2:", worst_groups[2])
        print("3:", worst_groups[3])
        """
        Logging the results on WandB
        """
        if configs["wandb"]:
            wandb.log({"acc_top1 (val)": acc_top1, "acc_top1 (train)": acc_top1_train, "pruning_rate": pruning_rate, "group accuracy (val)": worst_groups, "group accuracy (train)": worst_groups_train})
            print(f"Logged the results of {pruning_rate}% Pruning Rate to wandb!")
        progress_bar.update(1)

    # empty up the GPU memory and CUDA cache, model and dataset
    progress_bar.close()
    del pruner
    torch.cuda.empty_cache()

    top1_auc = compute_auc(top1_acc_list, pruning_rates)
    if configs["wandb"]:
        wandb.log({"top1_auc": top1_auc})
        print(f"Logged the AUC of the Top1 Accuracy to wandb!")

    print(f"Top1 AUC: {top1_auc}")

    #worst_auc = compute_auc(worst_acc_list, pruning_rates)
    #print(f"Worst AUC: {worst_auc}")


def compute_auc(top1_acc_list, pruning_rates):
    """
    Compute the Area Under the Curve (AUC) for the accuracy over the pruning rates
    """
    top1_auc = np.trapz(top1_acc_list, pruning_rates)
    print(f"Top1 AUC: {top1_auc}")
    return top1_auc


if __name__ == "__main__":
    start()
