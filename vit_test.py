from functools import partial
import os

import torch.nn as nn
from torchvision.models import vit_b_16
import torch

import argparse

from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data
from utils import evaluate, get_y_p
print("start")


parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
parser.add_argument(
    "--data_dir", type=str,
    default=None,
    help="Train dataset directory")



MODEL_NAME = f"vit_waterbirds"
NUM_CLASSES = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pre-trained ViT model
model = vit_b_16(weights='DEFAULT')
# Modify the final layer for the dataset
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
model.to(device)
model.eval()
# load the pre-trained model
model.load_state_dict(torch.load(f"models/{MODEL_NAME}.pth", map_location=device))
print(model)


target_resolution = (224, 224)
test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=False)
test_wb_dir = os.path.expandvars("$HPC_SCRATCH/waterbird")
testset = WaterBirdsDataset(basedir=test_wb_dir, split="test", transform=test_transform)



loader_kwargs = {'batch_size': 32,
                 'num_workers': 4, 'pin_memory': True,
                 "reweight_places": None}

test_loader = get_loader(
    testset, train=False, reweight_groups=None, reweight_classes=None,
    **loader_kwargs)

results = {}
get_yp_func = partial(get_y_p, n_places=testset.n_places)
results["test"] = evaluate(model, test_loader, get_yp_func)
print(results["test"])
