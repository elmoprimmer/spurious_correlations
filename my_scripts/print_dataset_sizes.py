import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT)
from ISIC_ViT.isic_data import ISICDataset
from external.dfr.wb_data import WaterBirdsDataset
import torch

import argparse

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
    "--dataset", type=str,
    default="isic",
    help="are we using isic or waterbirds")


args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



if args.dataset == "isic":
    print("train")
    train_dataset = ISICDataset(basedir=args.data_dir,
                                csv_file=args.metadata_csv,
                                transform=None,
                                split="train")
    print("val")
    val_dataset = ISICDataset(basedir=args.data_dir,
                                csv_file=args.metadata_csv,
                                transform=None,
                                split="val")
    print("test")
    test_dataset = ISICDataset(basedir=args.data_dir,
                                csv_file=args.metadata_csv,
                                transform=None,
                                split="test")




if args.dataset == "waterbirds":
    print("train")
    train_dataset = WaterBirdsDataset(basedir=args.data_dir,
                                      transform=None,
                                      split="train")
    print("val")
    val_dataset = WaterBirdsDataset(basedir=args.data_dir,
                                    transform=None,
                                    split="val")
    print("test")
    test_dataset = WaterBirdsDataset(basedir=args.data_dir,
                                    transform=None,
                                    split="test")




