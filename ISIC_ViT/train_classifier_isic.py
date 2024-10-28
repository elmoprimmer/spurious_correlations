import pickle

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import os
import tqdm
import argparse
import sys
from collections import defaultdict
import json
from functools import partial

from torchvision.models import ViT_B_16_Weights

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from isic_data import ISICDataset
from utils import Logger, AverageMeter, set_seed, evaluate, get_results, write_dict_to_tb, get_y_p

parser = argparse.ArgumentParser(description="Train Vision Transformer on ISIC dataset")
parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
parser.add_argument("--label_csv", type=str, default=None, help="Path to csv with labels")
parser.add_argument("--output_dir", type=str, default="logs/", help="Output directory for logs and model checkpoints")
parser.add_argument("--pretrained_model", action='store_true', help="Use pretrained ViT model")
parser.add_argument("--scheduler", action='store_true', help="Use learning rate scheduler")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--init_lr", type=float, default=1e-4)
parser.add_argument("--eval_freq", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")

args = parser.parse_args()

print('Preparing directory %s' % args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
    args_json = json.dumps(vars(args))
    f.write(args_json)

set_seed(args.seed)

writer = SummaryWriter(log_dir=args.output_dir)
logger = Logger(os.path.join(args.output_dir, 'log.txt'))

# data preparation
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
trainset = ISICDataset(basedir=args.data_dir, csv_file=args.label_csv, transform=train_transform, split="train")
testset = ISICDataset(basedir=args.data_dir, csv_file=args.label_csv, transform=test_transform, split="test")
valset = ISICDataset(basedir=args.data_dir, csv_file=args.label_csv, transform=test_transform, split="val")


loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, **loader_kwargs)
test_loader = torch.utils.data.DataLoader(testset, shuffle=False, **loader_kwargs)
val_loader = torch.utils.data.DataLoader(valset, shuffle=False, **loader_kwargs)

unique_combinations = trainset.metadata.groupby(['benign_malignant', 'patches']).size()
logger.write(f"Unique combinations in the dataset: {unique_combinations}")

train_bincounts = np.bincount(trainset.group_array)
logger.write(f"Train bincounts: {train_bincounts}")
test_bincounts = np.bincount(testset.group_array)
logger.write(f"Test bincounts: {test_bincounts}")
val_bincounts = np.bincount(valset.group_array)
logger.write(f"Validation bincounts: {test_bincounts}")

get_yp_func = partial(get_y_p, n_places=trainset.n_secondary_classes)

model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

in_features = model.heads.head.in_features
model.heads.head = torch.nn.Linear(in_features, 2)  # binary classification, benign (0), malignant (1)

model = model.cuda()

# optimiser, scheduler and loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
else:
    scheduler = None

criterion = torch.nn.CrossEntropyLoss()

logger.flush()

# training loop
for epoch in range(args.num_epochs):
    model.train()
    loss_meter = AverageMeter()

    for batch in tqdm.tqdm(train_loader):
        x, y, _, _ = batch
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), x.size(0))

    if args.scheduler:
        scheduler.step()

    logger.write(f"Epoch {epoch}\t Loss: {loss_meter.avg}\n")

    # Evaluation at each epoch
    if epoch % args.eval_freq == 0:
        test_results = evaluate(model, test_loader, get_yp_func)
        logger.write(f"Test results at epoch {epoch}: {test_results}\n")
        write_dict_to_tb(writer, test_results, "test/", epoch)

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"))

val_results = evaluate(model, val_loader, get_yp_func)
test_results = evaluate(model, test_loader, get_yp_func)
logger.write(f"Validation results at end: {val_results}")
torch.save(model.state_dict(), os.path.join(args.output_dir, 'vit_isic_final_checkpoint_test.pt'))
logger.write('\n')

all_results = {}
all_results['test'] = test_results
all_results['val'] = val_results

with open(args.result_path, 'wb') as f:
    pickle.dump(all_results, f)
