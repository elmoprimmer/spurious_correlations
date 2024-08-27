"""Evaluate DFR on spurious correlations datasets."""

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
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision.models import vit_b_16

from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p


# WaterBirds
C_OPTIONS = [1., 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100., 300., 1000.]
# CelebA
REG = "l1"
# # REG = "l2"
# C_OPTIONS = [3., 1., 0.3, 0.1, 0.03, 0.01, 0.003]
# CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100, 300, 500]

CLASS_WEIGHT_OPTIONS = [{0: 1, 1: w} for w in CLASS_WEIGHT_OPTIONS] + [
        {0: w, 1: 1} for w in CLASS_WEIGHT_OPTIONS]


parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
parser.add_argument(
    "--data_dir", type=str,
    default=None,
    help="Train dataset directory")
parser.add_argument(
    "--result_path", type=str, default="logs/",
    help="Path to save results")
parser.add_argument(
    "--ckpt_path", type=str, default=None, help="Checkpoint path")
parser.add_argument(
    "--batch_size", type=int, default=100, required=False,
    help="Checkpoint path")
parser.add_argument(
    "--balance_dfr_val", type=bool, default=True, required=False,
    help="Subset validation to have equal groups for DFR(Val)")
parser.add_argument(
    "--notrain_dfr_val", type=bool, default=True, required=False,
    help="Do not add train data for DFR(Val)")
parser.add_argument(
    "--tune_class_weights_dfr_train", action='store_true',
    help="Learn class weights for DFR(Train)")
parser.add_argument(
    "--seed", type=int, default=0, required=False, help="Random seed for reproducibility")
parser.add_argument(
    "--skip_dfr_train_subset_tune", type=bool, default=False, required=False, help="Skip dfr_train_subset_tune, so set best_hyper = [1.0, 1.0, 1.0]")
parser.add_argument(
    "--model_type", type=str, default="resnet50", required=False, help="pick model type: resnet50 or vit_b_16")



args = parser.parse_args()
set_seed(args.seed)

def dfr_on_validation_tune(
        all_embeddings, all_y, all_g, preprocess=True,
        balance_val=False, add_train=True, num_retrains=1):

    worst_accs = {}
    for i in range(num_retrains):
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1

        n_val = len(x_val) // 2
        idx = np.arange(len(x_val))
        np.random.shuffle(idx)

        x_valtrain = x_val[idx[n_val:]]
        y_valtrain = y_val[idx[n_val:]]
        g_valtrain = g_val[idx[n_val:]]

        n_groups = np.max(g_valtrain) + 1
        g_idx = [np.where(g_valtrain == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        if balance_val:
            x_valtrain = np.concatenate([x_valtrain[g[:min_g]] for g in g_idx])
            y_valtrain = np.concatenate([y_valtrain[g[:min_g]] for g in g_idx])
            g_valtrain = np.concatenate([g_valtrain[g[:min_g]] for g in g_idx])

        x_val = x_val[idx[:n_val]]
        y_val = y_val[idx[:n_val]]
        g_val = g_val[idx[:n_val]]

        n_train = len(x_valtrain) if add_train else 0

        x_train = np.concatenate([all_embeddings["train"][:n_train], x_valtrain])
        y_train = np.concatenate([all_y["train"][:n_train], y_valtrain])
        g_train = np.concatenate([all_g["train"][:n_train], g_valtrain])
        print(np.bincount(g_train))
        if preprocess:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)


        if balance_val and not add_train:
            cls_w_options = [{0: 1., 1: 1.}]
        else:
            cls_w_options = CLASS_WEIGHT_OPTIONS
        for c in C_OPTIONS:
            for class_weight in cls_w_options:
                logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                            class_weight=class_weight)
                logreg.fit(x_train, y_train)
                preds_val = logreg.predict(x_val)
                group_accs = np.array(
                    [(preds_val == y_val)[g_val == g].mean()
                     for g in range(n_groups)])
                worst_acc = np.min(group_accs)
                if i == 0:
                    worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
                else:
                    worst_accs[c, class_weight[0], class_weight[1]] += worst_acc
                # print(c, class_weight[0], class_weight[1], worst_acc, worst_accs[c, class_weight[0], class_weight[1]])
    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval(
        c, w1, w2, all_embeddings, all_y, all_g, num_retrains=20,
        preprocess=True, balance_val=False, add_train=True):
    coefs, intercepts = [], []
    if preprocess:
        scaler = StandardScaler()
        scaler.fit(all_embeddings["train"])

    for i in range(num_retrains):
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1
        g_idx = [np.where(g_val == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        if balance_val:
            x_val = np.concatenate([x_val[g[:min_g]] for g in g_idx])
            y_val = np.concatenate([y_val[g[:min_g]] for g in g_idx])
            g_val = np.concatenate([g_val[g[:min_g]] for g in g_idx])

        n_train = len(x_val) if add_train else 0
        train_idx = np.arange(len(all_embeddings["train"]))
        np.random.shuffle(train_idx)
        train_idx = train_idx[:n_train]

        x_train = np.concatenate(
            [all_embeddings["train"][train_idx], x_val])
        y_train = np.concatenate([all_y["train"][train_idx], y_val])
        g_train = np.concatenate([all_g["train"][train_idx], g_val])
        print(np.bincount(g_train))
        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                    class_weight={0: w1, 1: w2})
        logreg.fit(x_train, y_train)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    g_test = all_g["test"]
    print(np.bincount(g_test))

    if preprocess:
        x_test = scaler.transform(x_test)
    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                class_weight={0: w1, 1: w2})
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)
    preds_test = logreg.predict(x_test)
    preds_train = logreg.predict(x_train)
    n_groups = np.max(g_train) + 1
    test_accs = [(preds_test == y_test)[g_test == g].mean()
                 for g in range(n_groups)]
    test_mean_acc = (preds_test == y_test).mean()
    train_accs = [(preds_train == y_train)[g_train == g].mean()
                  for g in range(n_groups)]
    return test_accs, test_mean_acc, train_accs


def dfr_train_subset_tune(
        all_embeddings, all_y, all_g, preprocess=True,
        learn_class_weights=False):

    if args.skip_dfr_train_subset_tune:
      return [1.0,1.0,1.0]

    x_val = all_embeddings["val"]
    y_val = all_y["val"]
    g_val = all_g["val"]

    x_train = all_embeddings["train"]
    y_train = all_y["train"]
    g_train = all_g["train"]

    if preprocess:
        scaler = StandardScaler()
        scaler.fit(x_train)

    n_groups = np.max(g_train) + 1
    g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
    for g in g_idx:
        np.random.shuffle(g)
    min_g = np.min([len(g) for g in g_idx])
    x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
    y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
    g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])
    print(np.bincount(g_train))
    if preprocess:
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)

    worst_accs = {}
    if learn_class_weights:
        cls_w_options = CLASS_WEIGHT_OPTIONS
    else:
        cls_w_options = [{0: 1., 1: 1.}]
    for c in C_OPTIONS:
        for class_weight in cls_w_options:
            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                        class_weight=class_weight, max_iter=20)
            logreg.fit(x_train, y_train)
            preds_val = logreg.predict(x_val)
            group_accs = np.array(
                [(preds_val == y_val)[g_val == g].mean() for g in range(n_groups)])
            worst_acc = np.min(group_accs)
            worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
            print(c, class_weight, worst_acc, group_accs)

    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]


    return best_hypers


def dfr_train_subset_eval(
        c, w1, w2, all_embeddings, all_y, all_g, num_retrains=10,
        preprocess=True):
    coefs, intercepts = [], []
    x_train = all_embeddings["train"]
    scaler = StandardScaler()
    scaler.fit(x_train)

    for i in range(num_retrains):
        x_train = all_embeddings["train"]
        y_train = all_y["train"]
        g_train = all_g["train"]
        n_groups = np.max(g_train) + 1

        g_idx = [np.where(g_train == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        x_train = np.concatenate([x_train[g[:min_g]] for g in g_idx])
        y_train = np.concatenate([y_train[g[:min_g]] for g in g_idx])
        g_train = np.concatenate([g_train[g[:min_g]] for g in g_idx])
        print(np.bincount(g_train))

        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                        class_weight={0: w1, 1: w2})
        logreg.fit(x_train, y_train)

        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    g_test = all_g["test"]

    if preprocess:
        x_test = scaler.transform(x_test)

    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)

    preds_test = logreg.predict(x_test)
    preds_train = logreg.predict(x_train)
    n_groups = np.max(g_train) + 1
    test_accs = [(preds_test == y_test)[g_test == g].mean()
                 for g in range(n_groups)]
    test_mean_acc = (preds_test == y_test).mean()
    train_accs = [(preds_train == y_train)[g_train == g].mean()
                  for g in range(n_groups)]
    return test_accs, test_mean_acc, train_accs


def retrain_all_linear_layers(
        model, train_loader, criterion, num_epochs=100, learning_rate=0.001):
    """
    Retrains all linear layers of vit_b_16, with group reweighting.

    Args:
    - model
    - train_loader
    - criterion: Loss function
    - num_epochs
    - learning_rate: Learning rate for the optimizer.

    Returns:
    - model
    """

    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if "mlp" in name and isinstance(module, torch.nn.Linear):
            for param in module.parameters():
                param.requires_grad = True

    for name, module in model.named_modules():
        if "encoder.ln" in name and isinstance(module, torch.nn.LayerNorm):
            for param in module.parameters():
                param.requires_grad = True

    for param in model.heads.head.parameters():
        param.requires_grad = True


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    x_train_list, y_train_list, g_train_list = [], [], []
    for x, y, g, _ in train_loader:
        x_train_list.append(x)
        y_train_list.append(y)
        g_train_list.append(g)

    x_train = torch.cat(x_train_list, dim=0)
    y_train = torch.cat(y_train_list, dim=0)
    g_train = torch.cat(g_train_list, dim=0)

    n_groups = torch.max(g_train).item() + 1

    g_idx = [torch.where(g_train == g)[0] for g in range(n_groups)]
    min_g = min([len(g) for g in g_idx])
    for g in g_idx:
        indices = torch.randperm(len(g))
        g[:] = g[indices]
    x_train = torch.cat([x_train[g[:min_g]] for g in g_idx], dim=0)
    y_train = torch.cat([y_train[g[:min_g]] for g in g_idx], dim=0)
    g_train = torch.cat([g_train[g[:min_g]] for g in g_idx], dim=0)

    print(f"Balanced group sizes: {torch.bincount(g_train)}")

    # training loop
    model.train()
    for epoch in tqdm.tqdm(range(num_epochs)):
        epoch_loss = 0.0
        n = 0
        for i in range(0, len(x_train), train_loader.batch_size):
            x_batch = x_train[i:i + train_loader.batch_size].cuda()
            y_batch = y_train[i:i + train_loader.batch_size].cuda()

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        avg_loss = epoch_loss / n
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")


    return model



def evaluate_model(model, train_loader, test_loader):
    """
    Evaluates the model on the training and test sets, calculating group-wise accuracies.

    Args:
    - model
    - train_loader
    - test_loader

    Returns:
    - test_accs
    - test_mean_acc
    - train_accs
    """

    model.eval()

    y_train_list, g_train_list, preds_train_list = [], [], []
    with torch.no_grad():
        for x, y, g, _ in train_loader:
            x, y, g = x.cuda(), y.cuda(), g.cuda()
            outputs = model(x)
            preds_train = torch.argmax(outputs, dim=1)
            preds_train_list.append(preds_train.cpu())
            y_train_list.append(y.cpu())
            g_train_list.append(g.cpu())

    y_train = torch.cat(y_train_list, dim=0)
    g_train = torch.cat(g_train_list, dim=0)
    preds_train = torch.cat(preds_train_list, dim=0)

    y_test_list, g_test_list, preds_test_list = [], [], []
    with torch.no_grad():
        for x, y, g, _ in test_loader:
            x, y, g = x.cuda(), y.cuda(), g.cuda()
            outputs = model(x)
            preds_test = torch.argmax(outputs, dim=1)
            preds_test_list.append(preds_test.cpu())
            y_test_list.append(y.cpu())
            g_test_list.append(g.cpu())

    y_test = torch.cat(y_test_list, dim=0)
    g_test = torch.cat(g_test_list, dim=0)
    preds_test = torch.cat(preds_test_list, dim=0)

    n_groups = torch.max(g_train).item() + 1

    test_accs = [(preds_test == y_test)[g_test == g].float().mean().item() for g in range(n_groups)]
    test_mean_acc = (preds_test == y_test).float().mean().item()

    train_accs = [(preds_train == y_train)[g_train == g].float().mean().item() for g in range(n_groups)]

    return test_accs, test_mean_acc, train_accs


## Load data
target_resolution = (224, 224)
train_transform = get_transform_cub(target_resolution=target_resolution,
                                    train=True, augment_data=False)
test_transform = get_transform_cub(target_resolution=target_resolution,
                                   train=False, augment_data=False)

trainset = WaterBirdsDataset(
    basedir=args.data_dir, split="train", transform=train_transform)
testset = WaterBirdsDataset(
    basedir=args.data_dir, split="test", transform=test_transform)
valset = WaterBirdsDataset(
    basedir=args.data_dir, split="val", transform=test_transform)

loader_kwargs = {'batch_size': args.batch_size,
                 'num_workers': 4, 'pin_memory': True,
                 "reweight_places": None}
train_loader = get_loader(
    trainset, train=True, reweight_groups=False, reweight_classes=False,
    **loader_kwargs)
test_loader = get_loader(
    testset, train=False, reweight_groups=None, reweight_classes=None,
    **loader_kwargs)
val_loader = get_loader(
    valset, train=False, reweight_groups=None, reweight_classes=None,
    **loader_kwargs)



if args.model_type == "resnet50":
    # Evaluate model
    print("Model: ResNet50")

    # Load model
    n_classes = trainset.n_classes
    model = torchvision.models.resnet50(pretrained=False)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, n_classes)
    model.load_state_dict(torch.load(
        args.ckpt_path
    ))
    model.cuda()
    model.eval()


    print("Base Model")
    base_model_results = {}
    get_yp_func = partial(get_y_p, n_places=trainset.n_places)
    base_model_results["test"] = evaluate(model, test_loader, get_yp_func)
    base_model_results["val"] = evaluate(model, val_loader, get_yp_func)
    base_model_results["train"] = evaluate(model, train_loader, get_yp_func)
    print(base_model_results)
    print()

    model.eval()

    # Extract embeddings
    def get_embed(m, x):
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x


if args.model_type == "vit_b_16":
    print("Model: Vit-B_16")

    # Load model
    n_classes = trainset.n_classes
    model = vit_b_16(weights='DEFAULT')
    d = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(d, n_classes)
    model.load_state_dict(torch.load(
        args.ckpt_path
    ))
    model.cuda()
    model.eval()

    # Evaluate model
    print("Base Model")
    base_model_results = {}
    get_yp_func = partial(get_y_p, n_places=trainset.n_places)
    base_model_results["test"] = evaluate(model, test_loader, get_yp_func)
    base_model_results["val"] = evaluate(model, val_loader, get_yp_func)
    base_model_results["train"] = evaluate(model, train_loader, get_yp_func)
    print(base_model_results)
    print()

    model.eval()


    # Extract embeddings
    def get_embed(m, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        b, c, fh, fw = x.shape

        x = m.conv_proj(x)

        x = x.flatten(2).transpose(1, 2)
        if hasattr(m, 'class_token'):
            x = torch.cat((m.class_token.expand(b, -1, -1), x), dim=1)
        if hasattr(m, 'pos_embed'):
            x = x + m.pos_embed
        x = m.encoder(x)
        if hasattr(m, 'pre_logits'):
            x = m.pre_logits(x)
            x = torch.tanh(x)
        if hasattr(m, 'norm'):
            x = m.norm(x)[:, 0]
        else:
            x = x[:,0]
        x = x.view(x.size(0), -1)
        return x


all_embeddings = {}
all_y, all_p, all_g = {}, {}, {}
for name, loader in [("train", train_loader), ("test", test_loader), ("val", val_loader)]:
    all_embeddings[name] = []
    all_y[name], all_p[name], all_g[name] = [], [], []
    for x, y, g, p in tqdm.tqdm(loader):
        with torch.no_grad():

            all_embeddings[name].append(get_embed(model, x.cuda()).detach().cpu().numpy())
            all_y[name].append(y.detach().cpu().numpy())
            all_g[name].append(g.detach().cpu().numpy())
            all_p[name].append(p.detach().cpu().numpy())
    all_embeddings[name] = np.vstack(all_embeddings[name])
    all_y[name] = np.concatenate(all_y[name])
    all_g[name] = np.concatenate(all_g[name])
    all_p[name] = np.concatenate(all_p[name])



# DFR on all linear layers
print("DFR on all linear layers")
retrain_train_results = {}
model = retrain_all_linear_layers(
    model=model,
    train_loader=train_loader,
    criterion=torch.nn.CrossEntropyLoss(),
    num_epochs=10,
    learning_rate=0.001
)

test_accs, test_mean_acc, train_accs = evaluate_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader
)
retrain_train_results["test_accs"] = test_accs
retrain_train_results["train_accs"] = train_accs
retrain_train_results["test_worst_acc"] = np.min(test_accs)
retrain_train_results["test_mean_acc"] = test_mean_acc

# Print the results
print("Retrain Results:")
print(retrain_train_results)
print()


# DFR on validation
print("DFR on validation")
dfr_val_results = {}
c, w1, w2 = dfr_on_validation_tune(
    all_embeddings, all_y, all_g,
    balance_val=args.balance_dfr_val, add_train=not args.notrain_dfr_val)
dfr_val_results["best_hypers"] = (c, w1, w2)
print("Hypers:", (c, w1, w2))
test_accs, test_mean_acc, train_accs = dfr_on_validation_eval(
        c, w1, w2, all_embeddings, all_y, all_g,
    balance_val=args.balance_dfr_val, add_train=not args.notrain_dfr_val)
dfr_val_results["test_accs"] = test_accs
dfr_val_results["train_accs"] = train_accs
dfr_val_results["test_worst_acc"] = np.min(test_accs)
dfr_val_results["test_mean_acc"] = test_mean_acc
print(dfr_val_results)
print()

# DFR on train subsampled
print("DFR on train subsampled")
dfr_train_results = {}
c, w1, w2 = dfr_train_subset_tune(
    all_embeddings, all_y, all_g,
    learn_class_weights=args.tune_class_weights_dfr_train)
dfr_train_results["best_hypers"] = (c, w1, w2)
print("Hypers:", (c, w1, w2))
test_accs, test_mean_acc, train_accs = dfr_train_subset_eval(
        c, w1, w2, all_embeddings, all_y, all_g)
dfr_train_results["test_accs"] = test_accs
dfr_train_results["train_accs"] = train_accs
dfr_train_results["test_worst_acc"] = np.min(test_accs)
dfr_train_results["test_mean_acc"] = test_mean_acc
print(dfr_train_results)
print()



all_results = {}
all_results["base_model_results"] = base_model_results
all_results["dfr_val_results"] = dfr_val_results
all_results["dfr_train_results"] = dfr_train_results
print(all_results)


with open(args.result_path, 'wb') as f:
    pickle.dump(all_results, f)