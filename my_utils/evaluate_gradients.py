from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class AvgGradientByGroup(object):
    """
    Compute the average gradient with regard to neuron components across groups
    """

    def __init__(self, group):
        print("init, group: ", group)
        self.group = group
        self.reset()

    def reset(self):
        print("reset")
        self.total = 0

        self.gradient_dict = defaultdict(lambda: 0)

    def update(self, norms, batch_size):
        self.total += batch_size
        for name, norm in norms:
            self.gradient_dict[name] += norm * batch_size

    def compute(self):
        for k, v in self.gradient_dict.items():
            self.gradient_dict[k] /= self.total
        results = {
            "group": self.group,
            "n_total": self.total,
            "gradients": self.gradient_dict

        }
        return results


def evaluate(model, dataloader, metric, group, device):
    model.eval()
    metric.reset()

    criterion = torch.nn.CrossEntropyLoss()

    for imgs, ys, gs, _ in tqdm(dataloader, desc="Evaluating", unit="batch"):
        imgs, ys, gs = imgs.to(device), ys.to(device), gs.to(device)

        group_mask = (gs == group).to(device)  # [B]

        imgs, ys = imgs[group_mask], ys[group_mask]
        if imgs.size(0) == 0:
            continue

        imgs.requires_grad_(True)
        outs = model(imgs)  # [number of group members, classes]

        loss = criterion(outs, ys)

        model.zero_grad()
        loss.backward()

        norms = []
        for name, param in model.named_parameters():
            if param.requires_grad and "weight" in name and param.dim() == 2:
                norms.append((name, torch.norm(param.grad, p=2, dim=1).detach()))

        metric.update(norms, imgs.size(0))

    return metric.compute()
