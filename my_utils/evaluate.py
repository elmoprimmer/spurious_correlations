import torch
import numpy as np
from tqdm import tqdm


class AccuracyWithGroups(object):

    def __init__(self, n_groups=4):
        print("init")
        self.correct = None
        self.total = None
        self.correct_per_group = None
        self.total_per_group = None
        self.n_groups = n_groups
        self.reset()

    def reset(self):
        print("reset")
        self.correct = 0
        self.total = 0

        self.correct_per_group = {g: 0 for g in range(self.n_groups)}
        self.total_per_group = {g: 0 for g in range(self.n_groups)}

    @torch.no_grad()
    def update(self, outs, ys, groups):
        preds = torch.argmax(outs, dim=1)
        correct_mask = (preds == ys)

        self.correct += correct_mask.sum().item()
        self.total += ys.size(0)

        for group in range(self.n_groups):
            group_mask = (groups == group)
            group_correct = (correct_mask & group_mask).sum().item()
            group_total = group_mask.sum().item()

            self.correct_per_group[group] += group_correct
            self.total_per_group[group] += group_total

    def compute(self):
        results = {
            "n_correct": self.correct,
            "n_total": self.total,
            "correct_per_group": self.correct_per_group,
            "total_per_group": self.total_per_group,
            "total_accuracy": self.correct / self.total if self.total else 0,
            "groups_accuracy": {}
        }
        for k, v in self.correct_per_group.items():
            total_k = self.total_per_group[k]
            results['groups_accuracy'][k] = v / total_k if total_k else 0
        return results


def evaluate(model, dataloader, metric, device):
    model.eval()
    metric.reset()

    with torch.no_grad():
        for imgs, ys, gs, _ in tqdm(dataloader, desc="Evaluating", unit="batch"):
            imgs, ys, gs = imgs.to(device), ys.to(device), gs.to(device)
            outs = model(imgs)
            metric.update(outs, ys, gs)

    return metric.compute()


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_y_p(g, n_places):
    y = g // n_places
    p = g % n_places
    return y, p
