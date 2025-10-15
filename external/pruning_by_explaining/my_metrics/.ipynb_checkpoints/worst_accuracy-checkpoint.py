import torch
from collections import defaultdict
from tqdm.auto import tqdm


def compute_worst_accuracy(model, dataloader, device):
    """
    Computes the worst group accuracy of the model on the given dataloader.

    Args:
        model (torch.nn.Module): model to evaluate
        dataloader (torch.utils.data.DataLoader): dataloader to evaluate the model
        device (torch.device): device to use
    Returns:
        float: worst accuracy
    """

    model.eval()
    model.to(device)

    correct_dict = defaultdict(int)
    total_dict = defaultdict(int)
    with torch.no_grad():
        for batch in tqdm(
                dataloader,
                total=len(dataloader),
                desc="evaluating group acc",
                leave=False
        ):
            images, labels, groups = batch
            images = images.to(device)
            labels = labels.to(device)
            groups = groups.to(device)

            if groups.dim() == 0:
                groups = groups.unsqueeze(0)
            elif groups.dim() > 1:
                groups = groups.view(-1)

            outputs = model(images)

            # Calculate worst group accuracy
            _, predicted = outputs.topk(1, dim=1)
            correct = predicted.eq(labels.view(-1, 1)).flatten()

            for group, is_correct in zip(groups, correct):
                correct_dict[group.item()] += is_correct.item()
                total_dict[group.item()] += 1

    group_accuracies = {group: correct_dict[group] / total_dict[group] for group in total_dict}
    worst_accuracy = min(group_accuracies.values())

    del (
        labels,
        images,
        correct_dict,
        total_dict,
        outputs,
        predicted,
    )
    return worst_accuracy, group_accuracies
