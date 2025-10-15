import argparse, hashlib, yaml, torch
from models import ModelLoader
from metrics import compute_accuracy
from my_datasets import WaterBirds, get_sample_indices_for_group, WaterBirdSubset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    print('checkpoint sha1:',
          hashlib.sha1(open(args.ckpt,'rb').read()).hexdigest())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    val_set = WaterBirds(args.dataset_path).get_valid_set()

    validation_indices = get_sample_indices_for_group(  # these are just used for printing acc
        val_set, 'all', "cuda"
    )



    print('validation indices len:', len(validation_indices))

    subset = WaterBirdSubset(val_set, validation_indices)

    val_loader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch, shuffle=False, num_workers=0)

    model = ModelLoader.get_basic_model(
        'vit_b_16', args.ckpt, device, num_classes=2)
    model.eval()

    top1 = compute_accuracy(model, val_loader, device)
    print(f'plain val accuracy = {top1:.2f} %')

if __name__ == '__main__':
    main()