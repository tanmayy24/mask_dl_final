import argparse
import torch
import os
import tqdm

from maskpredformer.unet_inference import WenmaSet, DEFAULT_TRANSFORM, get_inference


def main(args):
    train_set_pred = torch.load("data/DL/train_masks.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.jit.load(args.model_checkpoint).to(device)
    dataset = WenmaSet(
        os.path.join(args.data_root, args.split),
        args.split,
        transform=DEFAULT_TRANSFORM,
    )
    print("Dataset length: ", len(dataset))

    all_preds = []
    for i in tqdm.tqdm(range(len(dataset))):
        imgs, _ = dataset[i]
        all_imgs = []
        for j in range(22):
            image = imgs[j].unsqueeze(0).to(device)
            all_imgs.append(image)
        all_imgs = torch.cat(all_imgs, dim=0)
        pred = get_inference(model, all_imgs)
        all_preds.append(pred)

    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
    print("Saving predictions to", args.output_file)
    torch.save(torch.stack(all_preds), args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="checkpoints/unet9.pt",
        help="Model to use",
    )
    parser.add_argument("--data_root", type=str, default="data/Dataset_Student")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_file", type=str, default="data/DL/train_masks.pt")

    args = parser.parse_args()

    main(args)
