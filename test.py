import torch
from torch.utils.data import DataLoader

from unet import get_unet_model
from dataset import PolypDataset
from utils.metrics import dice_score, iou_score


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # CVC TEST DATASET
    test_dataset = PolypDataset(
        images_dir="data/cvc_test/images",
        masks_dir="data/cvc_test/masks",
        transform=None
    )

    if len(test_dataset) == 0:
        print("Warning: CVC test dataset is empty!")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )

    print(f"CVC Test samples: {len(test_dataset)}")

    # LOAD MODEL (trained on Kvasir)
    model = get_unet_model().to(device)
    model.load_state_dict(
        torch.load("models/unet_polyp.pth", map_location=device)
    )
    model.eval()

    # METRIC SUMS
    dice_total = 0.0
    iou_total = 0.0

    # TEST LOOP (single pass)
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            dice_total += dice_score(outputs, masks)
            iou_total += iou_score(outputs, masks)

    avg_dice = dice_total / len(test_loader)
    avg_iou = iou_total / len(test_loader)

    print("\nCVC TEST RESULTS (Trained on Kvasir)")
    print(f"Dice Score: {avg_dice:.4f}")
    print(f"IoU Score:  {avg_iou:.4f}")


if __name__ == "__main__":
    main()
