import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from utils.train_utils import validate

from dataset import PolypDataset
from unet import get_unet_model



#from utils.dataset import PolypDataset -> this caused an error for me
#from models.unet import get_unet_model


def main():
    # 1) Select device (use GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 2) Create dataset
    dataset = PolypDataset(
        images_dir="data/kvasir/images",
        masks_dir="data/kvasir/masks",
        transform=None
    )

    # If dataset is empty warn
    if len(dataset) == 0:
        print("Warning: Dataset appears empty! Are there files in data/kvasir/images and masks?")
        return

    # 3) Train / Validation split (80% / 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")


    # 4) Get model
    model = get_unet_model().to(device)

    # 5) Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 6) Training loop (few epochs for testing)
    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            preds = model(images)

            # Loss
            loss = criterion(preds, masks)

            # Backpropagation + weight update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device
        )

        print(f"""
        Epoch [{epoch + 1}/{num_epochs}]
        Train Loss: {avg_train_loss:.4f}
        Val Loss:   {val_loss:.4f}
        Dice:       {val_dice:.4f}
        IoU:        {val_iou:.4f}
        """)



    # 7) Save model
    torch.save(model.state_dict(), "models/unet_polyp.pth")
    print("Model saved: models/unet_polyp.pth")


if __name__ == "__main__":
    main()

