import torch
from utils.metrics import dice_score, iou_score

def validate(model, val_loader, criterion, device):
    model.eval()

    val_loss = 0.0
    dice_total = 0.0
    iou_total = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            dice_total += dice_score(outputs, masks)
            iou_total += iou_score(outputs, masks)

    return (
        val_loss / len(val_loader),
        dice_total / len(val_loader),
        iou_total / len(val_loader)
    )
