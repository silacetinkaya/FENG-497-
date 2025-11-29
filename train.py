import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from utils.dataset import PolypDataset
from models.unet import get_unet_model


def main():
    # 1) Cihaz seç (GPU varsa GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 2) Dataset oluştur
    train_dataset = PolypDataset(
        images_dir="data/kvasir/images",
        masks_dir="data/kvasir/masks",
        transform=None  # ilk versiyonda veri artırma yok
    )

    # Eğer dataset boşsa uyarı verelim
    if len(train_dataset) == 0:
        print("Uyarı: Dataset boş görünüyor! data/kvasir/images ve masks içinde dosya var mı?")
        return

    # 3) DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 4) Modeli al
    model = get_unet_model().to(device)

    # 5) Loss ve optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 6) Eğitim döngüsü (deneme için az epoch)
    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            # İleri yayılım
            preds = model(images)

            # Loss
            loss = criterion(preds, masks)

            # Geri yayılım + ağırlık güncelleme
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    # 7) Modeli kaydet
    torch.save(model.state_dict(), "models/unet_polyp.pth")
    print("Model kaydedildi: models/unet_polyp.pth")


if __name__ == "__main__":
    main()

