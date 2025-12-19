import os
import cv2
import torch
import numpy as np


from unet import get_unet_model


def load_image(path, size=(256, 256)):
    """Read image, convert to RGB, resize and prepare tensor."""
    image_bgr = cv2.imread(path)
    if image_bgr is None:
        raise ValueError(f"Image could not be read: {path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, size)

    # [H, W, C] -> [1, C, H, W]
    img_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # batch dimension

    return img_tensor, image_rgb  # tensor, RGB image for display


def save_mask_overlay(image_rgb, mask, save_path="overlay_result.png"):
    """
    Apply the mask as a red overlay on the original image and save.
    image_rgb: [H, W, 3] (RGB)
    mask: [H, W] 0-1
    """
    # Scale mask to 0-255
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Create red overlay
    mask_color = np.zeros_like(image_rgb)
    mask_color[:, :, 0] = mask_uint8  # fill R channel (red)

    # Original + mask overlay
    overlay = cv2.addWeighted(image_rgb, 0.7, mask_color, 0.3, 0)

    # Convert to BGR for OpenCV and save
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, overlay_bgr)
    print(f"Overlay saved: {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1) Modeli yükle
    model = get_unet_model().to(device)
    model.load_state_dict(torch.load("models/unet_polyp.pth", map_location=device))
    model.eval()
    print("Model loaded.")

    # Folder for overlay results
    os.makedirs("results/overlays", exist_ok=True)


    images_dir = "data/cvc_test/images"
    test_image_paths = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not test_image_paths:
        print("Warning: no images found in data/cvc_test/images!")
        return

    print(f"{len(test_image_paths)} test images found.")


    for test_image_path in test_image_paths:
        print(f"\nProcessing: {test_image_path}")

        # Load image
        img_tensor, img_rgb = load_image(test_image_path)
        img_tensor = img_tensor.to(device)

        # Get prediction from model
        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)
            pred = pred.squeeze().cpu().numpy()

        # Apply threshold
        mask = (pred > 0.5).astype(np.float32)

        # Filename to save
        image_name = os.path.splitext(os.path.basename(test_image_path))[0]
        save_path = f"results/overlays/{image_name}_overlay.png"

        # Save overlay
        save_mask_overlay(img_rgb, mask, save_path)

    print("\nAll test images processed.")


if __name__ == "__main__":
    main()
