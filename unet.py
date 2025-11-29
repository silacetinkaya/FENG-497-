import segmentation_models_pytorch as smp

def get_unet_model():
    model = smp.Unet(
        encoder_name="resnet34",      # backbone
        encoder_weights="imagenet",   # önceden eğitilmiş ağırlıklar
        in_channels=3,
        classes=1
    )
    return model

