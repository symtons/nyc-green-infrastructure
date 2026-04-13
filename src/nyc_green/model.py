"""Step 6 — U-Net model builder with ImageNet-pretrained ResNet18 encoder.

Handles the 5-channel input hack: ImageNet is 3-channel RGB, but we have
5 channels (B, G, R, NIR, NDVI). We expand the first conv layer of the
pretrained encoder so that:
  - channels 0-2 (BGR) get the pretrained ImageNet weights
  - channels 3-4 (NIR, NDVI) get Kaiming-initialized fresh weights
"""
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_unet(
    num_classes: int = 3,
    in_channels: int = 5,
    encoder_name: str = "resnet18",
    encoder_weights: str = "imagenet",
) -> nn.Module:
    """Construct a U-Net with a pretrained encoder and expanded input layer."""

    # Build with 3-channel input to get the pretrained weights
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )

    # Replace the first conv layer with a 5-channel version.
    # Channels 0-2 get pretrained RGB weights; channels 3-4 get Kaiming init.
    first_conv = model.encoder.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=(first_conv.bias is not None),
    )

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = first_conv.weight.clone()
        nn.init.kaiming_normal_(
            new_conv.weight[:, 3:, :, :],
            mode="fan_out",
            nonlinearity="relu",
        )
        # Scale extra channels so overall activation magnitude is preserved
        new_conv.weight[:, 3:, :, :] *= (3.0 / in_channels)

        if first_conv.bias is not None:
            new_conv.bias.copy_(first_conv.bias)

    model.encoder.conv1 = new_conv
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)