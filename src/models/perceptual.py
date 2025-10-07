import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class PerceptualModule(nn.Module):
    """
    Perceptual module wrapping a pre-trained ResNet50.
    Extracts activations from the penultimate conv stage (layer4 output) and
    applies a 1x1 point-wise convolution to reduce channels from 2048 to
    the desired embedding size (typically the RNN hidden size), followed by
    global average pooling to produce a vector per image.

    Args:
        out_channels: Desired output dimensionality (e.g., RNN hidden size)
        pretrained: If True, load ImageNet pre-trained weights. If weights
            are unavailable (e.g., offline), will fallback to random init.
        freeze_backbone: If True, freeze the ResNet backbone params.
        capture_exact_layer42_relu: If True, register a forward hook to capture
            the output of layer4[2].relu explicitly (identical numerically to
            layer4 output, but matches the paper's exact naming).
    """

    def __init__(self, out_channels: int, pretrained: bool = True, freeze_backbone: bool = True,
                 capture_exact_layer42_relu: bool = False):
        super().__init__()

        weights = None
        if pretrained:
            # Torchvision >= 0.13 uses the 'weights' API
            try:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.IMAGENET1K_V2
            except Exception:
                weights = None

        try:
            self.backbone = models.resnet50(weights=weights)
        except Exception:
            # Fallback to non-pretrained if weights download fails
            self.backbone = models.resnet50(weights=None)

        # Keep only up to layer4 (exclude avgpool and fc)
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # 1x1 conv to reduce channels 2048 -> out_channels
        self.reduce = nn.Conv2d(2048, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        # Global average pooling to (B, C, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Optional exact capture of layer4[2].relu
        self.capture_exact_layer42_relu = capture_exact_layer42_relu
        self._hook_handle = None
        self._hook_feat: Optional[torch.Tensor] = None
        if self.capture_exact_layer42_relu:
            self._register_layer42_relu_hook()

    def forward(self, x: torch.Tensor, return_feature_map: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input images of shape (B, 3, H, W)
            return_feature_map: If True, also return the reduced feature map before pooling.
        Returns:
            embedding: Tensor of shape (B, out_channels)
            feature_map (optional): Tensor of shape (B, out_channels, H', W')
        """
        if self.capture_exact_layer42_relu:
            # Reset buffer and run forward to trigger hook
            self._hook_feat = None
            feat_full = self.feature_extractor(x)  # triggers hook on layer4[2].relu
            feat = self._hook_feat if self._hook_feat is not None else feat_full
        else:
            feat = self.feature_extractor(x)  # (B, 2048, H', W') from layer4 output
        reduced = self.reduce(feat)       # (B, out_channels, H', W')
        pooled = self.gap(reduced)        # (B, out_channels, 1, 1)
        embedding = pooled.flatten(1)     # (B, out_channels)

        if return_feature_map:
            return embedding, reduced
        return embedding, None

    def _register_layer42_relu_hook(self):
        # Register a forward hook on the exact target module
        try:
            target = self.backbone.layer4[2].relu
        except Exception as e:
            # If unexpected backbone structure, skip hooking silently
            return

        def _save_hook(module, inputs, output):
            self._hook_feat = output

        # Remove existing hook if any
        if self._hook_handle is not None:
            try:
                self._hook_handle.remove()
            except Exception:
                pass
            self._hook_handle = None

        self._hook_handle = target.register_forward_hook(_save_hook)

    def enable_exact_layer42_relu(self, enable: bool = True):
        """Toggle capturing the exact layer4[2].relu activation via forward hook."""
        if enable and not self.capture_exact_layer42_relu:
            self.capture_exact_layer42_relu = True
            self._register_layer42_relu_hook()
        elif not enable and self.capture_exact_layer42_relu:
            self.capture_exact_layer42_relu = False
            if self._hook_handle is not None:
                try:
                    self._hook_handle.remove()
                except Exception:
                    pass
                self._hook_handle = None
