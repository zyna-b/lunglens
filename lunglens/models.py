"""The three models LungLens compares.

Every model takes grayscale chest X-rays as floats in [0, 1], shape [B, 1, H, W], does its own
normalisation, and ends in global average pooling + one linear layer. That last detail means the
class activation map (CAM, Zhou et al. 2016) falls out of the same forward pass, so the browser can
show where the model looked without computing gradients.

  cnn_scratch           a 5-block CNN trained from random weights (the baseline)
  densenet121_imagenet  DenseNet-121 pretrained on ImageNet photos
  densenet121_cxr       the same DenseNet-121, pretrained on chest X-rays from 7 public datasets (TorchXRayVision "all")

The two DenseNets are identical apart from their starting weights, so comparing them isolates one
question: does it help to start from a model that has already seen chest X-rays?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

CLASSES = ["healthy", "sick_but_not_tb", "tb"]
MODELS = ["cnn_scratch", "densenet121_imagenet", "densenet121_cxr"]


class ImageNetPrep(nn.Module):
    """Gray [0,1] -> 3 channels normalised the way ImageNet models expect."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        return (x.repeat(1, 3, 1, 1) - self.mean) / self.std


class XrvPrep(nn.Module):
    """Gray [0,1] -> [-1024, 1024], the range TorchXRayVision models were trained on."""

    def forward(self, x):
        return x * 2048.0 - 1024.0


class CenterPrep(nn.Module):
    def forward(self, x):
        return (x - 0.5) / 0.25


class CamNet(nn.Module):
    """backbone feature map -> ReLU -> global average pool -> linear."""

    def __init__(self, prep, features, channels, n_classes=len(CLASSES)):
        super().__init__()
        self.prep = prep
        self.features = features
        self.head = nn.Linear(channels, n_classes)

    def feature_map(self, x):
        return F.relu(self.features(self.prep(x)))

    def forward(self, x):
        return self.head(self.feature_map(x).mean((2, 3)))

    def forward_with_cam(self, x):
        """Logits plus one activation map per class, [B, n_classes, h, w] (h = w = input / 32)."""
        f = self.feature_map(x)
        logits = self.head(f.mean((2, 3)))
        cam = torch.einsum("kc,bchw->bkhw", self.head.weight, f) + self.head.bias.view(1, -1, 1, 1)
        return logits, cam


def _scratch_features():
    def block(i, o):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256), block(256, 256))


def _densenet_features(in_channels):
    f = torchvision.models.densenet121(weights=None).features
    if in_channels == 1:
        f.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return f


def build_model(name, pretrained=False):
    """pretrained=True downloads the starting weights (only needed for training)."""
    if name == "cnn_scratch":
        return CamNet(CenterPrep(), _scratch_features(), 256)
    if name == "densenet121_imagenet":
        w = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        return CamNet(ImageNetPrep(), torchvision.models.densenet121(weights=w).features, 1024)
    if name == "densenet121_cxr":
        f = _densenet_features(1)
        if pretrained:
            import torchxrayvision as xrv  # only needed to fetch the starting weights

            src = xrv.models.DenseNet(weights="densenet121-res224-all").features.state_dict()
            f.load_state_dict(src, strict=True)
        return CamNet(XrvPrep(), f, 1024)
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


def load_checkpoint(path, map_location="cpu"):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = build_model(ck["model"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


class ExportWrapper(nn.Module):
    """What goes into the ONNX file: probabilities + CAMs, softmax included."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        logits, cam = self.model.forward_with_cam(x)
        return torch.softmax(logits, 1), cam
