from torchvision import models
from torch import nn
import torch.nn.utils.prune as prune

class PretrainedAlexNet2D(nn.Module):
    def __init__(self, num_classes=8, pretrained=True):  # Change num_classes to match your dataset
        super(PretrainedAlexNet2D, self).__init__()

        # Load the pretrained AlexNet model
        self.base_model = models.alexnet(pretrained=pretrained)

        # Modify the classifier's final layer for custom output classes
        if num_classes != 1000:  # ImageNet has 1000 classes
            self.base_model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.base_model(x)


class PretrainedAlexNet2DPrunedL1(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, pruning_amount=0.2):
        super(PretrainedAlexNet2DPrunedL1, self).__init__()

        # Load the pretrained AlexNet model
        self.base_model = models.alexnet(pretrained=pretrained)

        # Modify the classifier's final layer for custom output classes
        if num_classes != 1000:  # ImageNet has 1000 classes
            self.base_model.classifier[6] = nn.Linear(4096, num_classes)

        # Apply L1 pruning to convolutional layers
        for layer in self.base_model.features:
            if isinstance(layer, nn.Conv2d):
                prune.l1_unstructured(layer, name="weight", amount=pruning_amount)

    def forward(self, x):
        return self.base_model(x)


class PretrainedAlexNet2DPrunedL2(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, pruning_amount=0.2):
        super(PretrainedAlexNet2DPrunedL2, self).__init__()

        # Load the pretrained AlexNet model
        self.base_model = models.alexnet(pretrained=pretrained)

        # Modify the classifier's final layer for custom output classes
        if num_classes != 1000:  # ImageNet has 1000 classes
            self.base_model.classifier[6] = nn.Linear(4096, num_classes)

        # Apply structured pruning (L2-norm) to convolutional layers
        for layer in self.base_model.features:
            if isinstance(layer, nn.Conv2d):
                prune.ln_structured(layer, name="weight", amount=pruning_amount, n=2, dim=0)

    def forward(self, x):
        return self.base_model(x)


class PretrainedMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(PretrainedMobileNetV2, self).__init__()

        # Load the pretrained MobileNetV2 model
        self.base_model = models.mobilenet_v2(pretrained=pretrained)

        # Modify the classifier's final layer for custom output classes
        if num_classes != 1000:  # ImageNet has 1000 classes
            self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)


class PretrainedMobileNetV2WithL1Pruning(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, pruning_amount=0.2):
        super(PretrainedMobileNetV2WithL1Pruning, self).__init__()

        # Load the pretrained MobileNetV2 model
        self.base_model = models.mobilenet_v2(pretrained=pretrained)

        # Modify the classifier's final layer for custom output classes
        if num_classes != 1000:  # ImageNet has 1000 classes
            self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

        # Apply L1 pruning to the layers
        for layer in self.base_model.features:
            if isinstance(layer, nn.Conv2d):
                prune.l1_unstructured(layer, name="weight", amount=pruning_amount)

    def forward(self, x):
        return self.base_model(x)


class PretrainedMobileNetV2WithL2Pruning(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True, pruning_amount=0.2):
        super(PretrainedMobileNetV2WithL2Pruning, self).__init__()

        # Load the pretrained MobileNetV2 model
        self.base_model = models.mobilenet_v2(pretrained=pretrained)

        # Modify the classifier's final layer for custom output classes
        if num_classes != 1000:  # ImageNet has 1000 classes
            self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

        # Apply L2 structured pruning to the layers
        for layer in self.base_model.features:
            if isinstance(layer, nn.Conv2d):
                prune.ln_structured(layer, name="weight", amount=pruning_amount, n=2, dim=0)

    def forward(self, x):
        return self.base_model(x)