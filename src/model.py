import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

class EmotionResNet(nn.Module):
    """
    State-of-the-art Transfer Learning Architecture for FER2013.
    Uses Pretrained ResNet18 as a feature extractor.
    """
    def __init__(self, num_classes=7, pretrained=False):
        super(EmotionResNet, self).__init__()
        # Load Pretrained ResNet18
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
            
        self.model = resnet18(weights=weights)
        
        # Replace the final fully connected layer and inject Dropout
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.6), # Intense Regularization for FER2013 Overfitting
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class VGG(nn.Module):
    # Kept for backward-compatibility if you ever want to load the old weights
    def __init__(self, vgg_name='VGG19'):
        super(VGG, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, 7)
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out
    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M': layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
