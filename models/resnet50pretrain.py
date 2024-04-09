import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import models

class ModifiedResNet50pretrain(nn.Module):
    def __init__(self, config=None):
        super(ModifiedResNet50pretrain, self).__init__()
        self.config = config
        resnet_weights = ResNet50_Weights.DEFAULT
        original_resnet = models.resnet50(weights=resnet_weights)
        
        for param in original_resnet.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(
            *(list(original_resnet.children())[:-2])
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.additional_layers = nn.Sequential(
            nn.Linear(original_resnet.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.config['num_classes'])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  
        x = torch.flatten(x, 1)
        x = self.additional_layers(x)  
        return x
