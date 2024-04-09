import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
from torchvision import models

class ModifiedResNet101pretrain(nn.Module):
    def __init__(self, config=None):
        super(ModifiedResNet101pretrain, self).__init__()
        self.config = config
        resnet_weights = ResNet101_Weights.DEFAULT
        original_resnet = models.resnet101(weights=resnet_weights)  
        
        for param in original_resnet.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(
            *(list(original_resnet.children())[:-2])
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(original_resnet.fc.in_features, self.config['num_classes'])

    def forward(self, x):
        x = self.features(x)  
        x = self.avgpool(x)   
        x = torch.flatten(x, 1)  
        x = self.fc(x)  
        return x


