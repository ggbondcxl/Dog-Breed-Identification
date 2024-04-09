import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class ModifiedVGG16bnpretrain(nn.Module):
    def __init__(self, config=None):
        super(ModifiedVGG16bnpretrain, self).__init__()
        self.config = config
        vgg16 = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        
        self.features = vgg16.features

        for param in self.features.parameters():
            param.requires_grad = False
      
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.config['num_classes'])
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


