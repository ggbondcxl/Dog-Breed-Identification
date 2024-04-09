import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights

class ModifiedResNext101_64x4dpretrain(nn.Module):
    def __init__(self, config=None):
        super(ModifiedResNext101_64x4dpretrain, self).__init__()
        self.config = config
        
        # resnext = models.resnext101_32x8d(pretrained=True) 
        """ model_name = 'resnext101_64x4d'
        resnext = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet') """
       
        resnext = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.DEFAULT)

        for param in resnext.parameters():
            param.requires_grad = False
        
        self.features = nn.Sequential(*list(resnext.children())[:-2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.config['num_classes'])
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  
        x = self.fc(x)
        return x
