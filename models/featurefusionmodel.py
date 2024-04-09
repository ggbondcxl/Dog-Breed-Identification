import torch
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils as utils

""" class FeatureFusionModel(nn.Module):
    def __init__(self, config=None):
        super(FeatureFusionModel, self).__init__()
        self.config = config

        # self.inceptionv3 = pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')
        self.xception = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')
        # self.nasnetalarge = pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet')
        self.nasnetalarge = pretrainedmodels.__dict__['nasnetalarge'](num_classes=1001, pretrained='imagenet+background')
        self.inceptionv4 = pretrainedmodels.__dict__['inceptionv4'](num_classes=1001, pretrained='imagenet+background')
        self.inceptionresnetv2 = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1001, pretrained='imagenet+background')
        
        self.nasnetalarge.eval()
        # self.inceptionv3.eval()
        self.inceptionv4.eval()
        self.xception.eval()
        self.inceptionresnetv2.eval()

        self.nasnetalarge.logits = nn.Identity()
        # self.inceptionv3.logits = nn.Identity()
        self.xception.logits = nn.Identity()
        self.inceptionv4.logits = nn.Identity()
        self.inceptionresnetv2.logits = nn.Identity()

        for model in [self.nasnetalarge, self.inceptionv4, self.xception, self.inceptionresnetv2]:
            for param in model.parameters():
                param.requires_grad = False

        nasnetalarge_features = self.nasnetalarge.last_linear.in_features
        # inceptionv3_features = self.inceptionv3.last_linear.in_features
        xception_features = self.xception.last_linear.in_features
        inceptionv4_features = self.inceptionv4.last_linear.in_features
        inceptionresnetv2_features = self.inceptionresnetv2.last_linear.in_features

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.fusion_layer = nn.Linear(nasnetalarge_features + inceptionv3_features + xception_features, 2048)
        total_features = nasnetalarge_features + inceptionv4_features + xception_features + inceptionresnetv2_features
        self.fusion_layer = nn.Linear(total_features, 4096)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(4096, self.config['num_classes'])
        )

    def forward(self, x):
        x1 = self.nasnetalarge.features(x)
        # x2 = self.inceptionv3.features(x)
        x2 = self.inceptionv4.features(x)
        x3 = self.xception.features(x)
        x4 = self.inceptionresnetv2.features(x)
        
        x1 = self.adaptive_pool(x1)
        x2 = self.adaptive_pool(x2)
        x3 = self.adaptive_pool(x3)
        x4 = self.adaptive_pool(x4)
        
        x1 = x1.view(x1.size(0), -1)  
        x2 = x2.view(x2.size(0), -1)  
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)

        # 融合特征并进行分类
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fusion_layer(x)
        x = self.classifier(x)

        return x """

import torch
import torch.nn as nn
import pretrainedmodels

class FeatureFusionModel(nn.Module):
    def __init__(self, config=None):
        super(FeatureFusionModel, self).__init__()
        self.config = config

        # Initialize the ResNeXt models
        self.resnext101_32x4d = pretrainedmodels.__dict__['resnext101_32x4d'](num_classes=1000, pretrained='imagenet')
        self.resnext101_64x4d = pretrainedmodels.__dict__['resnext101_64x4d'](num_classes=1000, pretrained='imagenet')

        # Set models to evaluation mode
        self.resnext101_64x4d.eval()
        self.resnext101_32x4d.eval()

        # Replace logits with nn.Identity()
        self.resnext101_64x4d.logits = nn.Identity()
        self.resnext101_32x4d.logits = nn.Identity()

        # Disable gradient updates
        for model in [self.resnext101_64x4d, self.resnext101_32x4d]:
            for param in model.parameters():
                param.requires_grad = False

        # Adjust feature sizes for the ResNeXt models
        resnext101_64x4d_features = self.resnext101_64x4d.last_linear.in_features
        resnext101_32x4d_features = self.resnext101_32x4d.last_linear.in_features

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Adjust the total features and the fusion layer
        total_features = resnext101_64x4d_features + resnext101_32x4d_features
        self.fusion_layer = nn.Linear(total_features, 2048)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, self.config['num_classes'])
        )

    def forward(self, x):
        x1 = self.resnext101_64x4d.features(x)
        x2 = self.resnext101_32x4d.features(x)
        
        x1 = self.adaptive_pool(x1)
        x2 = self.adaptive_pool(x2)

        x1 = x1.view(x1.size(0), -1)  
        x2 = x2.view(x2.size(0), -1)  

        # Fuse features and perform classification
        x = torch.cat((x1, x2), dim=1)
        x = self.fusion_layer(x)
        x = self.classifier(x)

        return x
