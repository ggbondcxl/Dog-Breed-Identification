""" import torch
import torch.nn as nn
from transformers import SwinForImageClassification, SwinConfig

class ModifiedSwinTransformerpretrain(nn.Module):
    def __init__(self, config=None):
        super(ModifiedSwinTransformerpretrain, self).__init__()
        self.config = config

        swin_config = SwinConfig.from_json_file("/risk1/chengxilong/dog_breed_identificatiopn/swintransformerin22/config.json")
        self.swin_transformer = SwinForImageClassification(swin_config)
        self.swin_transformer.load_state_dict(torch.load("/risk1/chengxilong/dog_breed_identificatiopn/swintransformerin22/pytorch_model.bin"))
        # print(self.swin_transformer)
        
        for param in self.swin_transformer.parameters():
            param.requires_grad = False
        
        self.swin_transformer.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 2048),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, self.config['num_classes'])
        )
    
    def forward(self, x):
        x = self.swin_transformer(x)  
        x = x.logits
        x = self.classifier(x) 
        return x """
    
from torchvision import models
import torch.nn as nn

class ModifiedSwinTransformerpretrain(nn.Module):
    def __init__(self, config=None):
        super(ModifiedSwinTransformerpretrain, self).__init__()
        self.config = config

        model = models.vit_l_16(pretrained=True)
        num_classifier_feature = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(num_classifier_feature, 120)
        )
        self.model = model

        for param in self.model.named_parameters():
            if "heads" not in param[0]:
                param[1].requires_grad = False

    def forward(self, x):
        return self.model(x)


