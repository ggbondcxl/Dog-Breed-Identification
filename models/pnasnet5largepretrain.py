import torch
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils as utils

class Pnasnet5Largepretrain(nn.Module):
    def __init__(self, config=None):
        super(Pnasnet5Largepretrain, self).__init__()
        self.config = config
        self.pnasnet5large = pretrainedmodels.__dict__['pnasnet5large'](num_classes=1001, pretrained='imagenet+background')
        
        # 获取pnasnet5large最后一个线性层的in_features
        pnasnet5large_features = self.pnasnet5large.last_linear.in_features

        # 替换为Identity，用于特征提取
        self.pnasnet5large.last_linear = nn.Identity()
        
        # 冻结预训练模型的参数
        for param in self.pnasnet5large.parameters():
            param.requires_grad = False

        # 自定义的全连接层用于分类
        self.fusion_layer = nn.Linear(pnasnet5large_features, 2048)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(2048, self.config['num_classes'])
        )

    def features(self, x):
        # 获取预训练模型的特征
        x = self.pnasnet5large.features(x)
        return x

    def forward(self, x):
        # 通过预训练模型提取特征
        x = self.features(x)
        # 自适应平均池化
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # 扁平化处理
        x = x.view(x.size(0), -1)
        # 通过全连接层
        x = self.fusion_layer(x)
        # 通过分类器
        x = self.classifier(x)
        return x