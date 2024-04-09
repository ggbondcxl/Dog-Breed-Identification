from .model import Model
from .resnet50pretrain import ModifiedResNet50pretrain
from .resnet50 import ModifiedResNet50
from .resnet101pretrain import ModifiedResNet101pretrain
from .resnet101 import ModifiedResNet101
from .resnext101_32x8dpretrain import ModifiedResNext101_32x8dpretrain
from .vgg16_bnpretrain import ModifiedVGG16bnpretrain
from .resnext101_64x4dpretrain import ModifiedResNext101_64x4dpretrain
from .featurefusionmodel import FeatureFusionModel
from .pnasnet5largepretrain import Pnasnet5Largepretrain
from .swintransformerpretrain import ModifiedSwinTransformerpretrain

MODEL = {
    'resnet50pretrain': ModifiedResNet50pretrain,
    'resnet50': ModifiedResNet50,
    'resnet101pretrain': ModifiedResNet101pretrain,
    'resnet101': ModifiedResNet101,
    'resnext101_32x8dpretrain': ModifiedResNext101_32x8dpretrain,
    'vgg16_bnpretrain': ModifiedVGG16bnpretrain,
    'resnext101_64x4dpretrain': ModifiedResNext101_64x4dpretrain,
    'featurefusionmodel': FeatureFusionModel,
    'pnasnet5largepretrain': Pnasnet5Largepretrain,
    'swintransformerpretrain': ModifiedSwinTransformerpretrain,
}
