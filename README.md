KaggleCompetitions
Dog Breed Identification

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if self.split == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.config['img_size'], self.config['img_size'])), 
                transforms.CenterCrop(self.config['img_size']),
                transforms.RandomRotation(20), 
                transforms.RandomHorizontalFlip(0.1), 
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1), 
                transforms.ToTensor(), 
                transforms.Normalize(mean, std),
                transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False)
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.config['img_size'], self.config['img_size'])),
                transforms.CenterCrop(self.config['img_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

resnext101_32x8dpretrain
Acc: 0.933
vgg16_bnpretrain
Acc: 0.724
resnet50pretrain
Acc: 0.869
resnet50
Acc: 0.391
resnet101pretrain
Acc: 0.771
featurefusionmodel-0:nasnetalarge+inceptionv3+xception
Acc: 0.935
featurefusionmodel-1:nasnetalarge+inceptionv4+xception+inceptionresnetv2
Acc: 0.945
pnasnet5largepretrain
Acc: 0.925
resnext101_64x4dpretrain
Acc: 0.932