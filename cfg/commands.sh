restnet50pretrain
original
CUDA_VISIBLE_DEVICES=0 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-restnet50pretrain-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnet50pretrain.yaml -o "batch_size=128|max_train_steps=200"
mixup
CUDA_VISIBLE_DEVICES=2 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-restnet50pretrain-1 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnet50pretrain.yaml -o "batch_size=64|max_train_steps=20"
cutmix
CUDA_VISIBLE_DEVICES=3 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-restnet50pretrain-2 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnet50pretrain.yaml -o "batch_size=64|max_train_steps=20"

restnet50
CUDA_VISIBLE_DEVICES=1 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-restnet50-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnet50.yaml -o "batch_size=64|max_train_steps=200"

restnet101pretrain
CUDA_VISIBLE_DEVICES=5 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-restnet101pretrain-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnet101pretrain.yaml -o "batch_size=128|max_train_steps=200"
param.requires_grad = Ture
CUDA_VISIBLE_DEVICES=5 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-restnet101pretrain-1 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnet101pretrain.yaml -o "batch_size=128|max_train_steps=200"

resnext101_32x8dpretrain
CUDA_VISIBLE_DEVICES=2 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-resnext101_32x8dpretrain-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnext101_32x8dpretrain.yaml -o "batch_size=128|max_train_steps=200"
cutmix
CUDA_VISIBLE_DEVICES=2 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-resnext101_32x8dpretrain-1 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnext101_32x8dpretrain.yaml -o "batch_size=128|max_train_steps=200"
original+simple_transform
CUDA_VISIBLE_DEVICES=2 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-resnext101_32x8dpretrain-2 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnext101_32x8dpretrain.yaml -o "batch_size=128|max_train_steps=200"

resnext101_64x4dpretrain
CUDA_VISIBLE_DEVICES=2 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-resnext101_64x4dpretrain-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnext101_64x4dpretrain.yaml -o "batch_size=128|max_train_steps=200"

vgg16_bnpretrain
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-vgg16_bnpretrain-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/vgg16_bnpretrain.yaml -o "batch_size=128|max_train_steps=200"
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-vgg16_bnpretrain-1 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/vgg16_bnpretrain.yaml -o "batch_size=128|max_train_steps=200"

featurefusionmodel
nasnetalarge+inceptionv3+xception
CUDA_VISIBLE_DEVICES=5 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-featurefusionmodel-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification331.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/featurefusionmodel.yaml -o "batch_size=128|max_train_steps=200"
nasnetalarge+inceptionv4+xception+inceptionresnetv2
CUDA_VISIBLE_DEVICES=5 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-featurefusionmodel-1 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification331.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/featurefusionmodel.yaml -o "batch_size=128|max_train_steps=200"

pnasnet5largepretrain
CUDA_VISIBLE_DEVICES=4 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-pnasnet5largepretrain-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification331.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/pnasnet5largepretrain.yaml -o "batch_size=128|max_train_steps=200"

mamba
CUDA_VISIBLE_DEVICES=3 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-mamba-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/mamba.yaml -o "batch_size=64|max_train_steps=200"

Production of the submission file
CUDA_VISIBLE_DEVICES=2 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-resnext101_32x8dpretrain-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/resnext101_32x8dpretrain.yaml -o "batch_size=128|max_train_steps=200" --resume --checkpoint best_ckpt-000053.pt --skip-training
CUDA_VISIBLE_DEVICES=2 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-featurefusionmodel-0 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification331.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/featurefusionmodel.yaml -o "batch_size=128|max_train_steps=200" --resume --checkpoint best_ckpt-000157.pt --skip-training
CUDA_VISIBLE_DEVICES=2 python /risk1/chengxilong/dog_breed_identificatiopn/train.py -l /risk1/chengxilong/dog_breed_identificatiopn/runs/dog-featurefusionmodel-1 -dc /risk1/chengxilong/dog_breed_identificatiopn/cfg/data/dogbreedidentification331.yaml -mc /risk1/chengxilong/dog_breed_identificatiopn/cfg/model/featurefusionmodel.yaml -o "batch_size=128|max_train_steps=200" --resume --checkpoint best_ckpt-000058.pt --skip-training

tensorboard
tensorboard --logdir=/risk1/chengxilong/dog_breed_identificatiopn/runs --port=6006


