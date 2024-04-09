import numpy as np
import re
import yaml
import os
import os.path as path
import shutil
import socket
from datetime import datetime
from argparse import ArgumentParser
from modulefinder import ModuleFinder
import pandas as pd
import glob
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.datasets as dsets
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pickle
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import requests
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import pandas as pd
torch.manual_seed(1) # Set manual seed
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from tqdm import tqdm

from dataset import DATASET
from models import MODEL
from utils import Timer

parser = ArgumentParser()
parser.add_argument('--model-config', '-mc', required=True)
parser.add_argument('--data-config', '-dc', required=True)
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', '-o', default='')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--no-backup', action='store_true')
parser.add_argument('--checkpoint', type=str, help='name of the checkpoint file to resume training', default=None)
parser.add_argument('--skip-training', action='store_true', help='need to resume to skip-training')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_config(config_path):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if 'include' in new_config:
        include_config = get_config(new_config['include'])
        config.update(include_config)
        del new_config['include']
    config.update(new_config)
    return config

def main():
    
    if torch.cuda.is_available():
        print(f'Running on {socket.gethostname()} | {torch.cuda.device_count()}x {torch.cuda.get_device_name()}')
    args = parser.parse_args()

    # Load config
    config = get_config(args.model_config)
    data_config = get_config(args.data_config)
    config.update(data_config)
    
    # Override options
    """ 
    When here = config and you do here = here[key] in a loop, you're actually modifying the reference to here to point to a deeper structure in the config dictionary.
    Since dictionaries are mutable objects in Python, when you change here, you change config.
    Since here is a reference to config, any changes you make to here will also be reflected in config.
    """
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)
    
    # Prevent overwriting
    config['log_dir'] = args.log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists and not args.resume:
        print(f'WARNING: {args.log_dir} already exists. Skipping...')
        exit(0)

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f'Config saved to {config_save_path}')

    # Save code
    if not args.no_backup:
        code_dir = path.join(config['log_dir'], 'code_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        script_name = os.path.basename(__file__)
        shutil.copy2(__file__, os.path.join(config['log_dir'], script_name)) 
        """
        mf = ModuleFinder([os.getcwd()])
        mf.run_script(__file__)
        for name, module in mf.modules.items():
            if module.__file__ is None:
                continue
            rel_path = path.relpath(module.__file__)
            new_path = path.join(code_dir, rel_path)
            new_dirname = path.dirname(new_path)
            os.makedirs(new_dirname, mode=0o750, exist_ok=True)
            shutil.copy2(rel_path, new_path)  
        """
        print(f'Code saved to {code_dir}')
    
    train(args, config)


def train(args, config):

    mps_device = torch.device("cuda")

    writer = SummaryWriter(config['log_dir'], flush_secs=15)
    # Build model
    model = MODEL[config['model']](config)
    model = model.to(mps_device)
    
    # optim = torch.optim.Adam(model.parameters(), lr=0.001)
    optim = getattr(torch.optim, config['optim'])(model.parameters(), **config['optim_args'])
    lr_sched = getattr(lr_scheduler, config['lr_sched'])(optim, **config['lr_sched_args'])
    start_step = 0

    """ if args.resume:
        ckpt_path = path.join(config['log_dir'], 'ckpt.pt')
        print(ckpt_path)  
        if path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optim'])
            # lr_sched.load_state_dict(ckpt['lr_sched'])
            start_step = ckpt['step']
            print(f'Checkpoint loaded from {ckpt_path}')
        else:
            print("No checkpoint found to resume training.")
    optim.zero_grad() """

    if args.resume:
        ckpt_path = None
        if args.checkpoint:
            specified_ckpt_path = path.join(config['log_dir'], args.checkpoint)
            if path.exists(specified_ckpt_path):
                ckpt_path = specified_ckpt_path
                print(f"Loading specified checkpoint: {ckpt_path}")
            else:
                print(f"Specified checkpoint does not exist: {specified_ckpt_path}")
            if args.skip_training:
                step_match = re.search(r'best_ckpt-(\d+).pt', args.checkpoint)
                if step_match:
                    step = int(step_match.group(1))
        else:
            ckpt_files = glob.glob(path.join(config['log_dir'], 'ckpt-*.pt'))
            if ckpt_files:
                steps = [int(f.split('-')[-1].split('.')[0]) for f in ckpt_files]
                latest_step = max(steps)
                ckpt_path = path.join(config['log_dir'], f'ckpt-{latest_step:06}.pt')
                print(f"Found latest checkpoint: {ckpt_path}")
            else:
                print("No checkpoint found to resume training.")
        
        if ckpt_path:
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optim'])
            start_step = ckpt['step']
            config = ckpt['config']
            print(f'Checkpoint loaded from {ckpt_path}')
    optim.zero_grad()

    # Data
    Dataset = DATASET[config['dataset']]
    train_set = Dataset(config, root='/dataset', split='train')
    print(config)
    validation_set = Dataset(config, root='/dataset', split='validation')
    test_set = Dataset(config, root='/dataset', split='test')
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'])
    validation_loader = DataLoader(
        validation_set,
        batch_size=config['batch_size'])
    test_loader = DataLoader(
        test_set,
        batch_size=config['batch_size'])

    # Main training loop
    start_time = datetime.now()
    best_val_acc = 0.0  
    best_ckpt_path = ''  
    if not args.skip_training:
        print(f'Training started at {start_time}')
        for step in range(start_step + 1, config['max_train_steps'] + 1):
            train_losses = []
            train_acc = []
            validation_losses = []
            validation_acc = []
            
            for train_x,train_y in train_loader:
            # for train_x, train_y in tqdm(train_loader, desc="Loading training data"):
                train_x = train_x.to(mps_device) 
                train_y = train_y.to(mps_device)
                """ print(train_x.shape)
                print(train_y.shape) """
                """ print(type(train_x))
                print(type(train_y)) """
                optim.zero_grad()
                output = model_forward_backward(model, train_x, train_y, evaluate=False)
                """ for name, param in model.named_parameters():
                        if param.requires_grad and name == 'mamba_layers.1.mamba.D':
                            print(name, param.data) """
                
                optim.step()
                train_losses.append(output['loss'])
                train_acc.append(output['acc'])
            lr_sched.step()
                
            train_losses = torch.stack(train_losses)
            train_acc = torch.tensor(train_acc)

            train_loss_mean = train_losses.mean()
            acc_train = train_acc.mean()
            

            writer.add_scalar('loss/train', train_loss_mean.item(), step)
            writer.add_scalar('lr', lr_sched.get_last_lr()[0], step)
            writer.add_scalar('acc/train', acc_train.item(), step)
            
            for validation_x,validation_y in validation_loader:
                validation_x = validation_x.to(mps_device) 
                validation_y = validation_y.to(mps_device)
                """ print(validation_x.shape)
                print(validation_y.shape) """
            
                output = model_forward_backward(model, validation_x, validation_y, evaluate=True)

                validation_losses.append(output['loss'])
                validation_acc.append(output['acc'])
            

            validation_losses = torch.tensor(validation_losses)
            validation_acc = torch.tensor(validation_acc)

            validation_loss_mean = validation_losses.mean()
            acc_validation = validation_acc.mean()
            """ print(validation_loss_mean)
            print(acc_validation) """
            writer.add_scalar('loss/validation', validation_loss_mean.item(), step)
            writer.add_scalar('acc/validation', acc_validation.item(), step)

            
            now = datetime.now()
            elapsed_time = now - start_time
            elapsed_steps = step 
            total_steps = config['max_train_steps']
            est_total = elapsed_time * total_steps / elapsed_steps
            # Remove microseconds for brevity
            elapsed_time = str(elapsed_time).split('.')[0]
            est_total = str(est_total).split('.')[0]
            print(f'\r[Epoch {step}] [{elapsed_time} / {est_total}]', end='\n')
            print("--> Epoch Number : {}".format(step),
                " | Training Loss : {}".format(round(train_loss_mean.item(),4)),
                " | validation Loss : {}".format(round(validation_loss_mean.item(),4)),
                " | Training Accuracy : {}%".format(round(acc_train.item() * 100, 2)),
                " | validation Accuracy : {}%".format(round(acc_validation.item() * 100, 2)))
            
            if acc_validation > best_val_acc:
                best_val_acc = acc_validation
                if best_ckpt_path:
                    try:
                        os.remove(best_ckpt_path)
                    except OSError as e:
                        print(f"Error deleting previous best checkpoint: {e}")

                best_ckpt_path = path.join(config['log_dir'], f'best_ckpt-{step:06}.pt')
                torch.save({
                    'step': step,
                    'config': config,
                    'model': model.state_dict(),
                    'optim': optim.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, best_ckpt_path)
                print(f'New best checkpoint saved to {best_ckpt_path} with validation accuracy: {best_val_acc.item()}')  
            
            if step % config['ckpt_interval'] == 0:
                # Remove old checkpoints
                ckpt_paths = sorted(glob.glob(path.join(config['log_dir'], 'ckpt-*.pt')))
                for ckpt_path in ckpt_paths[:-1]:
                    os.remove(ckpt_path)

                new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step:06}.pt')
                torch.save({
                    'step': step,
                    'config': config,
                    'model': model.state_dict(),
                    'optim': optim.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, new_ckpt_path)
                print(f'\nCheckpoint saved to {new_ckpt_path}')
            
    else:
        print("Skipping training and proceeding to testing.")
    
    test_y_list = []
    softmax_list = []
# Making scv files
    columns = None
    sample_submission_path = os.path.join('/dataset', 'dog-breed-identification', 'sample_submission.csv')
    with open(sample_submission_path) as f:
        columns = f.readline().strip().split(',')

    for test_x,test_y in test_loader:
        test_x = test_x.to(mps_device)
        """ print(test_y)
        print(type(test_y))
        print(type(test_x)) """
        """ print(test_x.shape) """

        softmax_values = model_forward(model, test_x, test_y)
        
        # print(softmax_values)
        """ print(type(softmax_values)) """
        
        test_y_list.extend(test_y) 
        softmax_list.append(softmax_values.detach().cpu().numpy())

    softmax_array = np.concatenate(softmax_list, axis=0)
    
    df = pd.DataFrame(data=softmax_array, columns=columns[1:])
    
    df.insert(0, columns[0], test_y_list)
    
    submission_path = os.path.join(config['log_dir'], 'submission.csv')
    if os.path.exists(submission_path):
        os.remove(submission_path)
    df.to_csv(submission_path, index=False)

    writer.flush()
    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    with open(path.join(config['log_dir'], 'completed.yaml'), 'a') as f:
        yaml.dump({
            'step': step,
            'end_time': end_time,
        }, f)
    
def model_forward_backward(model, x, y, evaluate=False):
    # original
    criterion = nn.CrossEntropyLoss()
    
    if evaluate:
        with torch.no_grad():  
            z = model(x)
            loss = criterion(z, y)
    else:
        z = model(x)
        loss = criterion(z, y)
        loss.backward()  

    _, predicted = torch.max(z, 1)
    correct = (predicted == y).sum().item()
    acc = correct / z.size(0)
    acc_mean = acc
    loss_mean = loss.mean()

    if evaluate:
        detached_output = {
            'loss': loss_mean.item(),  
            'acc': acc_mean,
        }
    else:
        detached_output = {
            'loss': loss_mean.detach(),
            'acc': acc_mean,
        }

    return detached_output

""" def model_forward_backward(model, x, y, alpha=0.2, evaluate=False):
    # mixup
    criterion = nn.CrossEntropyLoss()
    
    if evaluate:
        with torch.no_grad():
            z = model(x)
            loss = criterion(z, y)
            _, predicted = torch.max(z, 1)
            correct = (predicted == y).sum().item()
    else:
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=alpha)
        z = model(mixed_x)
        loss = mixup_criterion(criterion, z, y_a, y_b, lam)
        loss.backward()  
        _, predicted = torch.max(z, 1)
        correct = (lam * (predicted == y_a).sum().float() + (1 - lam) * (predicted == y_b).sum().float()).item()

    acc = correct / x.size(0)
    acc_mean = acc
    loss_mean = loss.mean()

    if evaluate:
        detached_output = {
            'loss': loss_mean.item(),  
            'acc': acc_mean,          
        }
    else:
        detached_output = {
            'loss': loss_mean.detach(),  
            'acc': acc_mean,             
        }

    return detached_output """

""" def model_forward_backward(model, x, y, alpha=1.0, evaluate=False):
    # cutmix
    criterion = nn.CrossEntropyLoss()

    if evaluate:
        with torch.no_grad():
            z = model(x)
            loss = criterion(z, y)
            _, predicted = torch.max(z, 1)
            correct = (predicted == y).sum().item()
    else:
        x, target_a, target_b, lam = cutmix_data(x, y, alpha=alpha)
        z = model(x)
        loss = lam * criterion(z, target_a) + (1 - lam) * criterion(z, target_b)
        loss.backward()
        _, predicted = torch.max(z, 1)
        correct = lam * (predicted == target_a).sum().float() + (1 - lam) * (predicted == target_b).sum().float()

    acc = correct / x.size(0)
    acc_mean = acc
    loss_mean = loss.mean()

    if evaluate:
        detached_output = {
            'loss': loss_mean.item(),  
            'acc': acc_mean,          
        }
    else:
        detached_output = {
            'loss': loss_mean.detach(),  
            'acc': acc_mean,             
        }

    return detached_output """

def model_forward(model, x, y=None):
    with torch.no_grad():  
        z = model(x)
        softmax_values = torch.softmax(z, dim=1)
        return softmax_values

def mixup_data(x, y, alpha=0.2):
        """Apply mixup to a batch of input data and labels."""
        lam = torch.distributions.beta.Beta(alpha, alpha).sample()
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the mixup loss given the original loss and mixup parameters."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    target_a = y
    target_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, target_a, target_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def denormalize(tensor, means, stds):

    denormalized = tensor.clone()
    for t, mean, std in zip(denormalized, means, stds):
        t.mul_(std).add_(mean)  
    return denormalized


if __name__ == '__main__':
    main()