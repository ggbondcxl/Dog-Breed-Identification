import base64
import os
import os.path as path
import random
import struct
import pickle
from collections import defaultdict
from glob import glob
from io import BytesIO
from multiprocessing import Pool
import pandas as pd

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from einops import unpack, rearrange, repeat, pack
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100, Omniglot, CIFAR10
from torchvision.transforms.functional import pil_to_tensor
from tqdm.auto import tqdm
from torch.distributions.beta import Beta
from utils import Timer
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class DogBreedIdentification(IterableDataset):

    name = 'DogBreedIdentification'

    def __init__(self, config, root='/dataset', split='train', output_dir='/risk1/chengxilong/dog_breed_identificatiopn/dataset'):
        super().__init__()
        self.dataset_path = os.path.join(root, 'dog-breed-identification')
        self.config = config
        self.root = root
        self.split = split
        self.output_dir = output_dir
        self.train_pickle_path = os.path.join(self.output_dir, 'DogBreedIdentification_train.pickle')
        self.validation_pickle_path = os.path.join(self.output_dir, 'DogBreedIdentification_validation.pickle')
        self.test_pickle_path = os.path.join(self.output_dir, 'DogBreedIdentification_test.pickle')
        self.labels_path = os.path.join(root, 'dog-breed-identification', 'labels.csv')
        self.sample_submission_path = os.path.join(root, 'dog-breed-identification', 'sample_submission.csv')
        if not path.exists(self.train_pickle_path):
            self.download()
            self.build_pickle()

        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data()  # Assuming data is preprocessed and pickled

    def download(self):
        if not path.exists(self.dataset_path):
            raise RuntimeError(f'Please download {self.name} dataset manually, following the instructions in README.md')
        else:
            print('dataset exists!')
    
    def __iter__(self):
        self.data, self.labels = self.load_data()
        self.current_index = 0
        
        indices = list(range(len(self.data)))
        if self.split == 'train':
            
            random.shuffle(indices)

        self.data = [self.data[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        return self

    def __next__(self):
        if self.current_index >= len(self.data):
            raise StopIteration
        
        data = torch.tensor(self.data[self.current_index])
        label = self.labels[self.current_index]
        self.current_index += 1

        transform = self.get_transform()
        data = transform(data)

        return data, label
    
    def load_data(self):
        if self.split == 'train':
            pickle_path = self.train_pickle_path
        elif self.split == 'test':
            pickle_path = self.test_pickle_path
        elif self.split == 'validation':
            pickle_path = self.validation_pickle_path
        else:
            raise ValueError(f"Unsupported split: {self.split}. Expected 'train', 'test', or 'validation'.")

        with open(pickle_path, 'rb') as f:
            data, labels = pickle.load(f)
            #print(len(data))
            """ if self.split == 'validation':
                print(len(data))
                print(len(data)) """
        return data, labels

    
    def get_transform(self):
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
        
    """ def get_transform(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if self.split == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.config['img_size'], self.config['img_size'])), 
                transforms.CenterCrop(self.config['img_size']),
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.config['img_size'], self.config['img_size'])),
                transforms.CenterCrop(self.config['img_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)]) """
    
    def build_pickle(self):
        self.labels_df = pd.read_csv(self.labels_path)
        sample_submission_df = pd.read_csv(self.sample_submission_path)
        breed_columns = sample_submission_df.columns[1:]

        self.breed_to_code = {breed: code for code, breed in enumerate(breed_columns)}
        validation_size = 500 

        for split in ['train', 'test', 'validation']:
            if split == 'validation':
                continue 

            pickle_path = self.train_pickle_path if split == 'train' else self.test_pickle_path
            if split == 'train':
                split_path = os.path.join(self.dataset_path, 'train')
            elif split == 'validation':
                split_path = os.path.join(self.dataset_path, 'train')  # Validation data is typically a subset of the training data
            elif split == 'test':
                split_path = os.path.join(self.dataset_path, 'test')
            else:
                raise ValueError(f"Invalid split name: {split}")

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(f"Created output directory: {self.output_dir}")
            
            x_validation = []
            y_validation = []
            x_dict = []
            y_dict = []
            total_samples = 0

            image_files = os.listdir(split_path)
            if split == 'train':
                validation_images = random.sample(image_files, validation_size)
                validation_pickle_path = os.path.join(self.output_dir, 'DogBreedIdentification_validation.pickle')

            print(f'Converting {split} data to Python dictionary...')
            for image_file in tqdm(image_files, desc=f'Processing images in {split}'):
                if not image_file.endswith('.jpg'):
                    continue

                img_path = os.path.join(split_path, image_file)
                img_id = os.path.splitext(image_file)[0]
                
                img_binary = open(img_path, 'rb').read()
                img_array = np.frombuffer(img_binary, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img_resized = cv2.resize(img, (self.config['img_size'], self.config['img_size']))
                img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_tensor = torch.tensor(np.transpose(img_resized_rgb, (2, 0, 1)))

                if split == 'train' and image_file in validation_images:
                    x_validation.append(img_tensor.numpy())
                elif split == 'train' and image_file not in validation_images:
                    x_dict.append(img_tensor.numpy()) 
                elif split == 'test':
                    x_dict.append(img_tensor.numpy())
                else:
                    raise ValueError(f"Invalid split name: {split}")

                if split in ['train', 'validation']:
                    breed = self.labels_df[self.labels_df['id'] == img_id]['breed'].values[0]
                    code = self.breed_to_code[breed]
                    y_dict.append(code) if split == 'train' and image_file not in validation_images else y_validation.append(code)
                else:
                    y_dict.append(img_id)

                total_samples += 1

            print(f"Total samples in {split}: {total_samples}")
            print(f"Total number of classes in {split}: {len(self.breed_to_code)}")
            tmp_pickle_path = pickle_path + '.tmp'
            with open(tmp_pickle_path, 'wb') as f:
                pickle.dump((x_dict, y_dict), f)
            os.rename(tmp_pickle_path, pickle_path)

            if split == 'train':
                print(f'Saving validation data with {validation_size} samples...')
                with open(validation_pickle_path, 'wb') as f:
                    pickle.dump((x_validation, y_validation), f)
    
class DogBreedIdentification331(IterableDataset):

    name = 'DogBreedIdentification331'

    def __init__(self, config, root='/dataset', split='train', output_dir='/risk1/chengxilong/dog_breed_identificatiopn/dataset'):
        super().__init__()
        self.dataset_path = os.path.join(root, 'dog-breed-identification')
        self.config = config
        self.root = root
        self.split = split
        self.output_dir = output_dir
        self.train_pickle_path = os.path.join(self.output_dir, 'DogBreedIdentificatio331_train.pickle')
        self.validation_pickle_path = os.path.join(self.output_dir, 'DogBreedIdentification331_validation.pickle')
        self.test_pickle_path = os.path.join(self.output_dir, 'DogBreedIdentification331_test.pickle')
        self.labels_path = os.path.join(root, 'dog-breed-identification', 'labels.csv')
        self.sample_submission_path = os.path.join(root, 'dog-breed-identification', 'sample_submission.csv')
        if not path.exists(self.train_pickle_path):
            self.download()
            self.build_pickle()

        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data()  # Assuming data is preprocessed and pickled

    def download(self):
        if not path.exists(self.dataset_path):
            raise RuntimeError(f'Please download {self.name} dataset manually, following the instructions in README.md')
        else:
            print('dataset exists!')
    
    def __iter__(self):
        self.data, self.labels = self.load_data()
        self.current_index = 0
        
        indices = list(range(len(self.data)))
        if self.split == 'train':
            
            random.shuffle(indices)

        self.data = [self.data[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        return self

    def __next__(self):
        if self.current_index >= len(self.data):
            raise StopIteration
        
        data = torch.tensor(self.data[self.current_index])
        label = self.labels[self.current_index]
        self.current_index += 1

        transform = self.get_transform()
        data = transform(data)

        return data, label
    
    def load_data(self):
        if self.split == 'train':
            pickle_path = self.train_pickle_path
        elif self.split == 'test':
            pickle_path = self.test_pickle_path
        elif self.split == 'validation':
            pickle_path = self.validation_pickle_path
        else:
            raise ValueError(f"Unsupported split: {self.split}. Expected 'train', 'test', or 'validation'.")

        with open(pickle_path, 'rb') as f:
            data, labels = pickle.load(f)
            #print(len(data))
        return data, labels

    
    def get_transform(self):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        if self.split == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)), 
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.6),
                transforms.RandomRotation(degrees=(30)),
                transforms.ToTensor(), 
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        
    def build_pickle(self):
        self.labels_df = pd.read_csv(self.labels_path)
        sample_submission_df = pd.read_csv(self.sample_submission_path)
        breed_columns = sample_submission_df.columns[1:]

        self.breed_to_code = {breed: code for code, breed in enumerate(breed_columns)}
        validation_size = 500 

        for split in ['train', 'test', 'validation']:
            if split == 'validation':
                continue 

            pickle_path = self.train_pickle_path if split == 'train' else self.test_pickle_path
            if split == 'train':
                split_path = os.path.join(self.dataset_path, 'train')
            elif split == 'validation':
                split_path = os.path.join(self.dataset_path, 'train')  # Validation data is typically a subset of the training data
            elif split == 'test':
                split_path = os.path.join(self.dataset_path, 'test')
            else:
                raise ValueError(f"Invalid split name: {split}")

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(f"Created output directory: {self.output_dir}")
            
            x_validation = []
            y_validation = []
            x_dict = []
            y_dict = []
            total_samples = 0

            image_files = os.listdir(split_path)
            if split == 'train':
                validation_images = random.sample(image_files, validation_size)
                validation_pickle_path = os.path.join(self.output_dir, 'DogBreedIdentification331_validation.pickle')

            print(f'Converting {split} data to Python dictionary...')
            for image_file in tqdm(image_files, desc=f'Processing images in {split}'):
                if not image_file.endswith('.jpg'):
                    continue

                img_path = os.path.join(split_path, image_file)
                img_id = os.path.splitext(image_file)[0]
                
                img_binary = open(img_path, 'rb').read()
                img_array = np.frombuffer(img_binary, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img_resized_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = torch.tensor(np.transpose(img_resized_rgb, (2, 0, 1)))

                if split == 'train' and image_file in validation_images:
                    x_validation.append(img_tensor.numpy())
                elif split == 'train' and image_file not in validation_images:
                    x_dict.append(img_tensor.numpy()) 
                elif split == 'test':
                    x_dict.append(img_tensor.numpy())
                else:
                    raise ValueError(f"Invalid split name: {split}")

                if split in ['train', 'validation']:
                    breed = self.labels_df[self.labels_df['id'] == img_id]['breed'].values[0]
                    code = self.breed_to_code[breed]
                    y_dict.append(code) if split == 'train' and image_file not in validation_images else y_validation.append(code)
                else:
                    y_dict.append(img_id)

                total_samples += 1

            print(f"Total samples in {split}: {total_samples}")
            print(f"Total number of classes in {split}: {len(self.breed_to_code)}")
            tmp_pickle_path = pickle_path + '.tmp'
            with open(tmp_pickle_path, 'wb') as f:
                pickle.dump((x_dict, y_dict), f)
            os.rename(tmp_pickle_path, pickle_path)

            if split == 'train':
                print(f'Saving validation data with {validation_size} samples...')
                with open(validation_pickle_path, 'wb') as f:
                    pickle.dump((x_validation, y_validation), f)


DATASET = {
    'dogbreedidentification': DogBreedIdentification,
    'dogbreedidentification331': DogBreedIdentification331,
}