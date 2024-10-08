# import library
# function to check is pil image or is numpy
# random horizontal flip ( image and depth) # note that depth has 3 channels, we have to change to grayscale
# random swap channels of images
# custom toTensor ( from PIL Image)
# dataset class ( load jpg, png image to PIL, and from PIL to Tensor
# function to create dataloader
# function to get path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from PIL import Image
from io import BytesIO
import PIL
import numpy as np
import random
import glob


def _is_pil_image(image):
    return isinstance(image, Image.Image)

def is_numpy_image(image):
    return isinstance(image, np.ndarray) and (image.ndim in {2, 3})

class RandomHorizontalFlip:
    def __init__(self, p):
        self.p = p
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'image should be PIL image. Got {}'.format(type(image))
            )
        if not _is_pil_image(depth):
            raise TypeError(
                'image should by PIL image. Got {}'.format(type(image))
            )

        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}

class RandomChannelsSwap:
    def __init__(self, p):
        self.p = p
        from itertools import permutations
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'image should be PIL image. Got {}'.format(type(image))
            )
        if not _is_pil_image(depth):
            raise TypeError(
                'depth should be PIL image. Got {}'.format(type(depth))
            )

        if random.random() < self.p:
            image_np = np.array(image)  # Convert to numpy array
            # Ensure the shape is correct
            if image_np.ndim == 3 and image_np.shape[2] == 3:  # Check if image has 3 channels
                # Randomly swap channels
                permuted_indices = random.choice(self.indices)  # Pick one random permutation
                image_np = image_np[..., permuted_indices]  # Swap channels
                image = Image.fromarray(image_np)  # Convert back to PIL image
            else:
                raise ValueError("Input image does not have 3 channels")

        return {'image': image, 'depth': depth}


import torch
import numpy as np
from PIL import Image

class ToTensor:
    def __init__(self, is_test=True):
        self.is_test = is_test

    def to_tensor(self, image):
        if isinstance(image, np.ndarray):
            # Convert numpy array to tensor and change from HWC to CHW format
            img = torch.from_numpy(image.transpose(2, 0, 1))
            return img.float() / 255  # Normalize to [0, 1]

        if not _is_pil_image(image):
            raise TypeError('Image should be a PIL Image. Got {}'.format(type(image)))

        # Convert PIL image to numpy array and then to tensor
        img = np.array(image)

        if img.ndim == 2:  # Grayscale image
            img = img[:, :, np.newaxis]  # Add channel dimension

        img = torch.from_numpy(img.transpose(2, 0, 1))  # Change from HWC to CHW format

        return img.float() / 255  # Normalize to [0, 1]

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        # Resize images
        image = image.resize((320, 240))  # Resize to desired dimensions
        depth = depth.resize((320, 240))  # Resize to desired dimensions

        # Convert to tensor
        image = self.to_tensor(image)
        depth = self.to_tensor(depth)

        # Adjust depth based on whether it's for testing or training
        if self.is_test:
            depth = depth.float() / 10
        else:
            depth = depth.float() * 10

        return {'image': image, 'depth': depth}

def noTransform( is_test = False):
    return transforms.Compose([
        ToTensor(is_test = is_test),
        #transforms.Resize((224,224))
    ])

def getTransform():
    return transforms.Compose([
        RandomHorizontalFlip(0.5),
        RandomChannelsSwap(0.5),
        ToTensor(),
        #transforms.Resize((224, 224))
    ])

class DepthDataset(Dataset):
    def __init__(self, image_path, depth_path, transform = None):
        self.transform = transform
        self.image_path = image_path
        self.depth_path = depth_path
    def __len__(self):
        return len(self.image_path)
    def __getitem__(self, idx):
        images = Image.open(self.image_path[idx])
        depths = Image.open(self.depth_path[idx])
        sample = { 'image' : images, 'depth' : depths}
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

def getData(batch_size, is_train, image_path, depth_path):
    if is_train:
        dataset_train = DepthDataset(image_path, depth_path, transform=getTransform())
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        return dataloader_train
    else:
        dataset_test = DepthDataset(image_path, depth_path, transform= noTransform())
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        return dataloader_test

def getPath(): # nyu2 dataset
    train_image_path = np.array(glob.glob(r'data/image_depth_train/image/*.jpg'))
    train_depth_path = np.array(glob.glob(r'data/image_depth_train/depth/*.png'))

    test_image_path = np.array(glob.glob(r'data/image_depth_test/image/*.png'))
    test_depth_path = np.array(glob.glob(r'data/image_depth_test/depth/*.png'))

    return { 'train_image' : train_image_path,
             'train_depth': train_depth_path,
             'test_image': test_image_path,
             'test_depth' : test_depth_path}






