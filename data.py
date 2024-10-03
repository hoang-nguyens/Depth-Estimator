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
                'image should be PIL image. Got {}'.format(type(image))
            )

        if random.random() < self.p:
            image = np.asarray(image)
            image = Image.fromarray(image[..., self.indices])

        return { 'image': image, 'depth' : depth}

class ToTensor:
    def __init__(self, is_test = True):
        self.is_test = is_test

    def to_tensor(self, image):
        if (not _is_pil_image(image)) or (not is_numpy_image(image)):
            raise TypeError([
                'image should be PIL image or numpy image. Got {}'.format(type(image))
            ])

        if isinstance(image, np.ndarray):
            img = torch.from_numpy(image.transpose(2, 0, 1))
            return img.float().div(255)

        if image.mode == 'I':
            img = torch.from_numpy(np.array(image, np.int32, copy=False))
        elif image.mode == 'I;16':
            img = torch.from_numpy(np.array(image, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(image.tobytes()))

        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if image.mode == 'YCbCr':
            nchannel = 3
        elif image.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(image.mode)

        img = img.view(image.size[1], image.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 1000

        depth = torch.clamp(depth, 10, 1000)

        return { 'image': image, 'depth' : depth}

def noTransform( is_test = False):
    return transforms.Compose([
        ToTensor(is_test = is_test),
    ])

def getTransform():
    return transforms.Compose([
        RandomHorizontalFlip(0.5),
        RandomChannelsSwap(0.5),
        ToTensor(),
    ])

class DepthDataset(Dataset):
    def __init__(self, image_path, depth_path, transform = None):
        self.transform = transform
        self.image_path = image_path
        self.depth_path = depth_path

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

    test_image_path = np.array(glob.glob(r'data/image_depth_test/image/*.jpg'))
    test_depth_path = np.array(glob.glob(r'data/image_depth_test/depth/*.jpg'))

    return { 'train_image' : train_image_path,
             'train_depth': train_depth_path,
             'test_image': test_image_path,
             'test_depth' : test_depth_path}







