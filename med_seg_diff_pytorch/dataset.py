import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import random
import torchvision.transforms.functional as F


import json
import h5py
import torchvision.transforms as T


class NYUv2Dataset(Dataset):
    # A workaround for a pytorch bug: https://github.com/pytorch/vision/issues/2194
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)
    
    def __init__(self, args, mode, flip_p=0.5):
        super().__init__()

        self.args = args
        self.mode = mode
        self.flip_p = flip_p

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        # For NYUDepthV2, crop size is fixed
        height, width = (240, 320)
        # crop_size = (228, 304)
        crop_size = (self.args.image_size, self.args.image_size)
        

        self.height = height
        self.width = width
        self.crop_size = crop_size

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        """Get the images"""
        path_file = os.path.join(self.args.data_path, self.sample_list[idx]['filename'])

        f = h5py.File(path_file, 'r')
        rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
        dep_h5 = f['depth'][:]

        rgb = Image.fromarray(rgb_h5, mode='RGB')
        dep = Image.fromarray(dep_h5.astype('float32'), mode='F')

        t_rgb = T.Compose([
            T.Resize(self.height),
            T.CenterCrop(self.crop_size),
            T.ToTensor()
            # T.ToTensor(),
            # T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        t_dep = T.Compose([
            T.Resize(self.height),
            T.CenterCrop(self.crop_size),
            self.ToNumpy(),
            T.ToTensor()
        ])

        rgb = t_rgb(rgb)
        dep = t_dep(dep)

        if random.random() < self.flip_p:
            rgb = F.vflip(rgb)
            dep = F.vflip(dep)

        # dep_sp = self.get_sparse_depth(dep, self.args.num_sample)
        # output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep, 'K': K}

        return (rgb, dep)

    def get_sparse_depth(self, dep, num_sample): pass


class ISICDataset(Dataset):
    def __init__(self, data_path, csv_file, img_folder, transform=None, training=True, flip_p=0.5):
        df = pd.read_csv(os.path.join(data_path, csv_file), encoding='gbk')
        self.img_folder = img_folder
        self.name_list = df.iloc[:, 0].tolist()
        self.label_list = df.iloc[:, 1].tolist()
        self.data_path = data_path
        self.transform = transform
        self.training = training
        self.flip_p = flip_p

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index] + '.jpg'
        img_path = os.path.join(self.data_path, self.img_folder, name)

        mask_name = name.split('.')[0] + '_Segmentation.png'
        mask_path = os.path.join(self.data_path, self.img_folder, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.training:
            label = 0 if self.label_list[index] == 'benign' else 1
        else:
            label = int(self.label_list[index])

        if self.transform:
            # save random state so that if more elaborate transforms are used
            # the same transform will be applied to both the mask and the img
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)
            if random.random() < self.flip_p:
                img = F.vflip(img)
                mask = F.vflip(mask)

        if self.training:
            return (img, mask)
        return (img, mask, label)


class GenericNpyDataset(torch.utils.data.Dataset):
    def __init__(self, directory: str, transform, test_flag: bool = True):
        '''
        Genereic dataset for loading npy files.
        The npy store 3D arrays with the first two dimensions being the image and the third dimension being the channels.
        channel 0 is the image and the other channel is the label.
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.test_flag = test_flag
        self.filenames = [x for x in os.listdir(self.directory) if x.endswith('.npy')]

    def __getitem__(self, x: int):
        fname = self.filenames[x]
        npy_img = np.load(os.path.join(self.directory, fname))
        img = npy_img[:, :, :1]
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = npy_img[:, :, 1:]
        mask = np.where(mask > 0, 1, 0)
        image = img[:, ...]
        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        if self.transform:
            # save random state so that if more elaborate transforms are used
            # the same transform will be applied to both the mask and the img
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)
        if self.test_flag:
            return image, mask, fname
        return image, mask

    def __len__(self) -> int:
        return len(self.filenames)
