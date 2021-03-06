#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 20:08:59 2020

@author: ajv012
"""
# imports

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class DRDataset(Dataset):
    '''
    Data loader for the DR dataset
    '''
    
    def __init__(self, csv, img_dir, transform = None):
        '''
        

        Parameters
        ----------
        csv_file : string
            directory to the csv file with image ids and diagnosis.
        img_dir : string
            directory with all the images.

        Returns
        -------
        None.

        '''
        self.image_ids = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        '''

        Returns
        -------
        len of the dataset

        '''
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        '''
        

        Parameters
        ----------
        idx : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # acquire image
        curr_image_name = self.image_ids.iloc[idx, 0] + '.png'
        img_path = os.path.join(self.img_dir, curr_image_name)
        
        # read image
        image = io.imread(img_path)
        diagnosis = self.image_ids.iloc[idx, 1]
        
        # preapre sample return
        sample = {'image': image, 'd': diagnosis}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'd': sample['d']}

class Rescale(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        new_h, new_w = self.output_size
        
        img = transform.resize(image, (new_h, new_w))
        

        return {'image': img, 'd': sample['d']}
    
class Normalize(object):
    '''
    Performs a global centering of the pixel values, which are already normalized?
    '''
    
    
    def __call__(self, sample):
        image = sample['image']
        image =  image - image.mean()
        image = image / image.std()
        return {'image': image, 'd': sample['d']}
        

def display(sample_batched):
    '''

    Parameters
    ----------
    image :numpy array. Name of the image to be displayed
    Currently just shows image, put more intricacies can be applied later when displaying an image

    Returns
    -------
    None.

    '''
    images_batch = sample_batched['image']
    
    grid = utils.make_grid(images_batch)
    toshow = grid.numpy()
    toshow = toshow - toshow.min()
    toshow = toshow/toshow.max()
    plt.imshow(toshow.transpose((1, 2, 0)))
    
    plt.title('Batch from dataloader')
    

    
def load_data(csv, img_dir, batch):
# csv = '/nfs/unixspace/linux/accounts/student/a/ajv012/DR/Train/train.csv'
# root = '/nfs/unixspace/linux/accounts/student/a/ajv012/DR/Train'
# batch = 32

    composed =  transforms.Compose([Rescale(256), ToTensor(), Normalize()]) # transforms.Compose([transforms.Resize(256,256), transforms.ToTensor(), transforms.Normalize((0.13), (0.38))])
    
    DR_dataset = DRDataset(csv, img_dir, composed)

    dataloader = DataLoader(DR_dataset, batch, shuffle = True, num_workers = 4)  
        
    return dataloader
    
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(i_batch, sample_batched['image'].size(), sample_batched['d'])
    
    #     plt.figure()
    #     display(sample_batched)
    #     plt.axis('off')
    #     plt.ioff()
    #     plt.show()
        
# load_data()
            
    

        
        

        
        


