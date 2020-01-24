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
import numpy as np
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

class DRDataset(Dataset):
    '''
    Data loader for the DR dataset
    '''
    
    def __init__(self, csv_file, root_dir):
        '''
        

        Parameters
        ----------
        csv_file : string
            directory to the csv file with image ids and diagnosis.
        root_dir : string
            directory with all the images.

        Returns
        -------
        None.

        '''
        self.image_ids = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        '''

        Returns
        -------
        len of the dataset

        '''
        return len(self.image_ids)
    
    def __getitem__(self, idx, transform = None):
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
        img_path = os.path.join(self.root_dir, curr_image_name)
        
        # read image
        image = io.imread(img_path)
        diagnosis = self.image_ids.iloc[idx, 1]
        
        # preapre sample return
        sample = {'image': image, 'diagnosis': diagnosis}
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, diagnosis = sample['image'], sample['diagnosis']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'diagnosis': diagnosis}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, diagnosis = sample['image'], sample['diagnosis']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'diagnosis': diagnosis}

def display(image):
    '''

    Parameters
    ----------
    image :numpy array. Name of the image to be displayed
    Currently just shows image, put more intricacies can be applied later when displaying an image

    Returns
    -------
    None.

    '''
    plt.imshow(image[...,1], cmap = 'gray')
    
def main():
    
    DR_dataset = DRDataset(csv_file='/nfs/unixspace/linux/accounts/student/a/ajv012/DR/Images/train.csv', root_dir='/nfs/unixspace/linux/accounts/student/a/ajv012/DR/Images')
    
    # apply rescale transform
    #scale = Rescale(256)
    
    fig = plt.figure()
    for i in range(len(DR_dataset)):
        sample = DR_dataset[i]
        print(i, sample['image'].shape, sample['diagnosis'])
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        display(sample['image'])
        
        if i == 3:
            plt.show()
            break

    
    print('end of main')

print('Going to run main')
main()

        
        

        
        


