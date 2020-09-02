#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:43:40 2020

@author: ajv012
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 08:32:17 2020

@author: ajv012
"""

# imports
import torch
from torch import nn as nn
from Data_loader import *
import numpy as np
from sklearn import metrics
from scipy.special import softmax
import torch.nn.functional as f

# define the hyper parameters
num_epochs = 30
nr_classes = 5
batch_size = 8
learning_rate = 0.0001

# define the model
class Dense_Block(nn.Module):
    def __init__(self, in_channels):
        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(num_features = in_channels)
            
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
  
    def forward(self, x):
        bn = self.bn(x) 
        conv1 = self.relu(self.conv1(bn))
        conv2 = self.relu(self.conv2(conv1))
        # Concatenate in channel dimension
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))
        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
       
        conv4 = self.relu(self.conv4(c3_dense)) 
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
       
        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
       
        return c5_dense
    
class Transition_Layer(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__() 
        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(num_features = out_channels) 
        
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
        
    def forward(self, x): 
        bn = self.bn(self.relu(self.conv(x))) 
        out = self.avg_pool(bn)
        return out 

class DenseNet(nn.Module): 
    def __init__(self, nr_classes): 
        super(DenseNet, self).__init__()
        self.lowconv = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, padding = 3, bias = False)
        self.relu = nn.ReLU()
        
        # Make Dense Blocks 
        self.denseblock1 = self._make_dense_block(Dense_Block, 64) 
        self.denseblock2 = self._make_dense_block(Dense_Block, 128)
        self.denseblock3 = self._make_dense_block(Dense_Block, 128)
        
        # Make transition Layers 
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128) 
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128) 
        self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 64)
        
        # Classifier 
        self.bn = nn.BatchNorm2d(num_features = 64) 
        self.pre_classifier = nn.Linear(64*4*4, 512) 
        self.classifier = nn.Linear(512, nr_classes)
        
    def _make_dense_block(self, block, in_channels): 
        layers = [] 
        layers.append(block(in_channels)) 
        return nn.Sequential(*layers) 
    
    def _make_transition_layer(self, layer, in_channels, out_channels): 
        modules = [] 
        modules.append(layer(in_channels, out_channels)) 
        return nn.Sequential(*modules) 
    
    def forward(self, x): 
        out = self.relu(self.lowconv(x)) 
        out = self.denseblock1(out) 
        out = self.transitionLayer1(out) 
        out = self.denseblock2(out) 
        out = self.transitionLayer2(out) 
        
        out = self.denseblock3(out) 
        out = self.transitionLayer3(out) 
     
        out = self.bn(out) 
        out = out.view(-1, 64*4*4) 
        
        out = self.pre_classifier(out) 
        out = self.classifier(out)
        return out
        
# open each csv and test csv for data 
root_dir = '/nfs/unixspace/linux/accounts/student/a/ajv012/DR'
img_dir = '/nfs/unixspace/linux/accounts/student/a/ajv012/DR/all_images'
l = {}
a = {}
v = {}
v_loss = {}
co_max_split = {}
cl_split = {}

# get trainable parameters
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(f'Number of trainable parameters {trainable_params}')

for split in range(1,2):
    train_temp_dir = '/split_data_full_multiclass/Fold' + str(split) + '.csv'
    train_dir = root_dir + train_temp_dir
    train_loader = load_data(train_dir, img_dir, batch_size)
    
    test_temp_dir = '/split_data_full_multiclass/Fold' + str(split) + '_test.csv'
    test_dir = root_dir + test_temp_dir
    test_loader = load_data(test_dir, img_dir, batch_size)
    
    
    # training variables
    total_step = len(train_loader)
    loss_total = []
    loss_vali = []
    acc = []   
    co = np.zeros((0,5))
    cl = np.zeros((0))
    vali_epoch = []
    
    # define model for the split 
    # define model
    model = DenseNet(nr_classes).cuda()
    model = model.float()
    print(f'Defined model for split {split}')
    
    # loss criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f'Going to training for split {split}')
    
    for epoch in range(num_epochs):
        model.train()
        loss_list = []
        vali_loss_list = []
        correct_train = 0
        total_train = 0
        for i, sample_batched in enumerate(train_loader):
            
            # get images and labels for the current pass
            images = sample_batched['image']
            labels = sample_batched['d']
            
            # convert to cuda
            images = images.float().cuda()
            labels = labels.cuda()
                
            # run the forward pass on the model
            output = model(images)
            
            # calculate the loss after the forward pass
            loss = criterion(output, labels)
            
            # append the loss
            loss_list.append(loss.item())
            
            # backpropogate and run the optimization
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            
            # track the accuracy
            total_train += labels.size(0)
            _, predicted = torch.max(output.data, 1) 
            predicted = predicted.cuda()
            correct_train += (predicted==labels).sum().item()
            
        # calculate average loss for every epoch
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, i + 1, total_step, np.mean(loss_list),
                  (correct_train / total_train) * 100))
        
        loss_total.append(np.mean(loss_list))
        acc.append(correct_train/total_train)
    
        # enter validation mode for model
        model.eval() 
      
        # validation loop
        with torch.no_grad():
            correct = 0
            total = 0
            for i, sample_batched in enumerate(test_loader):
                images = sample_batched['image']
                labels = sample_batched['d']
                
                # convert to cuda
                images = images.float().cuda()
                labels = labels.cuda()
                
                # push images in model 
                outputs = model(images)
                
                # calculate the loss after the forward pass
                loss = criterion(outputs, labels)
                vali_loss_list.append(loss.item())
                
                
                temp, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cuda()
                total = total + labels.size(0)
                correct = correct + (predicted == labels).sum().item()
                
                # if on last epoch, then accumlate outputs and labels
                if epoch == num_epochs - 1:
                    co = np.concatenate([co, outputs.cpu().detach().numpy()], axis=0)
                    cl = np.concatenate([cl, labels.cpu().detach().numpy()], axis=0)
                    # axis = 0 for columns, axis=1 for rows
                
            print('Test Accuracy of the model test images: {} %'.format((correct / total) * 100))
            
            vali_epoch.append(correct/total)
            loss_vali.append(np.mean(vali_loss_list))
            
    # get softmax on outputs
    co_max = softmax(co, axis = 1)
    co_predicted = np.argmax(co_max, axis=1)
    
    # book keeping for the split 
    l[split] = loss_total
    a[split] = acc
    v[split] = vali_epoch
    v_loss[split] = loss_vali
    co_max_split[split] = co_predicted
    cl_split[split] = cl
    
    print(f'done with split {split}')
    
    
