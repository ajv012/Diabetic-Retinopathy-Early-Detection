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
num_epochs = 1
num_classes = 5
batch_size = 8
learning_rate = 0.0001

# define the model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__() # calling the constructor of the parent class
        
        # define convolution layers
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(nn.Conv2d(35, 128, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(163, 326, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(nn.Conv2d(489, 1024, kernel_size=5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        
        # add the GAP layer
        self.GAP = nn.Sequential(nn.Dropout(), nn.AvgPool2d(32, ceil_mode=True))
        # self.drop_out = nn.Dropout()
        
        # define the fully connected layers 
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 5)
           
        print('done in constructor')
    
    def forward(self, x):
        
        # convolutional layers with residues
        # first layer
        out1 = self.layer1(x)
        # cat with input x
        x_reduced = f.max_pool2d(x, kernel_size = 2, stride = 2)
        joint = torch.cat((out1, x_reduced), 1)
        
        # second layer
        out2 = self.layer2(joint)
        # cat with input joint
        joint_reduced = f.max_pool2d(joint, kernel_size = 2, stride = 2)
        joint = torch.cat((out2, joint_reduced),1)
        
        # third layer
        out3 = self.layer3(joint)
        # cat with input joint
        joint_reduced = f.max_pool2d(joint, kernel_size = 2, stride = 2)
        joint = torch.cat((out3, joint_reduced),1)
        
        # fourth layer
        out4 = self.layer4(joint)
        
        # GAP layer
        out = self.GAP(out4)
        out = out.reshape(out.size(0), -1)
        
        # Fully connected buisness
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = torch.sigmoid(out)
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
    model = ConvNet().cuda()
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
    
    
