#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 22:30:36 2020

@author: ajv012
"""


import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

# get the transforms 
# ToTensor transform converts an image into a tensor, which is understood by PyTorch (can have multiple channels and not just the color ones)
# Normalize transform normalizes the images with the mean and standard deviation that go in ## HOW TO PICK MEAN AND STD
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])



# downloading dataset for training and validating
# shuffling the dataset and applying the transforms that we previously defined
### HOW TO ADAPT THIS TO OUR PROGRAM?
trainset = datasets.MNIST('/nfs/unixspace/linux/accounts/student/a/ajv012/DR', download=True, train=True, transform=transform)
valset = datasets.MNIST('/nfs/unixspace/linux/accounts/student/a/ajv012/DR', download=True, train=False, transform=transform)

# don't need to adapt this, we alrwady have a datset object

# create dataloader objects for the training and validating datasets so that you can create batches, shuffle
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

# Creating the network using Torches' nn module

input_size = 784 # this is the number of pixels in the input image
hidden_sizes = [128, 64] # this is the size for the two layers that are hidden
output_size = 10 # this is the size of the output layer

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]), \
                      nn.ReLU(), \
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]), \
                      nn.ReLU(), \
                      nn.Linear(hidden_sizes[1], output_size), \
                      nn.LogSoftmax(dim=1))

# model structure 
        # Sequential creates a network of the layers
        # Linear creates a one-to-many map from the entries in the first argument to entries in the second argument
            ### HOW TO DECIDE HOW MANY LAYERS
            ### HOW TO DECIDE WHAT THE SIZE OF THE LAYER IS GOING TO BE 
        # ReLU activation: allows positive values to pass thru and negative values are modified to zero 
            # ReLU is type of an activation function. ### HOW TO CHOOSE ACTIVATION FUNCTIONS FOR LAYERS?
        # Softmax is another activation function

# NLL algorithm: mostly used for classification 
# This is sort of the error function that you want to minimize
# you take the -ve log so that large numbers can be better representated
# The smaller the NLL, the better it is
criterion = nn.NLLLoss() # you are maximizing your algorithm by minimizing the error
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

# Now you have a model and a function to calculate your "error". But now you need to use the error to change the model
# so that the error decreases. You can do this by changing the weights given to each branch 

'''
our neural network iterates over the training set and updates the weights. 
We make use of torch.optim which is a module provided by PyTorch to optimize the model, 
perform gradient descent and update the weights by back-propagation. 
Thus in each epoch (number of times we iterate over the training set), we will be 
seeing a gradual decrease in training loss.
'''
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)

# VALIDATION 
correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count)) 

torch.save(model, './my_mnist_model.pt') 

print('Done')
