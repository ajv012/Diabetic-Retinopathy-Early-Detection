#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:31:36 2020

@author: ajv012
"""
# imports
import numpy as np
from sklearn.model_selection import StratifiedKFold
import csv


# load file 
file = open('train.csv')

# read all the data 
file.readline()
lines = file.readlines()
image_ids = []
diagnosis = []

for line in lines:
    line = line.split(',')
    image_ids.append(line[0])
    diagnosis.append(int(line[1]))

image_ids = np.asarray(image_ids)
diagnosis = np.asarray(diagnosis)

# apply the stratifiedkfold splits to get five sets of arrays 
n_splits = 5
skf = StratifiedKFold(n_splits)
kf = []

for k, (train_index, test_index) in enumerate(skf.split(image_ids, diagnosis)):
    #print(f'Fold {k}: training on {len(train_index)}; test on {len(test_index)}.')
    kf.append((train_index, test_index))

# write to csv files for each split
count = 1

for data in kf:
    train_file_name = 'Fold' + str(count) + '.csv'
    test_file_name = 'Fold' + str(count) + '_test.csv'
    
    train_file = open(train_file_name)
    test_file = open(test_file_name)
    
    train_indices = data[0]
    test_indices = data[1]
    
    with open(train_file_name, mode = 'w') as training_file:
        writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for index in train_indices:
            writer.writerow([image_ids[index], diagnosis[index]])
    
    with open(test_file_name, mode = 'w') as testing_file:
        writer = csv.writer(testing_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for index in test_indices:
            writer.writerow([image_ids[index], diagnosis[index]])
    
    count = count + 1
    train_file.close()
    test_file.close()

file.close()
print('done')

