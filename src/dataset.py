#dataset.py

import os
import cv2
import torch
import torchvision.transforms as tf
import pandas as pd
import numpy as np
import PIL as im

from torch.utils.data import Dataset



class Nyudepth_png(Dataset):
    def __init__(self,
                 dataset_path:str,
                 dataframe:pd.DataFrame,
                 transform_shape:tf,
                 transform_color:tf):
        
        self.dataset_path = dataset_path
        self.dataframe = dataframe
        self.transform_shape = transform_shape
        self.transform_color = transform_color

    def __len__(self):
        return self.dataframe.size
    
    def __getitem__(self, index):
        
        #--- retrieve the files paths and convert images to np.arrays
        rgb_file_path = os.path.join(self.dataset_path, self.df['rgb'].loc[index])
        dep_file_path = os.path.join(self.dataset_path, self.df['dep'].loc[index])

        rgb_data = np.array(im.open(rgb_file_path).convert('RGB'), dtype=np.float32) 
        dep_data = np.array(im.open(dep_file_path), dtype=np.float32)

        #--- resize the data for DPU
        rgb_data = cv2.resize(rgb_data, (224, 224))
        dep_data = cv2.resize(dep_data, (224, 224))

        #--- normalize the data
        rgb_data = np.transpose(rgb_data, (2, 0, 1)) #swap from (1, 1, 3) to (3, 1, 1)
        rgb_data = (rgb_data - np.min(rgb_data))/((np.max(rgb_data) - np.min(rgb_data))) #normalize data values to the interval [0,1]
        rgb_data = (rgb_data - 0.5) * 2.0 #shift data to the interval [-1:1]
        
            #for this part we know the data is encoded on 16bits,
            #and we assume captured values for distances varying 
            #between 0.7-10m measured in mm
            #based on the Nyudepth dataset informations
        dep_data = dep_data/1000.0 #convert in meters
        dep_data = (dep_data - 0.7)/(10.0 - 0.7) #normalize data values to the interval [0:1]
        dep_data = np.clip(dep_data, 0.0, 10.0) #clip data outside defined range

        #--- equalize dep_data array dimensions with rgb_data array dimensions
        dep_data = np.expand_dims(dep_data, axis=0)

        #--- combine dep_data and rgb_data in one array
        data = np.append(rgb_data, dep_data, axis=0)
        data = torch

        #--- apply transforms
        if self.transform_shape is not None:
            data = self.transform_shape(data)
        
        if self.transform_2 is not None:
            data[0:3] = self.transform_color(data[0:3])

        return data[0:3], data[3].unsqueeze(dim=0)   # (chanels, Height, Width) and (depth, Height, Width)



    
        