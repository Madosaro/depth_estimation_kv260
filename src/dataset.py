#dataset.py

import os
import cv2
import torch
import torchvision.transforms as tf
import pandas as pd
import numpy as np

from PIL import Image

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
        return len(self.dataframe)
    
    def __getitem__(self, index:int):
        
        #--- retrieve the files paths and image files   
        rgb_file_path = os.path.join(self.dataset_path, self.dataframe['rgb'].iloc[index])
        dep_file_path = os.path.join(self.dataset_path, self.dataframe['depth'].iloc[index])

        rgb_img = Image.open(rgb_file_path).convert('RGB')
        dep_img = Image.open(dep_file_path)

        #--- resize the data for DPU
        rgb_data = cv2.resize(np.array(rgb_img, dtype=np.float32), (224, 224))
        dep_data = cv2.resize(np.array(dep_img, dtype=np.float32), (224, 224))

        #--- normalize the data
        rgb_data = np.transpose(rgb_data, (2, 0, 1)) #swap from (1, 1, 3) to (3, 1, 1)
        rgb_data = (rgb_data - np.min(rgb_data))/(np.max(rgb_data) - np.min(rgb_data) + 1e-8) #normalize data values to the interval [0,1]
        rgb_data = (rgb_data - 0.5) * 2.0 #shift data to the interval [-1:1]
        
            #for this part we know the data is encoded on 16bits,
            #and we assume captured values for distances varying 
            #between 0.7-10m measured in mm
            #based on the Nyudepth dataset informations
        dep_data = dep_data/1000.0 #convert in meters
        dep_data = np.clip(dep_data, 0.7, 10.0) #clip data outside defined range
        dep_data = (dep_data - 0.7)/(10.0 - 0.7) #normalize data values to the interval [0:1]
        

        #--- equalize dep_data array dimensions with rgb_data array dimensions
        dep_data = np.expand_dims(dep_data, axis=0)

        #--- transform np.arrays to tensors
        rgb_tensor = torch.from_numpy(rgb_data)
        dep_tensor = torch.from_numpy(dep_data)
       
    
        #--- apply transforms
        if self.transform_color is not None:
            rgb_tensor = self.transform_color(rgb_tensor)

        if self.transform_shape is not None:
            data_tensor = torch.cat([rgb_tensor, dep_tensor], dim=0)
            data_tensor = self.transform_shape(data_tensor)
            rgb_tensor = data_tensor[0:3]
            dep_tensor = data_tensor[3:4]

        
        return rgb_tensor, dep_tensor   # (chanels, Height, Width) and (depth, Height, Width)



    
        