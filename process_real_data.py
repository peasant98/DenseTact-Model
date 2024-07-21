# 2024 ARMLab - DenseTact Calibration

import json
import os
import pickle

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader, Dataset
import timm
from PIL import Image
import numpy as np
import cv2

import pandas as pd

from tqdm import tqdm

import OpenEXR
import Imath

import h5py

from multiprocessing import Pool, Manager


DEPTH_CONSTANT = 17.437069

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

NAME_TO_RGB_MAPPINGS = {
    'CNORM': ('CNORMF-CNORMF1', 'CNORMF-CNORMF2', 'CNORMF-CNORMF3'),
    'S11': ('S-S11', 'S-S22', 'S-S33'),
    'S12': ('S-S12', 'S-S13', 'S-S23'),
    'UU1': ('U-U1', 'U-U2', 'U-U3'),
    'CNAREA': ('CNAREA', 'CSHEAR1', 'CSHEAR2'),
}

MAX = 0
MIN = 0

manager = Manager()
sample_id = manager.Value('i', 0)
sample_id_lock = manager.Lock()



def write_data(file_path, data, is_X = True, bounds_dict=None, y_imgs=None):
    global MAX
    if is_X:
        deformed_img_norm, undeformed_img_norm, image_diff = data
        
        # save all to png
        cv2.imwrite(f'{file_path}/deformed.png', deformed_img_norm * 255)
        cv2.imwrite(f'{file_path}/undeformed.png', undeformed_img_norm * 255)
        
        image_diff = (image_diff).astype(np.uint16)
        
        cv2.imwrite(f'{file_path}/diff.png', image_diff)
        
    else:
        cnorm_output, stress_data1_output, stress_data2_output, force_data_output, area_shear_data_output = data
        cnorm_img, stress1_img, stress2_img, force_img, area_shear_img = y_imgs
        
        # cat all data
        output = np.concatenate([cnorm_output, stress_data1_output, stress_data2_output, force_data_output, area_shear_data_output], axis=2)
        # save to npy file
        # np.save(f'{file_path}/data.npy', output)
        
        # convert all 
        cv2.imwrite(f'{file_path}/cnorm.png', cnorm_img)
        cv2.imwrite(f'{file_path}/stress1.png', stress1_img)
        cv2.imwrite(f'{file_path}/stress2.png', stress2_img)
        cv2.imwrite(f'{file_path}/force.png', force_img)
        cv2.imwrite(f'{file_path}/area_shear.png', area_shear_img)
        
        # save bounds to json
        with open(f'{file_path}/bounds.json', 'w') as f:
            json.dump(bounds_dict, f, indent=4)
        
class FullDataset(Dataset):
    def __init__(self, data_dir='real_world_output', transform=None, samples_dir='real_world_data', num_samples_to_cache=5000):
        self.root_dir = data_dir
        self.configuration_dirs = sorted(os.listdir(data_dir))
        self.data = []
        self.samples_dir = samples_dir
        self.transform = transform
        
        # real data list with root dir
        self.real_data_list = os.listdir(data_dir)
        # only get directories
        self.real_data_list = [f for f in self.real_data_list if os.path.isdir(os.path.join(data_dir, f))]
        print(self.real_data_list)
        
        # if not oes.path.exists(self.samples_dir):
        if True:
            self._construct_dataset_from_json()
            
            
    def create_cache(self):
        self.cache = []
        for i in range(self.num_samples_to_cache):
            res = self[i]
            self.cache.append(res)
            print("Cached", i)
            # add idx to cache
            
            
    def read_blender_info_json(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        return data
    
    
    def read_csv(self, csv_file):
        df = pd.read_csv(csv_file)
        return df
    
    def get_data_from_frame_and_name(self, root, frame, name, data):
        img_path = f'{root}/img_{name}_{frame}.png'
        # read image
        img = cv2.imread(img_path)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # remove duplicates from data, data is a dataframe
        data = data.drop_duplicates(subset=['field'])
        
        field1 = NAME_TO_RGB_MAPPINGS[name][0]
        field2 = NAME_TO_RGB_MAPPINGS[name][1]
        field3 = NAME_TO_RGB_MAPPINGS[name][2]
        
        data1 = data[data['field'] == field1]
        data2 = data[data['field'] == field2]
        data3 = data[data['field'] == field3]
        
        # get max and min values
        min_val_r = data1['min'].values[0]
        max_val_r = data1['max'].values[0]
        
        min_val_g = data2['min'].values[0]
        max_val_g = data2['max'].values[0]
        
        min_val_b = data3['min'].values[0]
        max_val_b = data3['max'].values[0]
        
        # get values from image, which are linearly scaled
        img_r = img[:,:,0]
        img_g = img[:,:,1]
        img_b = img[:,:,2]
        
        bounds_dict = {
            name: {
                'max_val_r': max_val_r,
                'min_val_r': min_val_r,
                'max_val_g': max_val_g,
                'min_val_g': min_val_g,
                'max_val_b': max_val_b,
                'min_val_b': min_val_b,
            }
        }
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # scale values
        channel1 = ((img_r / 255.0) * (max_val_r - min_val_r)) + min_val_r
        channel2 = ((img_g / 255.0) * (max_val_g - min_val_g)) + min_val_g
        channel3 = ((img_b / 255.0) * (max_val_b - min_val_b)) + min_val_b
        
        # combine channels into one image
        combined = np.stack([channel1, channel2, channel3], axis=2)
        return combined, img, bounds_dict
    
    def _construct_dataset_from_json(self):
        global MAX
        global MIN
        total_samples = 0
        for item in tqdm(self.real_data_list, desc="Processing Real World Data"):
            # go through each sequence of presses
            configuration_dir = self.root_dir
            folder_name = item
            combined = os.path.join(configuration_dir, folder_name)
            
            files = os.listdir(combined)
            
            # files with csv
            csv_files = [f for f in files if f.endswith('.csv')]
            
            undeformed_img_path = os.path.join(combined, f'img_1.png')
            
            
            undeformed_img = self.read_image(undeformed_img_path)
            
            
            undeformed_img = center_cropv2(undeformed_img, 0, 0, 150, 100)
            
                
            if len(csv_files) == 1:
  
                df = self.read_csv(os.path.join(combined, csv_files[0]))
                
                grouped = df.groupby('frame')
                
                pool = Pool(processes=8)
                
                tasks = [(group, frame, combined, undeformed_img, self.transform) for frame, group in grouped]

                for _ in tqdm(pool.imap_unordered(self.process_group, tasks), total=len(tasks), desc="Processing files in one press"):
                    pass

                pool.close()
                pool.join()
                    
    def process_group(self, args):
        global sample_id, sample_id_lock
        
        try:
        
            combined_dict = {}
            
            group, frame, combined, undeformed_img, transform  = args

            group['field'] = group['field'].str.replace(' ', '')

            cnorm_data = group[(group['field'] == 'CNORMF-CNORMF1') | 
                            (group['field'] == 'CNORMF-CNORMF2') | 
                            (group['field'] == 'CNORMF-CNORMF3')]

            cnorm_output, cnorm_img, bounds = self.get_data_from_frame_and_name(combined, frame, 'CNORM', cnorm_data)
            combined_dict.update(bounds)

            stress_data1 = group[(group['field'] == 'S-S11') | 
                                (group['field'] == 'S-S22') | 
                                (group['field'] == 'S-S33')]
            stress_data1_output, stress1_img, bounds = self.get_data_from_frame_and_name(combined, frame, 'S11', stress_data1)
            combined_dict.update(bounds)

            stress_data2 = group[(group['field'] == 'S-S12') |
                                (group['field'] == 'S-S13') | 
                                (group['field'] == 'S-S23')]
            stress_data2_output, stress2_img, bounds = self.get_data_from_frame_and_name(combined, frame, 'S12', stress_data2)
            combined_dict.update(bounds)
            

            force_data = group[(group['field'] == 'U-U1') | 
                            (group['field'] == 'U-U2') | 
                            (group['field'] == 'U-U3')]
            force_data_output, force_img, bounds = self.get_data_from_frame_and_name(combined, frame, 'UU1', force_data)
            combined_dict.update(bounds)
            

            area_shear_data = group[(group['field'] == 'CNAREA') | 
                                    (group['field'] == 'CSHEAR1') | 
                                    (group['field'] == 'CSHEAR2')]
            area_shear_data_output, area_shear_img, bounds = self.get_data_from_frame_and_name(combined, frame, 'CNAREA', area_shear_data)
            combined_dict.update(bounds)

            X = self.construct_rgb(undeformed_img, frame, combined)
            deformed_img_norm, undeformed_img_norm, image_diff = X
            
            X = (deformed_img_norm, undeformed_img_norm, image_diff)
            y = (cnorm_output, stress_data1_output, stress_data2_output, force_data_output, area_shear_data_output)
            y_imgs = (cnorm_img, stress1_img, stress2_img, force_img, area_shear_img)
            
            
            with sample_id_lock:
                sample_id.value += 1
                current_sample_id = sample_id.value
                
            os.makedirs(f'{self.samples_dir}/X{current_sample_id}', exist_ok=True)
            os.makedirs(f'{self.samples_dir}/y{current_sample_id}', exist_ok=True)

            write_data(f'{self.samples_dir}/X{current_sample_id}', X, is_X = True)
            write_data(f'{self.samples_dir}/y{current_sample_id}', y, is_X = False, bounds_dict=combined_dict, y_imgs=y_imgs)
        except Exception as e:
            print(e)
            print("Not writing any data!")
        
    def construct_rgb(self, undeformed_img, frame, combined):
        """
        Construct RGB inputs

        Args:
            undeformed_img (_type_): _description_
            undeformed_depth (_type_): _description_
            frame (_type_): _description_
            combined (_type_): _description_

        Returns:
            _type_: _description_
        """
        deform_img_path = os.path.join(combined, f'img_{frame+1}.png')
        deformed_img = self.read_image(deform_img_path)
        
        deformed_img = center_cropv2(deformed_img, 0, 0, 150, 100)
        
        # resize images to 512 by 512
        deformed_img = cv2.resize(deformed_img, (512, 512))
        undeformed_img = cv2.resize(undeformed_img, (512, 512))
        
        hsv_img1 = cv2.cvtColor(deformed_img, cv2.COLOR_RGB2HSV)
        hsv_img2 = cv2.cvtColor(undeformed_img, cv2.COLOR_RGB2HSV)

        deformed_img_norm = deformed_img / 255.0
        undeformed_img_norm = undeformed_img / 255.0
        image_diff = cv2.subtract(hsv_img1[:,:,2], hsv_img2[:,:,2])
        
        mask = np.ones_like(deformed_img)
        mask[deformed_img == 0] = 0

        # make mask 1 channel, make sure all channels are 0
        mask = mask[:,:,0] + mask[:,:,1] + mask[:,:,2]        
        
        # take vals that are 0
        mask[mask > 0] = 1 
        # apply mask to deformed depth
        X = (deformed_img_norm, undeformed_img_norm, image_diff)
        
        return X
        
    def __len__(self):
        """length of the dataset

        Returns:
            (int): length of the dataset (number of samples)
        """
        # length of files in the data folder
        
        length = sum(os.path.isdir(os.path.join(self.samples_dir, name)) for name in os.listdir(self.samples_dir)) / 2
        
        return int(length)
    
    def read_image(self, image_path):
        img = cv2.imread(image_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def __getitem__(self, idx):
        # if not self.creating_cache and idx < self.num_samples_to_cache:
        #     return self.cache[idx]
        
        sample_num = idx + 1
        # read deformed and undeformed images
        deformed_img_norm = cv2.imread(f'{self.samples_dir}/X{sample_num}/deformed.png')
        undeformed_img_norm = cv2.imread(f'{self.samples_dir}/X{sample_num}/undeformed.png')
        
        deformed_img_norm = cv2.cvtColor(deformed_img_norm, cv2.COLOR_BGR2RGB) / 255.0
        undeformed_img_norm = cv2.cvtColor(undeformed_img_norm, cv2.COLOR_BGR2RGB) / 255.0
        
        # read image diff
        image_diff = cv2.imread(f'{self.samples_dir}/X{sample_num}/diff.png', cv2.IMREAD_ANYDEPTH)
        # convert image diff to float
        image_diff = image_diff.astype(np.float32)
        # add extra dimension
        image_diff = image_diff[:,:,np.newaxis]
        
        # load bounds json
        # with open(f'{self.samples_dir}/y{sample_num}/bounds.json') as f:
        #     bounds = json.load(f)
            
        # # open each output image
        # # relative_depth = cv2.imread(f'{self.samples_dir}/y{sample_num}/depth.png', cv2.IMREAD_ANYDEPTH)
        # # relative_depth = relative_depth / 10000.0
        
        # cnorm_img = cv2.imread(f'{self.samples_dir}/y{sample_num}/cnorm.png')
        # cnorm_img = cv2.cvtColor(cnorm_img, cv2.COLOR_BGR2RGB)
        # cnorm_img[:,:,0] = ((cnorm_img[:,:,0] / 255.0) * (bounds['CNORM']['max_val_r'] - bounds['CNORM']['min_val_r'])) + bounds['CNORM']['min_val_r']
        # cnorm_img[:,:,1] = ((cnorm_img[:,:,1] / 255.0) * (bounds['CNORM']['max_val_g'] - bounds['CNORM']['min_val_g'])) + bounds['CNORM']['min_val_g']
        # cnorm_img[:,:,2] = ((cnorm_img[:,:,2] / 255.0) * (bounds['CNORM']['max_val_b'] - bounds['CNORM']['min_val_b'])) + bounds['CNORM']['min_val_b']
        
        # stress1_img = cv2.imread(f'{self.samples_dir}/y{sample_num}/stress1.png')
        # stress1_img = cv2.cvtColor(stress1_img, cv2.COLOR_BGR2RGB)
        # stress1_img[:, :, 0] = ((stress1_img[:,:,0] / 255.0) * (bounds['S11']['max_val_r'] - bounds['S11']['min_val_r'])) + bounds['S11']['min_val_r']
        # stress1_img[:, :, 1] = ((stress1_img[:,:,1] / 255.0) * (bounds['S11']['max_val_g'] - bounds['S11']['min_val_g'])) + bounds['S11']['min_val_g']
        # stress1_img[:, :, 2] = ((stress1_img[:,:,2] / 255.0) * (bounds['S11']['max_val_b'] - bounds['S11']['min_val_b'])) + bounds['S11']['min_val_b']
        
        # stress2_img = cv2.imread(f'{self.samples_dir}/y{sample_num}/stress2.png')
        # stress2_img = cv2.cvtColor(stress2_img, cv2.COLOR_BGR2RGB)
        # stress2_img[:, :, 0] = ((stress2_img[:,:,0] / 255.0) * (bounds['S12']['max_val_r'] - bounds['S12']['min_val_r'])) + bounds['S12']['min_val_r']
        # stress2_img[:, :, 1] = ((stress2_img[:,:,1] / 255.0) * (bounds['S12']['max_val_g'] - bounds['S12']['min_val_g'])) + bounds['S12']['min_val_g']
        # stress2_img[:, :, 2] = ((stress2_img[:,:,2] / 255.0) * (bounds['S12']['max_val_b'] - bounds['S12']['min_val_b'])) + bounds['S12']['min_val_b']
        
        # force_img = cv2.imread(f'{self.samples_dir}/y{sample_num}/force.png')
        # force_img = cv2.cvtColor(force_img, cv2.COLOR_BGR2RGB)
        # force_img[:, :, 0] = ((force_img[:,:,0] / 255.0) * (bounds['UU1']['max_val_r'] - bounds['UU1']['min_val_r'])) + bounds['UU1']['min_val_r']
        # force_img[:, :, 1] = ((force_img[:,:,1] / 255.0) * (bounds['UU1']['max_val_g'] - bounds['UU1']['min_val_g'])) + bounds['UU1']['min_val_g']
        # force_img[:, :, 2] = ((force_img[:,:,2] / 255.0) * (bounds['UU1']['max_val_b'] - bounds['UU1']['min_val_b'])) + bounds['UU1']['min_val_b']
        
        # area_shear_img = cv2.imread(f'{self.samples_dir}/y{sample_num}/area_shear.png')
        # area_shear_img = cv2.cvtColor(area_shear_img, cv2.COLOR_BGR2RGB)
        # area_shear_img[:, :, 0] = ((area_shear_img[:,:,0] / 255.0) * (bounds['CNAREA']['max_val_r'] - bounds['CNAREA']['min_val_r'])) + bounds['CNAREA']['min_val_r']
        # area_shear_img[:, :, 1] = ((area_shear_img[:,:,1] / 255.0) * (bounds['CNAREA']['max_val_g'] - bounds['CNAREA']['min_val_g'])) + bounds['CNAREA']['min_val_g']
        # area_shear_img[:, :, 2] = ((area_shear_img[:,:,2] / 255.0) * (bounds['CNAREA']['max_val_b'] - bounds['CNAREA']['min_val_b'])) + bounds['CNAREA']['min_val_b']
        
        # # use bounds for each channel rgb 
        # data = (cnorm_img, stress1_img, stress2_img, force_img, area_shear_img)
        # data = np.concatenate(data, axis=2)

        data = np.load(f'{self.samples_dir}/y{sample_num}/data.npy')
        
        # remove first channel
        data  = data[:,:,1:]
        X = (deformed_img_norm, undeformed_img_norm, image_diff)
        X = np.concatenate(X, axis=2)
        y = data
        
        y = self.transform(y).float()
        X = self.transform(X).float()
        
        return X, y
        
    
    def preprocess_depth(self, depth_image):
        if depth_image.ndim == 2:
            x = depth_image[:, :, np.newaxis]
            depth_image = np.copy(x)
        depth_image[depth_image[:,:,0]>50] = 15
        depth_image = depth_image[:,:,0]
        
        return depth_image
    
    def __repr__(self) -> str:
        return f'Depth Dataset with {self.__len__()} samples'


def center_cropv2(image, crop_top, crop_bottom, crop_left, crop_right):
    # Read the image
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        height, width, _ = image.shape
    
    # Calculate cropping coordinates
    top = crop_top
    bottom = height - crop_bottom
    left = crop_left
    right = width - crop_right
    
    # Perform the crop
    cropped_image = image[top:bottom, left:right]
    return cropped_image

def center_crop(image, crop_px):
    # Read the image
    if len(image.shape) == 2:
        height, width = image.shape
    else:
        height, width, _ = image.shape
    
    # Calculate cropping coordinates
    left = crop_px
    top = crop_px
    right = width - crop_px
    bottom = height - crop_px
    
    # Perform the crop
    cropped_image = image[top:bottom, left:right]
    return cropped_image

def read_exr_depth(filename, depth_channel='V'):
    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    # Get the header to determine the image dimensions
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read the depth channel
    if depth_channel in header['channels']:
        depth_data = exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))
    else:
        # raise ValueError(f"Depth channel '{depth_channel}' not found in the EXR file.")
        return None

    # Convert depth data to a numpy array
    depth_array = np.frombuffer(depth_data, dtype=np.float32).reshape((height, width))
    # depth_array = np.zeros((height, width))
    return depth_array

def read_exr_cv(filename):
    # Open the EXR file
    exr_image = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return exr_image
    
    
if __name__ == '__main__':
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
    ])
    
    dataset = FullDataset(transform=transform)
    
    dataset_length = len(dataset)
