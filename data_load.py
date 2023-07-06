#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os, sys, glob
from astropy.io import fits
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split, KFold
from datetime import datetime

class data_preprocess():
    
    def __init__(self, redshifts=[1,2], classification='binary', folding=False, num_pixel=96, nproj = 29, channels=['sz'], normalise = True, smoothing=False, save=True, readme=None):
        
        self.redshifts = redshifts
        self.classification = classification
        self.folding = folding
        self.npxl = num_pixel
        self.nproj = nproj
        self.channels = channels # list containing either 'sz', 'xray' or both
        self.normalise = normalise
        self.smoothing=smoothing
        self.save = save
        self.readme = readme
    
    def get_snapshot_list(self):
        
        # Loading merging population
        merging_cluster_list = pd.read_csv('../generating_cluster_list/lists/output_merger_list_75_10000_haloID.csv')
        merging_cluster_list = merging_cluster_list.rename({'merger_state': 'label'}, axis=1)
        
        # Filter for certain redshift bins
        merging_cluster_list = merging_cluster_list[merging_cluster_list['z_bin'].isin(self.redshifts)]
        
        # If working on a binary classification problem, just select the merging clusters ignoring pre / post
        if self.classification == 'binary':
            merging_sample = merging_cluster_list[merging_cluster_list['label'] == 'merging']
            
        # Loading control population
        control_sample = pd.read_csv('../generating_cluster_list/lists/output_control_sample.csv', index_col=False)
        
        # Filter for certain redshift bins
        control_sample = control_sample[control_sample['z_bin'].isin(self.redshifts)]

        # Full sample list
        full_sample = pd.concat([control_sample, merging_sample[control_sample.columns]]).reset_index(drop=True)
        
        return full_sample
        
    def get_fits_file_loc_sz(self, reg, snap, label):

        clnum='0000'+str(reg)
        clnum=clnum[-4:]
        cname = "NewMDCLUSTER_"+clnum+"/"
        s_str = str(snap).rjust(3, '0')
        snapname = 'snap_'+s_str

        if label == 'merging':
            parent_dir = "/home/ashleigh/mock_map_generator/generating_sz_maps/clusters/varying_AR/merging/method_final/"
        elif label == 'control':
            parent_dir = "/home/ashleigh/mock_map_generator/generating_sz_maps/clusters/varying_AR/control/method_final/"

        fits_files = glob.glob(parent_dir + cname + snapname + '*.fits')
        
        return fits_files
    
    def get_fits_file_loc_xr(self, reg, snap, label, smoothing, energy_range='0.1_15.0'):
        
        if smoothing:
            loc = 'sph'
        else:
            loc = 'no_smoothing'
        
        clnum='0000'+str(reg)
        clnum=clnum[-4:]
        cname = "NewMDCLUSTER_"+clnum+"/"
        s_str = str(snap).rjust(3, '0')
        snapname = 'snap_'+s_str
        
        parent_dir = '/home/ashleigh/mock_map_generator/generating_x_ray_maps/maps/pyatom_' + loc + '/updated_spec_' + energy_range +'/'
        fits_files = glob.glob(parent_dir + cname + 'UPDATED_' + snapname + '*.fits') #including updated for now to ensure using the right energy range
        
        return fits_files
    
    def load_imgs_to_arrays(self, full_sample):
        
        if 'sz' in self.channels:
            full_sample['fits_file_locs_sz'] = full_sample.apply(lambda x: self.get_fits_file_loc_sz(x['region'], x['snapshot'], x['label']), axis=1)
        
        if 'xray' in self.channels:
            full_sample['fits_file_locs_xray'] = full_sample.apply(lambda x: self.get_fits_file_loc_xr(x['region'], x['snapshot'], x['label'], smoothing=self.smoothing), axis=1)
            
        sz_X_list = []
        xray_X_list = []
        y_list = []
        indices = []
        
        def load_and_resize(img):
            img = np.array(Image.fromarray(img))
            img = cv2.resize(img, dsize=(self.npxl, self.npxl), interpolation=cv2.INTER_AREA)
            return img
            
        def sort_imgs(f, X_list):
            fits_file = fits.open(f, ignore_missing_simple=True)
            X_list.append(load_and_resize(fits_file[0].data))
            # np.array(Image.fromarray(fits_file[0].data))
            fits_file.close()

        for index, row in full_sample.iterrows():
            print('loading image', row['region'], row['snapshot'])
            if row['label'] == 'merging':
                label_ind = 1
            elif row['label'] == 'control':
                label_ind = 0
            
            if 'sz' in self.channels:
                for f in row['fits_file_locs_sz'][0:self.nproj]:     
                    sort_imgs(f, sz_X_list)
                    indices.append(index)
                    y_list.append(label_ind)
                    
            if 'xray' in self.channels:
                for f in row['fits_file_locs_xray'][0:self.nproj]:
                    sort_imgs(f, xray_X_list)
#                     print('avg values', np.mean(xray_X_list[index]))
                    if len(self.channels) == 1:
                        indices.append(index)
                        y_list.append(label_ind)
           
        print(xray_X_list[0:4])
        print(sz_X_list[0:4])
                    
        return sz_X_list, xray_X_list, y_list, indices
    
    def resize_and_reshape(self, sz_X_list, xray_X_list, y_list, indices):
        
        self.y = np.array(y_list)
        self.indices = np.array(indices)
        
        if 'sz' in self.channels:
#             sz_imgs_r = [cv2.resize(img, dsize=(self.npxl, self.npxl), interpolation=cv2.INTER_AREA) for img in sz_X_list]
#             sz_imgs_r = np.asarray(sz_imgs_r).astype('float32')
            sz_imgs_r = np.asarray(sz_X_list).astype('float32')

        if 'xray' in self.channels:
#             xray_imgs_r = [cv2.resize(img, dsize=(self.npxl, self.npxl), interpolation=cv2.INTER_AREA) for img in xray_X_list]
#             xray_imgs_r = np.asarray(xray_imgs_r).astype('float32')
            xray_imgs_r = np.asarray(xray_X_list).astype('float32')

        if len(self.channels) == 2:
            self.X = np.stack((sz_imgs_r, xray_imgs_r), axis=3)
            print(self.X)

        elif self.channels == ['sz']:
            self.X = sz_imgs_r.reshape(-1, self.npxl, self.npxl, 1)

        elif self.channels == ['xray']:
            self.X = xray_imgs_r.reshape(-1, self.npxl, self.npxl, 1)
    
    def train_val_test_split(self, test_p, full_sample):
    
        # Train test split - need to ensure each cluster (and all projections) are kept in the same group
        train_p = (1-test_p)*0.9
        valid_p = (1-test_p)*0.1


        df_train, df_valtest = train_test_split(full_sample, test_size=(1-train_p), random_state=58)

        if test_p > 0:
            df_valid, df_test = train_test_split(df_valtest, test_size=test_p/(valid_p+test_p), random_state=46) # creating

        else:
            df_valid = df_valtest.copy()
            df_test = pd.DataFrame()

        self.train_indices = [self.nproj*x + i for x in df_train.index for i in list(range(0,self.nproj))]
        self.valid_indices = [self.nproj*x + i for x in df_valid.index for i in list(range(0,self.nproj))]
        self.test_indices = [self.nproj*x + i for x in df_test.index for i in list(range(0,self.nproj))]
        
        if self.folding:
    
            # k-fold splits
            self.train_dict = {}
            self.valid_dict = {}

            # prepare cross validation
            n_splits = 10
            kfold = KFold(n_splits, shuffle=True, random_state=4)
            # enumerate splits
            for i, (train, valid) in enumerate(kfold.split(pd.concat([df_train, df_valid]))):
                train_set = [self.nproj*x + i for x in train for i in list(range(0,self.nproj))] #indices for train
                valid_set = [self.nproj*x + i for x in valid for i in list(range(0,self.nproj))] #indices for test
                self.train_dict[i] = train_set
                self.valid_dict[i] = valid_set  
                
    def normalise_arr(self, training_set, arr_to_normalise):
        
        if len(self.channels) > 1:
            n_channels = len(self.channels)
            
            sz_imgs_n = arr_to_normalise.copy()
            
            for c in range(n_channels):
                flat_sz_vals = training_set[:,:,:,c].flatten()

                glob_sz_mean = np.mean(flat_sz_vals)
                glob_sz_std = np.std(flat_sz_vals)

                sz_imgs_n[:,:,:,c] = (arr_to_normalise[:,:,:,c] - glob_sz_mean) / glob_sz_std
            
        else:
    
            flat_sz_vals = training_set.flatten()

            glob_sz_mean = np.mean(flat_sz_vals)
            glob_sz_std = np.std(flat_sz_vals)

            sz_imgs_n = (arr_to_normalise - glob_sz_mean) / glob_sz_std
    
        return sz_imgs_n  
         
    def create_training_sets(self, test_p, input_arrays = [], **iteration):
        
        # Iteration is optional depending on whether folding
        full_sample=self.get_snapshot_list()
        self.train_val_test_split(test_p, full_sample)
         
        if len(input_arrays) == 0:
            sz_X_list, xray_X_list, y_list, indices = self.load_imgs_to_arrays(full_sample)
        else:
            sz_X_list, xray_X_list, y_list, indices = input_arrays
        
        print('resize and reshape')
        self.resize_and_reshape(sz_X_list, xray_X_list, y_list, indices)
        
        if self.folding:
            self.train_indices = self.train_dict[iteration['iteration']]
            self.valid_indices = self.valid_dict[iteration['iteration']]   
        
        X_train = self.X[self.train_indices]
        X_valid = self.X[self.valid_indices]
        X_test = self.X[self.test_indices]

        y_train = self.y[self.train_indices]
        y_valid = self.y[self.valid_indices]
        y_test = self.y[self.test_indices]
        
        i_train = self.indices[self.train_indices]
        i_valid = self.indices[self.valid_indices]
        i_test = self.indices[self.test_indices]
        
        print('before norm', X_train)
        
        if self.normalise:
            X_test = self.normalise_arr(X_train, X_test)
            X_valid = self.normalise_arr(X_train, X_valid)
            X_train = self.normalise_arr(X_train, X_train)
        
        print('after norm', X_train)
            
        if self.save == True:
            # check whether folder exists then save full sample + training sets + indices mapping to a folder
            folder_name = "./data/data_gen_{}".format(datetime.now().strftime('%d-%m-%Y-%H-%M'))
            if not os.path.isdir(folder_name):
                os.makedirs(folder_name)
                print("created folder : ", folder_name)

            else:
                print(folder_name, "folder already exists.")
                
            self.folder_loc = folder_name
            
            full_sample.to_csv(f'./{folder_name}/full_sample.csv', index=False)
            np.save(f"./{folder_name}/X_train", X_train)
            np.save(f"./{folder_name}/X_valid", X_valid)
            np.save(f"./{folder_name}/X_test",X_test)
            np.save(f"./{folder_name}/y_train", y_train)
            np.save(f"./{folder_name}/y_valid", y_valid)
            np.save(f"./{folder_name}/y_test", y_test)
            np.save(f"./{folder_name}/i_train", i_train)
            np.save(f"./{folder_name}/i_valid", i_valid)
            np.save(f"./{folder_name}/i_test", i_test)
            
            if self.readme is not None:
                readme = self.readme + '\n' + \
                    f'z={self.redshifts} \nfolding = {self.folding} \nnum_pixel = {self.npxl} \nnproj = {self.nproj}' \
                    f'\nchannels = {self.channels} \nnormalise= {self.normalise} \nsmoothing={self.smoothing}'
                    
                with open(f"./{folder_name}/readme.txt", 'w') as f:
                    f.write(readme)
                
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test
            
        
