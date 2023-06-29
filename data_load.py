#!/usr/bin/env python
# coding: utf-8

import os, sys, glob
from astropy.io import fits
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split, KFold

class data_preprocess():
    
    def __init__(self, redshifts=[1,2,3], classification='binary', folding=False, num_pixel=96, nproj = 29, channels=['sz'], normalise = True):
        
        self.redshifts = redshifts
        self.classification = classification
        self.folding = folding
        self.npxl = num_pixel
        self.nproj = nproj
        self.channels = channels # list containing either 'sz', 'xray' or both
        self.normalise = normalise
        
#         if len(self.channels) < 2:
#             self.normalise = True
#         else:
#             self.normalise = False

        print('data preprocess has been called')
    
    def get_snapshot_list(self):
        
        # Loading merging population
        full_sample = pd.read_csv('/nfs/scratch/arendtas/the_three_hundred/cluster_list/full_sample.csv')

        # Filter for certain redshift bins
        full_sample = full_sample[full_sample['z_bin'].isin(self.redshifts)].reset_index(drop=True)
        
        return full_sample
        
    def get_fits_file_loc_sz(self, reg, snap, label):

        clnum='0000'+str(reg)
        clnum=clnum[-4:]
        cname = "NewMDCLUSTER_"+clnum+"/"
        s_str = str(snap).rjust(3, '0')
        snapname = 'snap_'+s_str

        if label == 'merging':
            parent_dir = "/nfs/scratch/arendtas/the_three_hundred/varying_z/sz/merging/"
        elif label == 'control':
            parent_dir = "/nfs/scratch/arendtas/the_three_hundred/varying_z/sz/control/"

        fits_files = glob.glob(parent_dir + cname + snapname + '*.fits')
        
        return fits_files
    
    def get_fits_file_loc_xr(self, reg, snap, label, smoothing=False, energy_range='0.1_15.0'):
        
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
            full_sample['fits_file_locs_xray'] = full_sample.apply(lambda x: self.get_fits_file_loc_xr(x['region'], x['snapshot'], x['label']), axis=1)
            
        sz_X_list = []
        sz_y_list = []
        xray_X_list = []
        xray_y_list = []
        
        def sort_imgs(f, X_list, y_list):
            fits_file = fits.open(f, ignore_missing_simple=True)
            img = cv2.resize(np.array(Image.fromarray(fits_file[0].data)), dsize=(self.npxl, self.npxl), interpolation=cv2.INTER_AREA) 
            X_list.append(img)
            # np.array(Image.fromarray(fits_file[0].data))
            if row['label'] == 'merging':
                label_ind = 1
            elif row['label'] == 'control':
                label_ind = 0
            y_list.append(label_ind)
            fits_file.close()

        for index, row in full_sample.iterrows():
            if 'sz' in self.channels:
                for f in row['fits_file_locs_sz'][0:self.nproj]:     
                    sort_imgs(f, sz_X_list, sz_y_list)
            if 'xray' in self.channels:
                for f in row['fits_file_locs_xray'][0:self.nproj]:     
                    sort_imgs(f, xray_X_list, xray_y_list)
        
        return sz_X_list, sz_y_list, xray_X_list, xray_y_list
    
    def resize_and_reshape(self, sz_X_list, sz_y_list, xray_X_list, xray_y_list):
        
        if 'sz' in self.channels:
#             sz_imgs_r = [cv2.resize(img, dsize=(self.npxl, self.npxl), interpolation=cv2.INTER_AREA) for img in sz_X_list]
            sz_imgs_r = [img for img in sz_X_list]
            sz_imgs_r = np.asarray(sz_imgs_r).astype('float32')
            sz_y = np.array(sz_y_list)

        if 'xray' in self.channels:
            xray_imgs_r = [cv2.resize(img, dsize=(self.npxl, self.npxl), interpolation=cv2.INTER_AREA) for img in xray_X_list]
            xray_imgs_r = np.asarray(xray_imgs_r).astype('float32')
            xray_y = np.array(xray_y_list)

        if len(self.channels) == 2:
            self.X = np.stack((sz_imgs_r, xray_imgs_r), axis=3)
            print(self.X)
            self.y = sz_y

        elif self.channels == ['sz']:
            self.X = sz_imgs_r.reshape(-1, self.npxl, self.npxl, 1)
            self.y = sz_y

        elif self.channels == ['xray']:
            self.X = xray_imgs_r.reshape(-1, self.npxl, self.npxl, 1)
            self.y = xray_y
    
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
    
        flat_sz_vals = training_set.flatten()

        glob_sz_mean = np.mean(flat_sz_vals)
        glob_sz_std = np.std(flat_sz_vals)

        sz_imgs_n = (arr_to_normalise - glob_sz_mean) / glob_sz_std
    
        return sz_imgs_n  
         
    def create_training_sets(self, test_p, input_arrays = [], **iteration):
        
        # Iteration is optional depending on whether folding
        full_sample=self.get_snapshot_list()
        print('about to load images')
        self.train_val_test_split(test_p, full_sample)
         
        if len(input_arrays) == 0:
            sz_X_list, sz_y_list, xray_X_list, xray_y_list = self.load_imgs_to_arrays(full_sample)
        else:
            [sz_X_list, sz_y_list, xray_X_list, xray_y_list] = input_arrays
        
        self.resize_and_reshape(sz_X_list, sz_y_list, xray_X_list, xray_y_list)
        
        if self.folding:
            self.train_indices = self.train_dict[iteration['iteration']]
            self.valid_indices = self.valid_dict[iteration['iteration']]   
        
        X_train = self.X[self.train_indices]
        X_valid = self.X[self.valid_indices]
        X_test = self.X[self.test_indices]

        y_train = self.y[self.train_indices]
        y_valid = self.y[self.valid_indices]
        y_test = self.y[self.test_indices]
        
        if self.normalise:
            X_test = self.normalise_arr(X_train, X_test)
            X_valid = self.normalise_arr(X_train, X_valid)
            X_train = self.normalise_arr(X_train, X_train)
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test
            
        
