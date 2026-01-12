import numpy as np
from astropy.io import fits
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
import random
from astropy.visualization.mpl_normalize import simple_norm
import tensorflow as tf
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from vis.utils import utils
from model import build_model
from data_load import data_preprocess

readme = 'Pre/post inclusive for sz and xray - multihead, no batch norm - corrected so same num convolutions'

# print('Preparing data')
# data_prep = data_preprocess(folding=False, classification='mergers_only', num_pixel=96, redshifts=[1,2,3], channels=['xray', 'sz'], normalise=False, save=True, \
#                             readme=readme, r200_zoom=False, energy_range='0.1_15.0')

# X_train, X_valid, X_test, y_train, y_valid, y_test = data_prep.create_training_sets(test_p = 0.1)

# folder_loc = data_prep.folder_loc + '/'

folder_loc = './data/data_gen_20-07-2023-03-59/'
X_train = np.load(folder_loc + "X_train.npy")
X_valid = np.load(folder_loc + "X_valid.npy")
X_test = np.load(folder_loc + "X_test.npy")
y_train = np.load(folder_loc + "y_train.npy")
y_valid = np.load(folder_loc + "y_valid.npy")
y_test = np.load(folder_loc + "y_test.npy")
i_train = np.load(folder_loc + "i_train.npy")
i_valid = np.load(folder_loc + "i_valid.npy")
i_test = np.load(folder_loc + "i_test.npy")

print('Building model')

print(X_train.shape)

model_class = build_model(X_train, y_train, X_valid, y_valid, X_test, y_test, \
                         learning_rate=0.0001, num_conv_layers=3, num_dense_layers=4, batch_size=32, n_epochs=200, \
                         kernel_size=3, dropout_perc=0.5, batch_norm=False, readme=f'{readme}\n Data is found at: {folder_loc}', \
                          normalise=True, model_type='multihead', log=False)

model_class.visualise_success()
model_class.save_model()
model_class.output_results()

# model_class.output_imbalanced_results()