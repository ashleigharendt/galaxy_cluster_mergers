#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import profile

from keras.layers import Input, Flatten, Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, Sequential, model_from_json
from keras.optimizers import Adam
import tensorflow as tf
import gc
import os

import json
from datetime import datetime

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score

class CustomMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

class build_model():
    
    def __init__(self, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None, \
                 model_type='', classification='binary', learning_rate=0.0001, num_conv_layers=3, \
                num_dense_layers=4, batch_size=32, n_epochs=20, kernel_size=3, dropout_perc=0, batch_norm=False, init_num_filters=16, dense_neuron_list = [200,200,100,64,32], early_stopping_patience=15, readme=None, pretrained=False, n_layers_unfrozen=0):
        
        self.X_tr = X_train
        self.y_tr = y_train
        self.X_v = X_valid
        self.y_v = y_valid
        self.X_te = X_test
        self.y_te = y_test
        self.model_type = model_type
        self.classification = classification
        self.lr = learning_rate
        self.ncl = num_conv_layers
        self.ndl = num_dense_layers
        self.ks = kernel_size
        self.drop = dropout_perc
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dnl = dense_neuron_list
        self.batch_norm = batch_norm
        self.i_num_filters = init_num_filters
        self.esp = early_stopping_patience
        self.readme = readme
        self.pretrained = pretrained
        self.n_layers_unfrozen = n_layers_unfrozen
        
    def define_model(self):
        
        imsize = self.X_tr.shape[1]
        num_channels = self.X_tr.shape[3]

        input_shape = (imsize, imsize, num_channels)

        self.model = Sequential()

        if self.pretrained:
            self.X_tr = np.repeat(self.X_tr, 3, axis=3)
            self.X_v = np.repeat(self.X_v, 3, axis=3)
            self.X_te = np.repeat(self.X_te, 3, axis=3)
            
            self.X_tr = preprocess_input(self.X_tr)
            self.X_v = preprocess_input(self.X_v)
            self.X_te = preprocess_input(self.X_te)
            
            rn_model = ResNet50(
                weights='imagenet',  # Load weights pre-trained on ImageNet.
                input_shape=(96, 96, 3),
                include_top=False)
            
            rn_model.trainable = False
            
            if self.n_layers_unfrozen > 0:
                for layer in rn_model.layers[-self.n_layers_unfrozen:]:
                    layer.trainable = True
                training=True
            else:
                training=False
            
            num_channels = self.X_tr.shape[3]
            input_shape = (imsize, imsize, num_channels)
            
            inputs = Input(shape=input_shape)

            x = rn_model(inputs, training=training)

            x = GlobalAveragePooling2D()(x)
            outputs = Dense(1)(x)
            
            self.model = Model(inputs, outputs)
            
        else:
            num_channels = self.X_tr.shape[3]
            input_shape = (imsize, imsize, num_channels)

            self.model = Sequential()

            # Convolutional layers
            self.model.add(Conv2D(filters = self.i_num_filters, kernel_size = (3,3),padding = 'Same', activation ='relu', input_shape = input_shape))
            
            if self.batch_norm:
                self.model.add(BatchNormalization())
            self.model.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid'))            

        # Fully connected
        self.model.add(GlobalAveragePooling2D())

        for i in range(self.ndl, 0, -1):
            self.model.add(Dense(self.dnl[-i], activation = "relu"))

            if self.drop > 0:
                self.model.add(Dropout(self.drop))
                      
        self.model.add(Dense(1, activation = "sigmoid", name="preds"))

    def compile_model(self):
        
        self.define_model()
        # Compile Model
        optimizer = Adam(learning_rate=self.lr)
        fit_metrics = ['accuracy']
        loss = 'binary_crossentropy'
        self.model.compile(loss=loss, optimizer=optimizer, metrics=fit_metrics)
        self.model.summary()
    
    def train_model(self, verbose):
        
        self.compile_model()
    
        es = EarlyStopping(monitor='val_loss', patience=self.esp) 
#         mem = CustomMemoryCallback()
                           
        # Train
        self.history = self.model.fit(self.X_tr, self.y_tr, 
                          batch_size=self.batch_size, 
                          epochs=self.n_epochs, 
                          steps_per_epoch=self.X_tr.shape[0] // self.batch_size,
                          validation_data=(self.X_v, self.y_v),
                          shuffle=True,
                        callbacks=[es], verbose=verbose)
            
    def visualise_success(self, verbose=True):
        
        self.train_model(verbose)
        
        # plotting from history
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        epochs = list(range(len(loss)))

        figsize = (8, 6)
        fig, axis1 = plt.subplots(figsize=figsize)
        plot1_lacc = axis1.plot(epochs, acc, 'navy', label='accuracy')
        plot1_val_lacc = axis1.plot(epochs, val_acc, 'deepskyblue', label="validation accuracy")

        plot1_loss = axis1.plot(epochs, loss, 'red', label='loss')
        plot1_val_loss = axis1.plot(epochs, val_loss, 'lightsalmon', label="validation loss")

        plots = plot1_loss + plot1_val_loss
        labs = [plot.get_label() for plot in plots]
        axis1.set_xlabel('Epoch')
        axis1.set_ylabel('Loss/Accuracy')
        plt.title("Loss/Accuracy History (Pristine Images)")
        plt.tight_layout()
        axis1.legend(loc='upper left')
        plt.show()

    def output_results(self, model=None, X_te=None, y_te=None):
        
        if model is None:
            model = self.model
        if X_te is None:
            X_te = self.X_te
        if y_te is None:
            y_te = self.y_te
        
        predictions = model.predict(X_te)

        threshold = 0.5
        y_prob = np.array(predictions)
        y_pred = np.array([1 if x > 0.5 else 0 for x in predictions])
        
        
        fpr, tpr, thresholds = roc_curve(y_te, y_prob)
        auc = roc_auc_score(y_te, y_prob)
        precision = precision_score(y_te, y_pred)
        recall = recall_score(y_te, y_pred)
        acc = accuracy_score(y_te, y_pred)

        auc = round(auc, 4)
        prec = round(precision, 4)
        recall = round(recall, 4)
        acc = round(acc, 4)
        
        print('AUC:', auc, '\nPrecision:', prec, '\nRecall:', recall, '\nAccuracy:', acc)
        
        return auc, prec, recall, acc
    
    def save_model(self):
        
        folder_name = "./models/model_{}".format(datetime.now().strftime('%d-%m-%Y-%H-%M'))
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)
            print("created folder : ", folder_name)

        else:
            print(folder_name, "folder already exists.")
            
        if self.readme is not None:
            readme = self.readme + '\n' + \
            f'model_type={self.model_type} \nlearning_rate = {self.lr} \nnum_conv_layers = {self.ncl} \nnum_dense_layers = {self.ndl}' \
            f'\nkernel_size = {self.ks} \ndropout_perc= {self.drop} \nbatch_size={self.batch_size} \nn_epochs={self.n_epochs}' \
            f'\nbatch_norm = {self.batch_norm} \ninit_num_filters= {self.i_num_filters} \nearly_stopping_patience= {self.esp}'

            with open(f"./{folder_name}/readme.txt", 'w') as f:
                f.write(readme)
        
        model_json = self.model.to_json()

        with open(folder_name + '/model.json', "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        self.model.save_weights(folder_name + '/model.h5')
        print("Saved model to", folder_name)
