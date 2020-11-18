"""
This file contains the models and other utility functions
"""

######      Import Packages     ######

import numpy as np
import pandas as pd
import h5py

from PIL import Image # load and show an image with Pillow

from datetime import datetime, timedelta #Date arithmetic
import datetime
import time
import os
import gc
import multiprocessing as mp
from os.path import exists

import glob #read csv files pattern matching
import re #regex

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold

#from tensorflow_addons.metrics.f_scores import F1Score, FBetaScore
import tensorflow_addons as tfa

from itertools import product #Create grid from dict

######      tensorflow Config    #########

#print(tf.__version__)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Memory allocation error fixed using https://www.tensorflow.org/guide/gpu
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print("Error for the fix done for mem allocation error : ",e)
 

print("Setup Mixed Precision")
# Mixed precision - GPU
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print("Mixed Precision policy applied")


IMG_SIZE = 128

##########################################################
################         Models         ################
##########################################################

def plotGridRun(df,metric):
    df_grp = df.groupby(['set'])
    for key,group in df_grp:            
        plt.plot(group['epoch'],group[metric])
        plt.plot(group['epoch'],group['val_'+ metric])    
        plt.title(group[-1:].to_string())
        plt.legend(['train_'+ metric, 'val_' + metric], loc='upper left')
        plt.show()


def plotSingleRun(df):
    #pd.DataFrame(history.history).plot(figsize=(8, 5)) 
    df.plot()
    plt.grid(True) 
    plt.gca().set_ylim(0, 1) # set the vertical range to [ 0 - 1 ]
    plt.show()

def plotFCExperiments(df,grpByCol):
    fig, ax = plt.subplots(nrows=2,ncols=2)
    NB_EXPERIMENTS_FIG_SIZE = (15,10)

    for label, grp in df.groupby(grpByCol):
        grp.plot('epoch', y = 'f1_score', ax = ax[0,0],label = label,figsize=NB_EXPERIMENTS_FIG_SIZE,title="Training F1 Score")

    for label, grp in df.groupby(grpByCol):
        grp.plot('epoch', y = 'loss', ax = ax[0,1],label = label,figsize=NB_EXPERIMENTS_FIG_SIZE,title="Training Loss")

    for label, grp in df.groupby(grpByCol):
        grp.plot('epoch', y = 'val_f1_score', ax = ax[1,0],label = label,figsize=NB_EXPERIMENTS_FIG_SIZE,title="Validation F1 Score")

    for label, grp in df.groupby(grpByCol):
        grp.plot('epoch', y = 'val_loss', ax = ax[1,1],label = label,figsize=NB_EXPERIMENTS_FIG_SIZE,title="Validation Loss"
        ,sharex=ax[0,0])

        
def extendedCSVLogger(history,gridRow,extendedCSVLogger_Fname):
   #temp_his = pd.concat([pd.DataFrame(history.history).reset_index(drop=True),pd.DataFrame(row).T.reset_index(drop=True)],axis=1)
   tempResult = pd.DataFrame(history.history)
   tempParams = pd.DataFrame(gridRow).T   
   tempParams = pd.concat([tempParams]*(len(tempResult)))
   
#   df = df.append(pd.concat([tempResult,tempParams.reset_index(drop=True)],axis=1))#,ignore_index=True)
   out = pd.concat([tempResult,tempParams.reset_index(drop=True)],axis=1)
   if exists(extendedCSVLogger_Fname):
      out.to_csv(extendedCSVLogger_Fname, sep=',',index_label='epoch',mode='a',header=False)
   else:
      out.to_csv(extendedCSVLogger_Fname, sep=',',index_label='epoch',header=True)


class MemCleanupCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()

def modelFC(learningRate,FCLayerSize,l2regRate,dropOutRate,printModelSummary=False):
#        ,keras.layers.Dropout(0.8)
#        ,kernel_initializer=tf.keras.initializers.GlorotNormal()
# model.add(keras.layers.Dropout(0.5))       

    model = keras.Sequential()
    
    model.add(keras.layers.Flatten(input_shape=(IMG_SIZE,IMG_SIZE,3)))

    model.add(keras.layers.experimental.preprocessing.Rescaling(1./255))
    #model.add(keras.layers.BatchNormalization())

    for fcls in FCLayerSize:
        for l in range(fcls[0]):
            model.add(keras.layers.Dense(int(fcls[1]),activation='relu'
                    #,kernel_regularizer=keras.regularizers.l2(l2regRate)
                    ))                        
            model.add(keras.layers.BatchNormalization())      

    model.add(keras.layers.Dense(3,activation='softmax'))

    print("<<<<<<<<<<===========>>>>>>>>>>>>>>")
    if printModelSummary == True:
        print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learningRate),
                    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    # The alternative loss function is defined as follow: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ture_y,logits=predict_y))
                    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=predict_y)),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    #loss="sparse_categorical_crossentropy",
                    metrics=['accuracy',tfa.metrics.F1Score(num_classes=3, average="weighted")]
                    #metrics=['accuracy',tfa.metrics.FBetaScore(num_classes=3, average="micro", threshold=None )]
                    #metrics=['accuracy']
                    
                    )

    return model


def modelCNN(learningRate,filterSizes,FCLayerSize,l2regRate,dropOutRate,printModelSummary=True):
#        ,keras.layers.Dropout(0.8)
#        ,kernel_initializer=tf.keras.initializers.GlorotNormal()

    model = keras.Sequential()    
    
    model.add(keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,3)))
    model.add(keras.layers.experimental.preprocessing.Rescaling(1./255))
    
    for f in filterSizes:
        for _ in range(2):
            model.add(keras.layers.Conv2D(f,(5,5)
                    ,padding='same'
                    ,activation='relu'
                    ,kernel_regularizer=keras.regularizers.l2(l2regRate)
                    ))        
        model.add(keras.layers.BatchNormalization())
        #model.add(keras.layers.Activation('relu'))        
        model.add(keras.layers.MaxPooling2D((2,2),padding='same'))

    model.add(keras.layers.Flatten())
    #model.add(keras.layers.Dropout(dropOutRate))
    
    for fcls in FCLayerSize:
        for l in range(fcls[0]):
            model.add(keras.layers.Dense(int(fcls[1])
                        ,activation='relu'
                        ,kernel_regularizer=keras.regularizers.l2(l2regRate)
                        ))            
            #model.add(keras.layers.Dropout(dropOutRate))
            model.add(keras.layers.BatchNormalization())
            

    model.add(keras.layers.Dense(3,activation='softmax'))

    print("<<<<<<<<<<===========>>>>>>>>>>>>>>")
    if printModelSummary == True:
        print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    # The alternative loss function is defined as follow: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ture_y,logits=predict_y))
                    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=predict_y)),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    #loss="sparse_categorical_crossentropy",
                    metrics=[tfa.metrics.F1Score(num_classes=3, average="weighted")]
                    #metrics=['accuracy',tfa.metrics.FBetaScore(num_classes=3, average="micro", threshold=None )]
                    #metrics=['accuracy']
                    
                    )

    return model

def modelTL(l2regRate,learningRate,hiddenUnits,printModelSummary=False):
#        ,keras.layers.Dropout(0.8)
#        ,kernel_initializer=tf.keras.initializers.GlorotNormal()
    baseModel = tf.keras.applications.vgg16(
        weights="imagenet"
        , include_top=False
        ,input_tensor=keras.layers.Input(shape=(128, 128, 3)))

    
    print("<<<<<<<<<<===========>>>>>>>>>>>>>>")
    if printModelSummary == True:
        print(model.summary())

    
    return model


    
def calcArrayMemorySize(array):
    return "Memory size is : " + str(array.nbytes/1024/1024) + " Mb"
    

