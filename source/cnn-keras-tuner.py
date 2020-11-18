### Package Setups
import tensorflow as tf

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
    print(e)
from tensorflow import keras
import time

from helperFunctions import *


from tensorflow.keras.callbacks import Callback

import pandas as pd

import wandb
from wandb.keras import WandbCallback



import kerastuner as kt

## Data Configuation
TRAIN_FILES_FOLDER = '../data/Train_Clean'
VAL_FILES_FOLDER = '../data/Validation_Clean'
TEST_FILES_FOLDER = '../data/Test_Clean'

TRAIN_STEPS_PER_EPOCH_MULTIPLIER = 2
VAL_STEPS_PER_EPOCH_MULTIPLIER = 2

data_config = dict(INPUT_SHAPE = (128,128,3)
                    ,TRAIN_FILES = 50
                    ,TRAIN_BATCH_SIZE = 512
                    ,VAL_FILES = 5
                    ,VAL_BATCH_SIZE = 512
                    ,PREFETCH = 5
                  )


data_config.update(TRAIN_SHUFFLE_BUFFER_SIZE = 100000)
# data_config.update(TRAIN_SHUFFLE_BUFFER_SIZE = samplesCount(data_config['TRAIN_FILES'],TRAIN_FILES_FOLDER))

data_config.update(TRAIN_STEPS_PER_EPOCH = round(data_config['TRAIN_SHUFFLE_BUFFER_SIZE']/data_config['TRAIN_BATCH_SIZE'])*TRAIN_STEPS_PER_EPOCH_MULTIPLIER)

data_config.update(VAL_SHUFFLE_BUFFER_SIZE = samplesCount(data_config['VAL_FILES'],VAL_FILES_FOLDER))
data_config.update(VAL_STEPS_PER_EPOCH = round(data_config['VAL_SHUFFLE_BUFFER_SIZE']/data_config['VAL_BATCH_SIZE'])*VAL_STEPS_PER_EPOCH_MULTIPLIER)
     
# samplesCount(data_config['TRAIN_FILES'],TRAIN_FILES_FOLDER)
# samplesCount(data_config['VAL_FILES'],VAL_FILES_FOLDER)


### Model Configuration
model_config = dict(
      EXPERIMENT = 'CNN Baseline'
      ,METRICS = [ keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall'),keras.metrics.AUC(name='auc')]
      ,LR = 1e-4
      ,EPOCHS = 100
      ,VAL_FREQUENCY = 1
)

### Data Loading
train = createIODataset(data_config['TRAIN_FILES'],TRAIN_FILES_FOLDER)
val = createIODataset(data_config['VAL_FILES'],VAL_FILES_FOLDER)

train = train.shuffle(buffer_size=data_config['TRAIN_SHUFFLE_BUFFER_SIZE'],reshuffle_each_iteration=True)
train = train.repeat(-1)
train = train.batch(data_config['TRAIN_BATCH_SIZE'],drop_remainder=True)
train = train.prefetch(data_config['PREFETCH'])

val = val.shuffle(buffer_size=data_config['VAL_SHUFFLE_BUFFER_SIZE'],reshuffle_each_iteration=True)
val = val.repeat(-1)
val = val.batch(data_config['VAL_BATCH_SIZE'],drop_remainder=True)
val = val.prefetch(data_config['PREFETCH'])





def model_builder(hp):

    #HPs :
    model_config['DROPOUT'] = hp.Float('dropout', min_value = 0.1, max_value = 0.8, step = 0.05)
    # hp_learning_rate = hp.Float('learning_rate', min_value = 1e-8, max_value = 1e-4, step = 0.1)
    model_config['LR'] = hp.Choice('learning_rate', values = [1e-4,1e-5,1e-6,1e-7,1e-8]) 


    model = tf.keras.Sequential()
    model.add(keras.layers.Input(shape=data_config['INPUT_SHAPE']))    
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))
    model.add(tf.keras.layers.BatchNormalization())            

    for filters in [8,16,32,64]:
        model.add(tf.keras.layers.Conv2D(filters = filters,kernel_size = (3,3), strides = (2,2), padding='same'
                                            ,activation='relu',kernel_initializer='he_normal'))

    model.add(tf.keras.layers.Dropout(model_config['DROPOUT']))

    model.add(tf.keras.layers.Flatten())    

    for units in [128,128]:
        model.add(tf.keras.layers.Dense(units,activation='relu',kernel_initializer='he_normal'))
       

    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=model_config['LR'])
                        ,loss=tf.keras.losses.binary_crossentropy
                        ,metrics=model_config['METRICS'])
  
    return model


model_config.update(EXPERIMENT = 'CNN Cleaned Keras Tuner')

run = wandb.init(project="candlestick-CNN", name = model_config['EXPERIMENT'] 
                    ,reinit= True,dir = '../data/'
                    ,config = {**data_config,**model_config})


tuner = kt.Hyperband(model_builder,
                     objective = 'val_loss', 
                     max_epochs = 30,
                     factor = 3,
                     hyperband_iterations = 5,
                     directory = '../data/kerastuner',
                     project_name = 'intro_to_kt')                       


# class ClearTrainingOutput(tf.keras.callbacks.Callback):
#   def on_train_end(*args, **kwargs):
#     IPython.display.clear_output(wait = True)                     


# Run the hyperparameter search. 
# The arguments for the search method are the same as those used for tf.keras.model.fit in addition to the callback above.

# callback_ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6)


tuner.search(train
                ,epochs=30
                ,steps_per_epoch=data_config['TRAIN_STEPS_PER_EPOCH']
                ,verbose=1
                ,validation_data=val                
                ,validation_freq = model_config['VAL_FREQUENCY']
                ,validation_steps = data_config['VAL_STEPS_PER_EPOCH']
                ,callbacks=[WandbCallback()]
                # ,ClearTrainingOutput()]
                # ,ckpt_callback]
                )
  

# tuner.search(set_x, set_y, epochs = 2, validation_data = (test_set_x, test_set_y)
#                 , callbacks = [ClearTrainingOutput(),callback_ES])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

import joblib
joblib.dump(tuner, "../data/kerastuner/tuner.pkl") 
#my_model_loaded = joblib.load("my_model.pkl")




