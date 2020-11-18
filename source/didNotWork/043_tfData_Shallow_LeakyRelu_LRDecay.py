EXPERIMENT = 'ReduceLoss_Shallow_LessData140Files_DataShuffle200K_LRDecay_LeakyRelu'

LR = "LRDecay"

import tensorflow as tf

initial_learning_rate = 0.1
lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=800*10, #every 10 epochs
    decay_rate=0.95,
    staircase=True)

### Package Setups

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

from functions_dataCreation import *
from functions_modelArchitectures import *
from class_CallbackClasses import *

import pandas as pd

import wandb
from wandb.keras import WandbCallback

### Data Loading
train = createIODataset(140,'../data/Train')
test = createIODataset(1,'../data/Test')

train = train.repeat(-1)
train = train.shuffle(buffer_size=10240*20,reshuffle_each_iteration=True)
train = train.batch(256,drop_remainder=True)
train = train.prefetch(100)

test = test.repeat(-1)
test = test.shuffle(buffer_size=10240,reshuffle_each_iteration=True)
test = test.batch(1024,drop_remainder=True)
test = test.prefetch(10)


### Model Architecuture
IMG_SIZE = 128
ACTIVATION = tf.keras.layers.LeakyReLU()
KERNEL_INITIALISER = 'glorot_normal'
KERNEL_SIZE = (3,3)
POOL_SIZE = (6,6)
# model.add(tf.keras.layers.Dropout(DROPOUT_RATE))

model = tf.keras.Sequential(name='Base')
model.add(tf.keras.layers.Input(shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))
model.add(tf.keras.layers.BatchNormalization())            

model.add(tf.keras.layers.Conv2D(filters=10,kernel_size=KERNEL_SIZE,activation=ACTIVATION,kernel_initializer=KERNEL_INITIALISER,padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = POOL_SIZE,padding = 'same'))
model.add(tf.keras.layers.Conv2D(filters=20,kernel_size=KERNEL_SIZE,activation=ACTIVATION,kernel_initializer=KERNEL_INITIALISER,padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = POOL_SIZE,padding = 'same'))


model.add(tf.keras.layers.Flatten())    

model.add(tf.keras.layers.Dense(64,activation=ACTIVATION,kernel_initializer=KERNEL_INITIALISER))
model.add(tf.keras.layers.Dense(128,activation=ACTIVATION,kernel_initializer=KERNEL_INITIALISER))
      
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.summary()


### Training

METRICS = [keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall'),keras.metrics.AUC(name='auc'),]

wandb.init(project="candlestick-CNN", name = EXPERIMENT + str(LR) )

df = pd.DataFrame()
start_time = time.time()

checkpoint_path = '../data/callbacks/'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True,period=500)   



model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr_scheduler)
                    ,loss=tf.keras.losses.binary_crossentropy
                    ,metrics=[METRICS])

history = model.fit(train
                #,batch_size = 128
                ,epochs=5000
                ,steps_per_epoch=10240*20/256 #800
                ,verbose=1
                ,validation_data=test                
                ,validation_freq = 20
                ,validation_steps = 100
                ,callbacks=[WandbCallback()
                            ,printLR_Callback()
                            # ,cp_callback
                            ]
                )

temp = pd.DataFrame(history.history).rename_axis("epoch")
temp['elapsed'] = round((time.time() - start_time)/60,2)
var_params = EXPERIMENT + "_LR_"  + str(LR)
temp['params'] = var_params

temp.to_pickle('./evaluation-data/' + EXPERIMENT + str(LR) + '.pkl')

print("Elapsed time " + str(round((time.time() - start_time)/60,2)) + var_params)

# model.save('../data/savedmodels') 







