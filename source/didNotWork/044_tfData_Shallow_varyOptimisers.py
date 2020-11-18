EXPERIMENT = 'ReduceLossShallow20FilesShuffle20K_LR1e-4'

### Script arguments

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

from functions_dataCreation import *
from functions_modelArchitectures import *

from tensorflow.keras.callbacks import Callback

import pandas as pd

import wandb
from wandb.keras import WandbCallback

### Data Loading
train = createIODataset(20,'../data/Train')
test = createIODataset(1,'../data/Test')

train = train.repeat(-1)
train = train.shuffle(buffer_size=10240*2,reshuffle_each_iteration=True)
train = train.batch(256,drop_remainder=True)
train = train.prefetch(100)

test = test.repeat(-1)
test = test.shuffle(buffer_size=10240,reshuffle_each_iteration=True)
test = test.batch(1024,drop_remainder=True)
test = test.prefetch(10)


### Model Architecuture
IMG_SIZE = 128
ACTIVATION = tf.keras.layers.ReLU()
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


df = pd.DataFrame()
start_time = time.time()

checkpoint_path = '../data/callbacks/'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True,period=500)   

LR = 0.0001
#'Adadelta' : tf.keras.optimizers.Adadelta(learning_rate= LR)
# 'RMSProp' : tf.keras.optimizers.RMSprop(learning_rate= LR)
# 'Adam' : tf.keras.optimizers.Adam(learning_rate= LR)

OPTIMIZER = {'Adagrad' :tf.keras.optimizers.Adagrad(learning_rate=LR)
              ,'RMSProp_rho0.99' : tf.keras.optimizers.RMSprop(learning_rate= LR,rho=0.99)
              ,'Adam_beta10.99' : tf.keras.optimizers.Adam(learning_rate= LR,beta_1=0.99)             
}


for key,val in OPTIMIZER.items():
    model.compile(optimizer=val
                        ,loss=tf.keras.losses.binary_crossentropy
                        ,metrics=[METRICS])

    run = wandb.init(project="candlestick-CNN", name = EXPERIMENT + '_' + str(key) ,reinit= True)

    history = model.fit(train
                    #,batch_size = 128
                    ,epochs=1000
                    ,steps_per_epoch=10240*2/256 #800
                    ,verbose=1
                    ,validation_data=test                
                    ,validation_freq = 50
                    ,validation_steps = 100
                    ,callbacks=[WandbCallback()                            
                                # ,cp_callback
                                ]
                    )

    # temp = pd.DataFrame(history.history).rename_axis("epoch")
    # temp = pd.DataFrame(history.history)
    # temp['elapsed'] = round((time.time() - start_time)/60,2)
    # var_params = EXPERIMENT + '_' + str(key)
    # temp['params'] = var_params

    # df = df.append(temp)

    run.finish()

    print("Elapsed time " + str(round((time.time() - start_time)/60,2)) + var_params)

# model.save('../data/savedmodels') 
df.to_pickle('./evaluation-data/varyOptimisers2.pkl')






