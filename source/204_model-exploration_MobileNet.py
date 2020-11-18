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




## Data Configuation
TRAIN_FILES_FOLDER = '../data/Train'
VAL_FILES_FOLDER = '../data/Validation'
TEST_FILES_FOLDER = '../data/Test'

TRAIN_STEPS_PER_EPOCH_MULTIPLIER = 2
VAL_STEPS_PER_EPOCH_MULTIPLIER = 2

data_config = dict(INPUT_SHAPE = (128,128,3)
                    ,TRAIN_FILES = 50
                    ,TRAIN_BATCH_SIZE = 512
                    ,VAL_FILES = 5
                    ,VAL_BATCH_SIZE = 512
                    ,PREFETCH = 5
                  )

###### buffer size reduced by half as memory overflow #########
data_config.update(TRAIN_SHUFFLE_BUFFER_SIZE = 100000)
# data_config.update(TRAIN_SHUFFLE_BUFFER_SIZE = samplesCount(data_config['TRAIN_FILES'],TRAIN_FILES_FOLDER))

data_config.update(TRAIN_STEPS_PER_EPOCH = round(data_config['TRAIN_SHUFFLE_BUFFER_SIZE']/data_config['TRAIN_BATCH_SIZE'])*TRAIN_STEPS_PER_EPOCH_MULTIPLIER)

data_config.update(VAL_SHUFFLE_BUFFER_SIZE = samplesCount(data_config['VAL_FILES'],VAL_FILES_FOLDER))
data_config.update(VAL_STEPS_PER_EPOCH = round(data_config['VAL_SHUFFLE_BUFFER_SIZE']/data_config['VAL_BATCH_SIZE'])*VAL_STEPS_PER_EPOCH_MULTIPLIER)
     
# samplesCount(data_config['TRAIN_FILES'],TRAIN_FILES_FOLDER)
# samplesCount(data_config['VAL_FILES'],VAL_FILES_FOLDER)


### Model Configuration
model_config = dict(
      EXPERIMENT = 'Mobilenetv2 Baseline'
      ,METRICS = [ keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall'),keras.metrics.AUC(name='auc')]
      ,LR = 1e-6
      ,EPOCHS = 1000
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




model_config.update(FINE_TUNE_AT = 100)
model_config.update(DROPOUT = 0.3)

#MobileNetV2 expects pixel vaues in [-1,1]
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=data_config['INPUT_SHAPE'],include_top=False,weights='imagenet')

#This feature extractor converts each 128x128x3 image into a 5x5x1280 block of features. 
# Let's see what it does to an example batch of images. (Batch size being 256)
image_batch, label_batch = next(iter(train))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# unfreeze the base_model and set the bottom layers to be un-trainable
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))
# Fine-tune from this layer onwards
fine_tune_at = model_config['FINE_TUNE_AT']
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
  

# Add a classification head
# To generate predictions from the block of features, average over the spatial 5x5 spatial locations, 
# using a tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a single 1280-element vector per image.

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image. 
prediction_layer = tf.keras.layers.Dense(1,activation='sigmoid')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

# Build a model by chaining together the rescaling, base_model and feature extractor 
# layers using the Keras Functional API. As previously mentioned, use training=False as our 
# model contains a BatchNormalization layer.

inputs = tf.keras.Input(shape=data_config['INPUT_SHAPE'])
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(model_config['DROPOUT'])(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


model.summary()




run = wandb.init(project="candlestick-CNN", name = model_config['EXPERIMENT'] ,reinit= True,dir = '../data/'
                    ,config = {**data_config,**model_config})

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=model_config['LR'])
                        ,loss=tf.keras.losses.binary_crossentropy
                        ,metrics=model_config['METRICS'])

history = model.fit(train
                ,epochs=model_config['EPOCHS']
                ,steps_per_epoch=data_config['TRAIN_STEPS_PER_EPOCH']
                ,verbose=1
                ,validation_data=val                
                ,validation_freq = model_config['VAL_FREQUENCY']
                ,validation_steps = data_config['VAL_STEPS_PER_EPOCH']
                ,callbacks=[WandbCallback()]
                # ,ckpt_callback]
                )
  
run.finish()




