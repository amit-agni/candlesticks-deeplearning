#python3 053_TransferLearning-Only-FineTuning.py

### Script arguments
import sys
import numpy as np
# FINE_TUNE_AT = np.int(sys.argv[1])
# LR = np.float(sys.argv[2])
# EPOCHS = np.int(sys.argv[3])

### Config
from functions_dataCreation import *

EPOCHS = 100
WANDB = True
FILES = 100
SAMPLES = 381000 #samplesCount(FILES,'../data/Train')
SHUFFLE_BUFFER_SIZE = 50000
BATCH_SIZE = 256
STEPS_PER_EPOCH = round(SHUFFLE_BUFFER_SIZE/BATCH_SIZE)*4
LR = 1e-5
DROPOUT = 0.5
FINE_TUNE_AT = 100

VALIDATION_FREQ = 1
VALIDATION_STEPS = 100
                
EXPERIMENT = 'TL_ShuffleThenRepeat' + \
              '_FILES' + str(FILES) + \
              '_SAMPLES' + str(round(SAMPLES/1000)) + 'k' + \
              '_SHUFFLE' + str(round(SHUFFLE_BUFFER_SIZE/1000)) + 'k' + \
              '_EPOCHSTEPS' + str(STEPS_PER_EPOCH) + \
              '_FINETUNEAT' + str(FINE_TUNE_AT) + \
              '_LR' + str(LR) + \
              '_DROPOUT' + str(DROPOUT)
                           
              

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


from tensorflow.keras.callbacks import Callback

import wandb
from wandb.keras import WandbCallback

### Data Loading
train = createIODataset(FILES,'../data/Train')
test = createIODataset(4,'../data/Test')

train = train.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE,reshuffle_each_iteration=True)
train = train.repeat(-1)
train = train.batch(256,drop_remainder=True)
train = train.prefetch(4)

test = test.shuffle(buffer_size=10240,reshuffle_each_iteration=True)
test = test.repeat(-1)
test = test.batch(256,drop_remainder=True)
test = test.prefetch(1)


#MobileNetV2 expects pixel vaues in [-1,1], but at this point, the pixel values in your images are in [0-255]. 
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SIZE = 128
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),include_top=False,weights='imagenet')

#This feature extractor converts each 128x128x3 image into a 5x5x1280 block of features. 
# Let's see what it does to an example batch of images. (Batch size being 256)
image_batch, label_batch = next(iter(train))
feature_batch = base_model(image_batch)
print(feature_batch.shape)


# unfreeze the base_model and set the bottom layers to be un-trainable
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))
# Fine-tune from this layer onwards
fine_tune_at = FINE_TUNE_AT
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

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(DROPOUT)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

METRICS = [keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall')
            ,keras.metrics.AUC(name='auc'),]


model.compile(optimizer=tf.keras.optimizers.Adam(lr=LR),            
              loss=tf.keras.losses.binary_crossentropy,
              metrics=METRICS)

model.summary()

print('\n\n\n*******************************\n\n')
print(EXPERIMENT)
print('\n\n\n*******************************\n\n')


if(WANDB):
  run = wandb.init(project="candlestick-CNN", name = EXPERIMENT ,reinit= True)
  history = model.fit(train
                ,epochs=EPOCHS
                ,steps_per_epoch=STEPS_PER_EPOCH
                ,verbose=1
                ,validation_data=test                
                ,validation_freq = VALIDATION_FREQ
                ,validation_steps = VALIDATION_STEPS
                ,callbacks=[WandbCallback()]
                )
  run.finish()
else:  
  history = model.fit(train
                ,epochs=EPOCHS
                ,steps_per_epoch=STEPS_PER_EPOCH
                ,verbose=1
                ,validation_data=test                
                ,validation_freq = VALIDATION_FREQ
                ,validation_steps = VALIDATION_STEPS
                )

