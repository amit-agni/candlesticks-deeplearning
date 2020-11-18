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

from tensorflow.keras.callbacks import Callback

import wandb
from wandb.keras import WandbCallback




### Data Loading
train = createIODataset(100,'../data/Train')
test = createIODataset(4,'../data/Test')

train = train.repeat(-1)
train = train.shuffle(buffer_size=10240*5,reshuffle_each_iteration=True)
train = train.batch(256,drop_remainder=True)
train = train.prefetch(4)

test = test.repeat(-1)
test = test.shuffle(buffer_size=10240,reshuffle_each_iteration=True)
test = test.batch(256,drop_remainder=True)
test = test.prefetch(1)




#MobileNetV2 expects pixel vaues in [-1,1], but at this point, the pixel values in your images are in [0-255]. 
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
IMG_SIZE = 128
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3),include_top=False,weights='imagenet')

# Freezing (by setting layer.trainable = False) prevents the weights in a given layer from being updated during training.
base_model.trainable = False
base_model.summary()

#This feature extractor converts each 128x128x3 image into a 5x5x1280 block of features. 
# Let's see what it does to an example batch of images. (Batch size being 256)

image_batch, label_batch = next(iter(train))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

# Add a classification head
# To generate predictions from the block of features, average over the spatial 5x5 spatial locations, 
# using a tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a single 1280-element vector per image.

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

# Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image. 
# You don't need an activation function here because this prediction will be treated as a logit,
# or a raw prediction value. Positive numbers predict class 1, negative numbers predict class 0.

# prediction_layer = tf.keras.layers.Dense(1)
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
x = tf.keras.layers.Dropout(0.3)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# Compile the model before training it. Since there are two classes, use a binary 
# cross-entropy loss with from_logits=True since the model provides a linear output.

METRICS = [keras.metrics.Precision(name='precision'),keras.metrics.Recall(name='recall')
            ,keras.metrics.AUC(name='auc'),]

base_learning_rate = 1e-4
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
            #   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=METRICS)

model.summary()


run = wandb.init(project="candlestick-CNN", name = 'TL100FilesShuffle50KSteps800_AdamLR1e-4_Base' ,reinit= True)

history = model.fit(train
                ,epochs=100
                ,steps_per_epoch=800 #800*256 = 200K
                ,verbose=1
                ,validation_data=test                
                ,validation_freq = 10
                ,validation_steps = 10
                ,callbacks=[WandbCallback()]
                )


run.finish()



# unfreeze the base_model and set the bottom layers to be un-trainable

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False
  


# As you are training a much larger model and want to readapt the pretrained weights, 
#it is important to use a lower learning rate at this stage. Otherwise, your model 
#could overfit very quickly.

# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
#               metrics=['accuracy'])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate/10),
            #   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=METRICS)

model.summary()


# Continue training the model
# If you trained to convergence earlier, this step will improve your accuracy by a few percentage points.

# fine_tune_epochs = 10
# total_epochs =  initial_epochs + fine_tune_epochs

# history_fine = model.fit(train_dataset,
#                          epochs=total_epochs,
#                          initial_epoch=history.epoch[-1],
#                          validation_data=validation_dataset)

run = wandb.init(project="candlestick-CNN", name = 'TL100FilesShuffle50KSteps800_AdamLR1e-5_Finetuned' ,reinit= True)

history = model.fit(train
                ,epochs=1000
                ,steps_per_epoch=800
                ,initial_epoch=history.epoch[-1]
                ,verbose=1
                ,validation_data=test                
                ,validation_freq = 1
                ,validation_steps = 5
                ,callbacks=[WandbCallback()]
                )


run.finish()

                         


