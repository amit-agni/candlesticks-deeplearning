#To be completed

import tensorflow as tf
from tensorflow import keras
import IPython
import kerastuner as kt

import sys, importlib
from project_functions import *
importlib.reload(sys.modules['project_functions'])
from project_functions import *


# Read Image Data
set_x,set_y = readXYfromDisk(noOfFiles=1)
print(set_x.shape, calcArrayMemorySize(set_x),set_y.shape)
values, counts = np.unique(set_y, axis=0, return_counts=True)
print(values,counts)

# Read validation data
file = h5py.File("data/Test/SetBOQ.AX.h5", "r")
test_set_x,test_set_y = file["set_x"][:],file["set_y"][:]
file.close()
print(test_set_x.shape)


def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(128,128,3)))
    model.add(keras.layers.BatchNormalization())

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units1 = hp.Int('units1', min_value = 32, max_value = 512, step = 32)
    hp_units2 = hp.Int('units2', min_value = 32, max_value = 512, step = 32)
    hp_units3 = hp.Int('units3', min_value = 32, max_value = 512, step = 32)
    hp_units4 = hp.Int('units4', min_value = 32, max_value = 512, step = 32)
    hp_dropout = hp.Float('dropout', min_value = 0.1, max_value = 0.8, step = 0.05)
    hp_l2 = hp.Choice('l2', values = [0.1,0.01,0.001]) 

    model.add(keras.layers.Dense(units = hp_units1, activation = 'elu'
            ,kernel_initializer=tf.keras.initializers.GlorotNormal()
            ,kernel_regularizer=keras.regularizers.l2(hp_l2)))

    model.add(keras.layers.Dropout(rate = hp_dropout))

    model.add(keras.layers.Dense(units = hp_units2, activation = 'elu'
                ,kernel_initializer=tf.keras.initializers.GlorotNormal()
                ,kernel_regularizer=keras.regularizers.l2(hp_l2)))
    model.add(keras.layers.Dropout(rate = hp_dropout))

    model.add(keras.layers.Dense(units = hp_units3, activation = 'elu'
            ,kernel_initializer=tf.keras.initializers.GlorotNormal()
            ,kernel_regularizer=keras.regularizers.l2(hp_l2)))
    model.add(keras.layers.Dropout(rate = hp_dropout))

    model.add(keras.layers.Dense(units = hp_units4, activation = 'elu'
            ,kernel_initializer=tf.keras.initializers.GlorotNormal()
            ,kernel_regularizer=keras.regularizers.l2(hp_l2)))
    
    model.add(keras.layers.Dense(3,activation='softmax'))

  
    # Tune the learning rate for the optimizer 
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-3,1e-4,1e-6,1e-8]) 
  
    model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                metrics = ['accuracy'])
  
    return model


tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy', 
                     max_epochs = 5,
                     factor = 3,
                     directory = 'my_dir',
                     project_name = 'intro_to_kt')                       


class ClearTrainingOutput(tf.keras.callbacks.Callback):
  def on_train_end(*args, **kwargs):
    IPython.display.clear_output(wait = True)                     


# Run the hyperparameter search. 
# The arguments for the search method are the same as those used for tf.keras.model.fit in addition to the callback above.

callback_ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6)

tuner.search(set_x, set_y, epochs = 2, validation_data = (test_set_x, test_set_y)
                , callbacks = [ClearTrainingOutput(),callback_ES])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

