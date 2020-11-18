# sklearn Grid Search
# https://stackabuse.com/grid-search-optimization-algorithm-in-python/
# Good example : https://github.com/keras-team/keras/blob/master/examples/mnist_sklearn_wrapper.py

# Below 4 lines needed to reload function definitions (after editing the project_functions.py file)
import sys, importlib
from projectFunctions import *
importlib.reload(sys.modules['projectFunctions'])
from projectFunctions import *

# tensorboard
#log_dir = "/data/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#Launch from terminal using : tensorboard --logdir data/logs/fit

print("<<<<<<<<<<===========>>>>>>>>>>>>>>")
set_x,set_y = readXYfromDisk(noOfFiles=20,folder="data")
print(set_x.shape, calcArrayMemorySize(set_x),set_y.shape)
values, counts = np.unique(set_y, axis=0, return_counts=True)
print(values,counts)

# Read Validation Data
test_set_x,test_set_y = readXYfromDisk(noOfFiles=5,folder='data/Test')
print(test_set_x.shape, calcArrayMemorySize(test_set_x),test_set_y.shape)
values, counts = np.unique(test_set_y, axis=0, return_counts=True)
print(values,counts)
print("<<<<<<<<<<===========>>>>>>>>>>>>>>")

# Define grid
l2_rate = [0.8]
lr = [1e-1]
batch_size = [256]
epochs = [1,2]
param_grid = dict(l2_rate=l2_rate,lr=lr, batch_size=batch_size, epochs=epochs )

model = KerasClassifier(build_fn = model_FC)

# Build and fit the GridSearchCV
seed = 123
validator = GridSearchCV(estimator=model, param_grid=param_grid
                     ,cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
                     ,verbose=10)

validator.fit(set_x, set_y)

print("done")



"""
# Create model and fit to training data
callback_ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50)#, restore_best_weights=True)

history = model.fit(set_x, set_y, batch_size = bs,epochs=150,verbose=1
                     ,validation_data=(test_set_x,test_set_y)
                     ,callbacks=[callback_ES])#, steps_per_epoch= (set_x.shape[0]/64)-10,shuffle=True)

#,validation_split=0.2,validation_freq=10

# Save the weights
#model.save_weights('./checkpoints/20200713/my_checkpoint')

# Create a new model instance
#model = model_FC()

# Restore the weights
#model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
#loss,acc = model.evaluate(test_set_x, test_set_y, verbose=2)

"""