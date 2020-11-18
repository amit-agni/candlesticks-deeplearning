### TO BE COMPLETED

# Below 4 lines needed to reload function definitions (after editing the project_functions.py file)
import sys, importlib
from project_functions import *
importlib.reload(sys.modules['project_functions'])
from project_functions import *

# tensorboard
log_dir = "/data/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callback_TB = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#Launch from terminal using : tensorboard --logdir data/logs/fit

#rm -rf data/logs/

import shutil
shutil.rmtree('data/logs/')


# Read Image Data
set_x,set_y = readXYfromDisk(noOfFiles=18,folder = "data")
print(set_x.shape, calcArrayMemorySize(set_x),set_y.shape)
values, counts = np.unique(set_y, axis=0, return_counts=True)
print(values,counts)

# Read validation data
file = h5py.File("data/Test/SetBOQ.AX.h5", "r")
test_set_x,test_set_y = file["set_x"][:],file["set_y"][:]
file.close()
print(test_set_x.shape)


### HP tuning

HP_NUM_UNITS1 = hp.HParam('num_units1', hp.Discrete([512,256])) 
HP_NUM_UNITS2 = hp.HParam('num_units2', hp.Discrete([128,64]))
#HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.2,0.8))
HP_DROPOUT = hp.HParam('dropout',hp.Discrete([0.05,0.1,0.8]))
#HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd','RMSprop']))
#HP_L2 = hp.HParam('l2 regularizer', hp.RealInterval(.001,.01))
HP_L2 = hp.HParam('l2 regularizer',hp.Discrete([0.1,0.01,0.001]))
HP_LR = hp.HParam('lr', hp.Discrete([1e-4,1e-6]))
HP_BATCHSIZE = hp.HParam('Batch Size', hp.Discrete([64,128]))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('data/logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS1,HP_NUM_UNITS2, HP_DROPOUT,HP_L2,HP_BATCHSIZE,HP_LR],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Val Accuracy')],
  )

callback_ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

def train_test_model(hparams,set_x,set_y,test_set_x,test_set_y):
    model = tf.keras.Sequential([
      keras.layers.Flatten(input_shape=(128,128,3))
    ,keras.layers.BatchNormalization()

    ,keras.layers.Dense(hparams[HP_NUM_UNITS1], activation='elu'
                    ,kernel_initializer=tf.keras.initializers.GlorotNormal()
                   ,kernel_regularizer=keras.regularizers.l2(hparams[HP_L2]))
    ,keras.layers.Dropout(hparams[HP_DROPOUT]) 

    ,keras.layers.Dense(hparams[HP_NUM_UNITS1], activation='elu'
                    ,kernel_initializer=tf.keras.initializers.GlorotNormal()
                   ,kernel_regularizer=keras.regularizers.l2(hparams[HP_L2]))
    ,keras.layers.Dropout(hparams[HP_DROPOUT]) 

    ,keras.layers.Dense(hparams[HP_NUM_UNITS1], activation='elu'
                    ,kernel_initializer=tf.keras.initializers.GlorotNormal()
                   ,kernel_regularizer=keras.regularizers.l2(hparams[HP_L2]))
    ,keras.layers.Dropout(hparams[HP_DROPOUT]) 

    ,keras.layers.Dense(hparams[HP_NUM_UNITS2], activation='elu'
                    ,kernel_initializer=tf.keras.initializers.GlorotNormal()
                   ,kernel_regularizer=keras.regularizers.l2(hparams[HP_L2]))

    ,keras.layers.Dense(hparams[HP_NUM_UNITS2], activation='elu'
                    ,kernel_initializer=tf.keras.initializers.GlorotNormal()
                   ,kernel_regularizer=keras.regularizers.l2(hparams[HP_L2]))

    ,keras.layers.Dense(3,activation='softmax')

  ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hparams[HP_LR]),
              #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              #loss=tf.keras.losses.CategoricalCrossentropy(),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
              
    #model.fit(set_x, set_y, batch_size = 128,epochs=50,verbose=1,callbacks=[tensorboard_callback])
    model.fit(set_x, set_y, batch_size = hparams[HP_BATCHSIZE],epochs=1000,verbose=1
              ,validation_data=(test_set_x,test_set_y)
              ,callbacks=[callback_TB]) 
        
    _,accuracy = model.evaluate(test_set_x,  test_set_y, verbose=1)
    
    return accuracy



def run(run_dir, hparams,set_x,set_y,test_set_x,test_set_y):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams,set_x,set_y,test_set_x,test_set_y)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)



#Run the simulation
session_num = 0
for num_units1 in HP_NUM_UNITS1.domain.values:
  for num_units2 in HP_NUM_UNITS2.domain.values:
      for dropout_rate in HP_DROPOUT.domain.values:
    #for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
      #for l2 in (HP_L2.domain.min_value, HP_L2.domain.max_value):
        #for optimizer in HP_OPTIMIZER.domain.values:
          for l2 in HP_L2.domain.values:
            for lr in HP_LR.domain.values:
              for batch in HP_BATCHSIZE.domain.values:
          
                hparams = {
                    HP_NUM_UNITS1: num_units1,
                    HP_NUM_UNITS2: num_units2,
                    HP_DROPOUT: dropout_rate,
                    HP_L2: l2,
                    #HP_OPTIMIZER: optimizer,                
                    HP_BATCHSIZE: batch,
                    HP_LR: lr
                    }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('data/logs/hparam_tuning/' + run_name, hparams,set_x,set_y,test_set_x,test_set_y)
                session_num += 1
                



