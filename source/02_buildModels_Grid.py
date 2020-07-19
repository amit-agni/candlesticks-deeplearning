# Basic hyperparameter Grid Search : Looping
# Checkpoints intermediate models if code crashes
# Custom results CSV file

from projectFunctions import *

# Get data
set_x,set_y = readXYfromDisk(noOfFiles=2,folder="data/Train") #Training data
test_set_x,test_set_y = readXYfromDisk(noOfFiles=3,folder='data/Test') #Validation data

# OHE needed for F1Score
set_y = tf.one_hot(set_y,depth = 3)
test_set_y = tf.one_hot(test_set_y,depth = 3)

# Config Grid Parameters
params = {
   "l2regRate" : [0.001]
   ,"learningRate" : [1e-2]
   ,"batchSize" : [512]
   ,"epochs" : [2,2]
}
paramsGrid = pd.DataFrame(product(*params.values()), columns=params.keys())

print(paramsGrid)

experimentShortName = "TestLoop" 
experimentShortName = experimentShortName + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_'

# loop through the grid
for index,row in paramsGrid.iterrows():
   print(index+1, ' of ', len(paramsGrid))
   print(row)
   start_time = time.time()

   callbackCP_Fname = 'data/callbacks/checkpoints/'+ 'gridRun_' + experimentShortName + str(index + 1) + '.h5'
   callbackCP = tf.keras.callbacks.ModelCheckpoint(filepath=callbackCP_Fname
                  ,save_weights_only=False,save_freq='epoch',period = 1,verbose=1)

   callbackES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, restore_best_weights=True)  
   
   model = modelFC(l2regRate = row['l2regRate'],learningRate = row['learningRate'])

   #values, counts = np.unique(set_y, axis=0, return_counts=True)
   #print('Min expected performance : ', values,counts,counts / counts.sum())
   
   history = model.fit(set_x, set_y, batch_size = int(row['batchSize']),epochs=int(row['epochs']),verbose=1
                     ,validation_data=(test_set_x,test_set_y)
                     ,callbacks=[callbackES,callbackCP]
                     ,workers=6
                     ,use_multiprocessing=True
                     ,class_weight={0:1.,1:2.,2:2.}
                     )#, steps_per_epoch= (set_x.shape[0]/64)-10,shuffle=True)

   row['set'] = index+1
   row['trainSize'] = set_x.shape[0]
   row['duration'] = time.time() - start_time
   extendedCSVLogger_Fname = 'data/callbacks/csvlogger/'+ 'gridRun_' + experimentShortName + '.csv'
   
   extendedCSVLogger(history,row,extendedCSVLogger_Fname)
   gc.collect()
   print("--- %s seconds ---" % (time.time() - start_time))




# Epoch 291/1000
# 664/664 [==============================] - 27s 41ms/step - 
# loss: 1.2276 - accuracy: 0.7264 - val_loss: 0.9702 - val_accuracy: 0.6019



# Save the weights and restore
#model.save_weights('./checkpoints/20200713/my_checkpoint')
#model = model_FC()
#model.load_weights('./checkpoints/my_checkpoint')
#loss,acc = model.evaluate(test_set_x, test_set_y, verbose=2)

# Tensorboard call back
#log_dir = "/data/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#callback_TB = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
#Launch from terminal using : tensorboard --logdir data/logs/fit

#from sklearn.model_selection import train_test_split
#set_x,null,set_y, null = train_test_split(set_x, set_y, test_size=0.0, random_state=42)
#test_set_x,null,test_set_y, null = train_test_split(test_set_x, test_set_y, test_size=0.0, random_state=40)


