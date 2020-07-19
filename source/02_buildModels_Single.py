# Baby-sit single model
# Use callbacks

from projectFunctions import *

# Get data
set_x,set_y = readXYfromDisk(noOfFiles=2,folder="data/Train") #Training data
test_set_x,test_set_y = readXYfromDisk(noOfFiles=3,folder='data/Test') #Validation data

# OHE needed for F1Score
set_y = tf.one_hot(set_y,depth = 3)
test_set_y = tf.one_hot(test_set_y,depth = 3)

# Config Parameters
params = {"l2regRate" : 0.0001
         ,"learningRate" : 1e-5
         ,"batchSize" : 512
         ,"epochs" : 2}

experimentShortName = "Test" 
experimentShortName = experimentShortName + '_' + datetime.datetime.now().strftime('%Y%m%d') + '_' + '_'.join("{!s}{!r}".format(key,val) for (key,val) in params.items())

#Callbacks
callbackCP_Fname = 'data/callbacks/checkpoints/'+ 'singleRun_' + experimentShortName + '.h5'
callbackCSV_Fname = 'data/callbacks/csvlogger/'+ 'singleRun_' +  experimentShortName + '.csv'

callbackCP = tf.keras.callbacks.ModelCheckpoint(filepath=callbackCP_Fname,save_weights_only=False,save_freq='epoch',period = 1,verbose=1)
callbackCSV= tf.keras.callbacks.CSVLogger(filename = callbackCSV_Fname,append=False)
callbackES = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=100)#, restore_best_weights=True)   


#callbackCSV.set_params({"batch" : 232323})
#callbackPB = tf.keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
#callbackLRPlateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_f1_score', factor=0.2,patience=2, min_lr=0,verbose=1)

model = modelFC(l2regRate = params["l2regRate"],learningRate = params["learningRate"])#,printModelSummary=True)

history = model.fit(
   set_x, set_y, batch_size = params["batchSize"],epochs=params["epochs"],verbose=1
   ,validation_data=(test_set_x,test_set_y)
   ,callbacks=[callbackCP,callbackCSV,callbackES]
   ,workers=6
   ,use_multiprocessing=True
   ,class_weight={0:0.73,1:3.23,2:3.04} #to be done
   )#, steps_per_epoch= (set_x.shape[0]/64)-10,shuffle=True)
