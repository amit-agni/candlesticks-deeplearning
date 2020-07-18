# Basic hyperparameter Grid Search
# Checkpoints intermediate models if code crashes

from projectFunctions import *

# Get data
set_x,set_y = readXYfromDisk(noOfFiles=20,folder="data") #Training data
test_set_x,test_set_y = readXYfromDisk(noOfFiles=3,folder='data/Test') #Validation data

# Config Parameters
resultsFname = "results.csv"
experimentShortName = ["FC15"]
experimentDescription = ["FC15 class_weight 1:2:2 and L2"]
timeStamp = [datetime.datetime.now().strftime('%Y%m%d_%H%M%S')]
l2regRate = [0.00001]
learningRate = [1e-6]
batchSize = [128]
epochs = [1000]
params = dict(experimentShortName = experimentShortName,experimentDescription=experimentDescription,timeStamp=timeStamp
               ,batchSize=batchSize,learningRate=learningRate, epochs=epochs
               ,l2regRate=l2regRate)
params = pd.DataFrame(product(*params.values()), columns=params.keys())

# loop through the grid
for index,row in params.iterrows():
   print(index+1, ' of ', len(params))
   print(row)
   start_time = time.time()

   callbackCP = tf.keras.callbacks.ModelCheckpoint(
      filepath='data/callbacks/checkpoints/'+ row['timeStamp'] + '_' 
               + row['experimentShortName'] + index + '.h5'
      ,save_weights_only=False,save_freq='epoch',period = 4,verbose=1)

   callbackES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100)#, restore_best_weights=True)   

   callbackCSV= tf.keras.callbacks.CSVLogger(
      'data/results/'+ 'callbackCSV_' + row['timeStamp'] + '_' + row['experimentShortName'] + '.csv'
      ,append=True)

   model = model_FC(l2regRate = row['l2regRate'],learningRate = row['learningRate'])
   
   values, counts = np.unique(set_y, axis=0, return_counts=True)
   print('Min expected performance : ', values,counts,counts / counts.sum())

   result = model.fit(set_x, set_y, batch_size = int(row['batchSize']),epochs=int(row['epochs']),verbose=1
                     ,validation_data=(test_set_x,test_set_y)
                     ,callbacks=[callbackES,callbackCP]
                     ,workers=6
                     ,use_multiprocessing=True
                     ,class_weight={0:1.,1:2.,2:2.}
                     )#, steps_per_epoch= (set_x.shape[0]/64)-10,shuffle=True)

   row['set'] = index+1
   row['trainSize'] = set_x.shape[0]
   row['duration'] = time.time() - start_time

   #temp_his = pd.concat([pd.DataFrame(history.history).reset_index(drop=True),pd.DataFrame(row).T.reset_index(drop=True)],axis=1)
   tempResult = pd.DataFrame(result.history)
   tempParams = pd.DataFrame(row).T   
   tempParams = pd.concat([tempParams]*(len(tempResult)))
   
   resultsFname = 'data/results/'+ row['timeStamp'] + '_' + row['experimentShortName'] + '.csv'
#   df = df.append(pd.concat([tempResult,tempParams.reset_index(drop=True)],axis=1))#,ignore_index=True)
   out = pd.concat([tempResult,tempParams.reset_index(drop=True)],axis=1)
   if exists(resultsFname):
      out.to_csv(resultsFname, sep=',',index_label='epoch',mode='a',header=False)
   else:
      out.to_csv(resultsFname, sep=',',index_label='epoch',header=True)

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


