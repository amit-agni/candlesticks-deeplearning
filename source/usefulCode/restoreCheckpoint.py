from projectFunctions import *

set_x,set_y = readXYfromDisk(noOfFiles=20,folder="data")
test_set_x,test_set_y = readXYfromDisk(noOfFiles=3,folder='data/Test')



# checkpoint_path = "data/checkpoints/model.h5"
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=False,
#                                                  save_freq='epoch',
#                                                  period = 4,
#                                                  verbose=1)


# Include the epoch in the file name (uses `str.format`)
CHECKPOINT_PATH = "../data/callbacks/checkpoints/" + model_config['EXPERIMENT'] + "/cp-{epoch:04d}.ckpt"
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 monitor = 'loss',
                                                 save_best_only = False,
                                                 save_freq='epoch',
                                                #  period = 5,
                                                 verbose=1)




callback_ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100)#, restore_best_weights=True)   

model = tf.keras.models.load_model(checkpoint_path)
result = model.fit(set_x, set_y, batch_size = 128,epochs=1,verbose=1
                     ,validation_data=(test_set_x,test_set_y)
                     ,callbacks=[callback_ES,cp_callback]
                     ,workers=6
                     ,use_multiprocessing=True
                     ,class_weight={0:1.,1:2.,2:2.}
                     )#, steps_per_epoch= (set_x.shape[0]/64)-10,shuffle=True)

tempResult = pd.DataFrame(result.history)

print(tempResult.head())
tempResult.to_csv("data/amit1.txt", sep=',',index_label='epoch',header=True)