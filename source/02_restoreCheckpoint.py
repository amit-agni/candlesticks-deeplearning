from projectFunctions import *

checkpoint_path = "data/checkpoints/model.h5"
model = tf.keras.models.load_model(checkpoint_path)

set_x,set_y = readXYfromDisk(noOfFiles=20,folder="data")
test_set_x,test_set_y = readXYfromDisk(noOfFiles=3,folder='data/Test')

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=False,
                                                 save_freq='epoch',
                                                 period = 4,
                                                 verbose=1)


callback_ES = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100)#, restore_best_weights=True)   

result = model.fit(set_x, set_y, batch_size = 128,epochs=1000,verbose=1
                     ,validation_data=(test_set_x,test_set_y)
                     ,callbacks=[callback_ES,cp_callback]
                     ,workers=6
                     ,use_multiprocessing=True
                     ,class_weight={0:1.,1:2.,2:2.}
                     )#, steps_per_epoch= (set_x.shape[0]/64)-10,shuffle=True)
