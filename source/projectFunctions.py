
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

import mplfinance as mpf
from PIL import Image # load and show an image with Pillow
#import PIL
#print('Pillow Version:', PIL.__version__)
from numpy.random import randint
from numpy.random import seed

from datetime import datetime, timedelta #Date arithmetic
import datetime
import time
import os
import gc
import multiprocessing as mp
from os.path import exists


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold

#from tensorflow_addons.metrics.f_scores import F1Score, FBetaScore
import tensorflow_addons as tfa

from itertools import product #Create grid from dict

#print(tf.__version__)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Memory allocation error fixed using https://www.tensorflow.org/guide/gpu
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
    print("Error for the fix done for mem allocation error : ",e)
 

print("Setup Mixed Precision")
# Mixed precision - GPU
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
print("Mixed Precision policy applied")


DATE_WINDOW = 6
UP_THRESHOLD_PCT = 1.5
DOWN_THRESHOLD_PCT = 1.5
IMG_SIZE = 128


def plotExperiments(df):
    df_grp = df.groupby(['set'])
    for key,group in df_grp:            
        plt.plot(group['epoch'],group['accuracy'])
        plt.plot(group['epoch'],group['val_accuracy'])    
        plt.title(group[-1:].to_string())
        plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
        plt.show()


def plotLoss(history):
    #pd.DataFrame(history.history).plot(figsize=(8, 5)) 
    plt.grid(True) 
    plt.gca().set_ylim(0, 1) # set the vertical range to [ 0 - 1 ]
    plt.show()


def extendedCSVLogger(history,gridRow,extendedCSVLogger_Fname):
   #temp_his = pd.concat([pd.DataFrame(history.history).reset_index(drop=True),pd.DataFrame(row).T.reset_index(drop=True)],axis=1)
   tempResult = pd.DataFrame(history.history)
   tempParams = pd.DataFrame(gridRow).T   
   tempParams = pd.concat([tempParams]*(len(tempResult)))
   
#   df = df.append(pd.concat([tempResult,tempParams.reset_index(drop=True)],axis=1))#,ignore_index=True)
   out = pd.concat([tempResult,tempParams.reset_index(drop=True)],axis=1)
   if exists(extendedCSVLogger_Fname):
      out.to_csv(extendedCSVLogger_Fname, sep=',',index_label='epoch',mode='a',header=False)
   else:
      out.to_csv(extendedCSVLogger_Fname, sep=',',index_label='epoch',header=True)




def modelFC(l2regRate,learningRate,printModelSummary=False):
#        ,keras.layers.Dropout(0.8)
#        ,kernel_initializer=tf.keras.initializers.GlorotNormal()

    model = keras.Sequential()
    
    model.add(keras.layers.Flatten(input_shape=(IMG_SIZE,IMG_SIZE,3)))
    model.add(keras.layers.BatchNormalization())

    for _ in range(5):
        model.add(keras.layers.Dense(256, activation='relu'
                    ,kernel_regularizer=keras.regularizers.l2(l2regRate)))
        model.add(keras.layers.BatchNormalization())

    for _ in range(5):
        model.add(keras.layers.Dense(128, activation='relu'
                    ,kernel_regularizer=keras.regularizers.l2(l2regRate)))
        model.add(keras.layers.BatchNormalization())

    for _ in range(5):
        model.add(keras.layers.Dense(64, activation='relu'
                    ,kernel_regularizer=keras.regularizers.l2(l2regRate)))
        model.add(keras.layers.BatchNormalization())   

    model.add(keras.layers.Dense(3,activation='softmax'))

    print("<<<<<<<<<<===========>>>>>>>>>>>>>>")
    if printModelSummary == True:
        print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                    #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    # The alternative loss function is defined as follow: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ture_y,logits=predict_y))
                    # loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=predict_y)),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    #loss="sparse_categorical_crossentropy",
                    metrics=[tfa.metrics.F1Score(num_classes=3, average="weighted")]
                    #metrics=['accuracy',tfa.metrics.FBetaScore(num_classes=3, average="micro", threshold=None )]
                    #metrics=['accuracy']
                    
                    )

    return model


def f1_weighted(true, pred): #shapes (batch, 4)
#Source : https://stackoverflow.com/questions/59963911/how-to-write-a-custom-f1-loss-function-with-weighted-average-for-keras

    #for metrics include these two lines, for loss, don't include them
    #these are meant to round 'pred' to exactly zeros and ones
    predLabels = K.argmax(pred, axis=-1)
    pred = K.one_hot(predLabels, 4) 


    ground_positives = K.sum(true, axis=0)       # = TP + FN
    pred_positives = K.sum(pred, axis=0)         # = TP + FP
    true_positives = K.sum(true * pred, axis=0)  # = TP
        #all with shape (4,)

    precision = (true_positives + K.epsilon()) / (pred_positives + K.epsilon()) 
    recall = (true_positives + K.epsilon()) / (ground_positives + K.epsilon()) 
        #both = 1 if ground_positives == 0 or pred_positives == 0
        #shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        #not sure if this last epsilon is necessary
        #matematically not, but maybe to avoid computational instability
        #still with shape (4,)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)


    return 1 - weighted_f1 #for metrics, return only 'weighted_f1'

    


def calcArrayMemorySize(array):
    return "Memory size is : " + str(array.nbytes/1024/1024) + " Mb"
    

def readXYfromDisk(noOfFiles,folder):
    """
    # Reads .h5 files
    # Appends them to a list
    # Finally converts the list to np
    """

    set_x = []
    set_y = []
    
    file_counter = 1
    for file in os.listdir(folder):
        if file_counter > noOfFiles:
            break
        if file.endswith(".h5"):
            fname = os.path.join(folder, file)
            file = h5py.File(fname, "r")
            set_x_temp = file["set_x"][:]
            set_y_temp = file["set_y"][:]

            set_x.append(set_x_temp)
            set_y.append(set_y_temp)

            file.close()
            file_counter += 1

    # Since set_x and set_y are list of arrays, use np.concatenate()
    # Better than result_array = np.array(result) ?

    set_x = np.concatenate(set_x,axis=0)
    set_y = np.concatenate(set_y,axis=0)

    print('X Shape : ', set_x.shape, calcArrayMemorySize(set_x)
            ,'Y Shape: ',set_y.shape)
    values, counts = np.unique(set_y, axis=0, return_counts=True)
    print('Values, counts, Avg Performance : ', values,counts,counts / counts.sum())


    return set_x,set_y

##########################################################
################    Data Prep Functions   ################
##########################################################

def saveXYtoDisk(result,folder,fname):
    """
    Function that separates x and y
    And also created .h5 files to save the arrays
    """
    first = [x for (x,y) in result]
    set_y = np.concatenate(first,axis =0)

    second = [y for (x,y) in result]
    set_x = np.concatenate(second,axis =0)

    print(fname)

    file = h5py.File(folder + fname + ".h5", "w")
    file.create_dataset('set_x', data=set_x,dtype='uint8')
    file.create_dataset('set_y', data=set_y,dtype='uint8')
    file.close()


def setTargetLabel(val):
    if val > UP_THRESHOLD_PCT: out = 2
    elif val < -DOWN_THRESHOLD_PCT: out = 1
    else: out = 0
    return out


def dataPrep(fname):
    df = pd.read_csv(fname,parse_dates=[1]) #,index_col=1)
    df.columns = ['Symbol','Date','Open','High','Low','Close','Volume','Adjusted']
    df = df[df['Close'] > 0]
    #Compute Growth and Target column
    df['Close_Prev'] = df.groupby(['Symbol'])['Close'].shift(1)
    df = df[df['Close_Prev'] > 0] #Remove rows with no data
    df['Target_val'] = (100 * ((df['Close']/df['Close_Prev']) - 1))
    df['Target'] = (100 * ((df['Close']/df['Close_Prev']) - 1)).apply(setTargetLabel)
    #df.groupby(['Target'])['Target'].count()
    return df

def createCandlesticksPlot(data,index,targetLabel,inRAM = True):  
    """
    Core function that creates the plot and saves to a file
    Argument:
    data -- to be done
    index -- to be done
    Returns:
    None -- to be done

    """      
    title = data[:1]['Date'].item().strftime('%d-%b-%Y') + " " + data[-1:]['Date'].item().strftime('%d-%b-%Y') + " Lbl:" + str(targetLabel)

    # To set a datetime as an index
    data = data.set_index(pd.DatetimeIndex(data['Date'])) 


   
    #Create custom styles
    mc = mpf.make_marketcolors(up='g',down='r')
    rc = {'xtick.major.bottom':False
        ,'ytick.major.left':False
        ,'xtick.major.size':0
        ,'ytick.major.size':0
        ,'axes.labelsize' : 0
        ,'savefig.jpeg_quality' : 95
        ,'savefig.bbox':'tight'


#        ,'patch.linewidth' : 100 #candle border
#        ,'lines.linewidth' : 100 #wick width        

#        ,'lines.markeredgewidth' : 40.0         
#        ,'hatch.linewidth' : 40.0
#        ,'axes.linewidth' : 40.0

        ,'axes.spines.left' :False #plot border
        ,'axes.spines.top' :False
        ,'axes.spines.bottom' :False
        ,'axes.spines.right' :False
        }
    s  = mpf.make_mpf_style(marketcolors=mc,rc = rc)
    
    # First we set the kwargs that we will use for all of these examples:
    kwargs = dict(type='candle',volume=False,figratio=(5,5),figscale=1,mav = (1,6)
                    #,title = title
                    )
    #mpf.plot(data,**kwargs,style = s,savefig=r'data/temp_image.png')
    
    #mpf.plot(data,**kwargs,style = s,savefig='data/temp_image'+ str(index) +'.png')
    #mpf.plot(data,**kwargs,style = s,savefig='/ramdisk/temp_image'+ str(index) +'.png')

    if inRAM == True:
        mpf.plot(data,**kwargs,style = s,savefig='/ramdisk/temp_image'+ str(index) +'.png')
    elif inRAM == False:
        mpf.plot(data,**kwargs,style = s,savefig='data/temp_image'+ str(index) +'.png')


    #time.sleep(1)


def applyParallel_groupby(dfGrouped, func):
    """To be used when data is split using df.groupby() """
    with mp.Pool(processes = mp.cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    p.close()    
    #p.join()    
    return ret_list
    #return pd.concat(ret_list)

def applyParallel_npsplit(dfGrouped, func):
    """To be used when data is split using np.array_split """
    with mp.Pool(processes = mp.cpu_count()) as p:
        ret_list = p.map(func, dfGrouped)
    p.close()    
    return ret_list


def createXYarrays(group):

    """
    Function that will be called by multiprocessing 
    Separate process will spawned for each Symbol
    First attempt was creating set_x_sub as a list but later settled with single array containing both x and y ie set_xy
    """

    loop_range=  (group['Symbol'].count()) -  (DATE_WINDOW) - 10    
    #loop_range = 5
    symbolDate = group[-1:]['Date'].item()
    symbolDate = symbolDate.strftime('%Y%m%d')

    fname = str(group[-1:]['Symbol'].item()[0:3]) + str(symbolDate)
    #random_no = randint(1e10) #Random number for each CPU to be appended to the file name
    
#    print("the file name is" + fname)
#    print(group.shape)       
#    print('Loop range : ' + str(loop_range)) 


    set_xy = (np.empty(shape=(loop_range),dtype = 'uint8')
            ,np.empty(shape=(loop_range,IMG_SIZE,IMG_SIZE,3)))

    
    for i in range(loop_range):    
        
        if i % 100 == 0:
            print("Iter:" + str(i))    

        targetLabel = group[-1:]['Target'].item()
        set_xy[0][i] = targetLabel

        #print(set_xy[0][i])
        #Remove the last row and plot
        group = group[:-1]        
        

        ### Create temp file ON HARD DISK ## 
        #create_candlesticks(group[-DATE_WINDOW:],fname,inRAM=False)
        #img_asNumpy = np.array(Image.open('data/temp_image'+ fname + '.png').resize((IMG_SIZE,IMG_SIZE)))
        
        ### Create temp file ON RAM DISK ## 
        createCandlesticksPlot(group[-DATE_WINDOW:],fname
                            ,targetLabel = targetLabel
                            ,inRAM=True)
        img_asNumpy = np.array(Image.open('/ramdisk/temp_image'+ fname + '.png').resize((IMG_SIZE,IMG_SIZE)))
        
        #image_without_alpha 
        img_asNumpy = img_asNumpy[:,:,:3]
        
        set_xy[1][i] = img_asNumpy
    
    #out = {"set_y": set_y_sub,"set_x": set_x_sub}
    #return out
    #return [set_y_sub,set_x_sub]
    return set_xy

