import mplfinance as mpf
import numpy as np
import pandas as pd
import h5py
import time
import os
import multiprocessing as mp

from PIL import Image
from numpy.random import randint
from numpy.random import seed

DATE_WINDOW = 15
UP_THRESHOLD_PCT = 1
DOWN_THRESHOLD_PCT = 1
IMG_SIZE = 192

def calcArrayMemorySize(array):
    return "Memory size is : " + str(array.nbytes/1024/1024) + " Mb"
    

def setTargetLabel(val):
    if val > UP_THRESHOLD_PCT: out = 2
    elif val < -DOWN_THRESHOLD_PCT: out = 1
    else: out = 0
    return out

def createCandlesticksPlot(data,index,inRAM = True):  
    """
    Core function that creates the plot and saves to a file
    Argument:
    data -- to be done
    index -- to be done
    Returns:
    None -- to be done
    """      
    # To set a datetime as an index
    data = data.set_index(pd.DatetimeIndex(data['Date'])) 
   
    #Create custom styles
    mc = mpf.make_marketcolors(up='g',down='r')
    rc = {'xtick.major.bottom':False
        ,'ytick.major.left':False
        ,'xtick.major.size':0,'ytick.major.size':0
        ,'axes.labelsize' : 0
        ,'savefig.jpeg_quality' : 5
        ,'savefig.bbox':'tight'
        ,'patch.linewidth' : 0 #candle border
        ,'lines.linewidth' : 1.5 #wick width
        ,'axes.spines.left' :False #plot border
        ,'axes.spines.top' :False
        ,'axes.spines.bottom' :False
        ,'axes.spines.right' :False
        }
    s  = mpf.make_mpf_style(marketcolors=mc,rc = rc)
    
    # First we set the kwargs that we will use for all of these examples:
    kwargs = dict(type='candle',volume=False,figratio=(5,5),figscale=1)
    #mpf.plot(data,**kwargs,style = s,savefig=r'data/temp_image.png')
    
    #mpf.plot(data,**kwargs,style = s,savefig='data/temp_image'+ str(index) +'.png')
    #mpf.plot(data,**kwargs,style = s,savefig='/ramdisk/temp_image'+ str(index) +'.png')

    if inRAM == True:
        mpf.plot(data,**kwargs,style = s,savefig='/ramdisk/temp_image'+ str(index) +'.png')
    elif inRAM == False:
        mpf.plot(data,**kwargs,style = s,savefig='data/temp_image'+ str(index) +'.png')


    #time.sleep(1)


def saveXYtoDisk(result,fname):
    """
    Function that separates x and y
    And also created .h5 files to save the arrays
    """
    first = [x for (x,y) in result]
    set_y = np.concatenate(first,axis =0)

    second = [y for (x,y) in result]
    set_x = np.concatenate(second,axis =0)

    print(fname)

    file = h5py.File("../data/" + fname + ".h5", "w")
    file.create_dataset('set_x', data=set_x,dtype='uint8')
    file.create_dataset('set_y', data=set_y,dtype='uint8')
    file.close()


def readXYfromDisk():
    """
    # Reads .h5 files
    # Appends them to a list
    # Finally converts the list to np
    """

    set_x = []
    set_y = []
    
    for file in os.listdir("../data/"):
        if file.endswith(".h5"):
            fname = os.path.join("../data", file)
            file = h5py.File(fname, "r")
            set_x_temp = file["set_x"][:]
            set_y_temp = file["set_y"][:]

            set_x.append(set_x_temp)
            set_y.append(set_y_temp)

            file.close()

    # Since set_x and set_y are list of arrays, use np.concatenate()
    # Better than result_array = np.array(result) ?

    set_x = np.concatenate(set_x,axis=0)
    set_y = np.concatenate(set_y,axis=0)

    return set_x,set_y




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

        set_xy[0][i] = group[-1:]['Target'].item()
        #print(set_xy[0][i])
        #Remove the last row and plot
        group = group[:-1]        
        

        ### Create temp file ON HARD DISK ## 
        #create_candlesticks(group[-DATE_WINDOW:],fname,inRAM=False)
        #img_asNumpy = np.array(Image.open('data/temp_image'+ fname + '.png').resize((IMG_SIZE,IMG_SIZE)))
        
        ### Create temp file ON RAM DISK ## 
        createCandlesticksPlot(group[-DATE_WINDOW:],fname,inRAM=True)
        img_asNumpy = np.array(Image.open('/ramdisk/temp_image'+ fname + '.png').resize((IMG_SIZE,IMG_SIZE)))
        
        #image_without_alpha 
        img_asNumpy = img_asNumpy[:,:,:3]
        
        set_xy[1][i] = img_asNumpy
    
    #out = {"set_y": set_y_sub,"set_x": set_x_sub}
    #return out
    #return [set_y_sub,set_x_sub]
    return set_xy

