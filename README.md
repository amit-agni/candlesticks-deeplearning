# Stock price movement prediction - Deep Learning

The project aims at predicting the stock price movements using deep learning models build on candlestick chart images

### Progress so far
* Data Preparation
    + Used `mplfinance` package to create candlestick charts from OHLC CSV data 
    + Customisation done using `matplotlib` [rcParams](https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files)
    + Created RAMDisk for read/write of chart image files
    + Used `uint8` for `numpy` arrays and also for `create_dataset` in `h5py` .h5 file creation
    + Used `multiprocessing` pool function to spawn `mp.cpu_count()` simultaneous process

* Fully Connected Network Build
    + Increased image size from 64x64 to 191x192 (Lowered underfit)
    + Batch Normalisation (Lower overfit)
        + `,keras.layers.BatchNormalization()`
    + Dropout Regularisation (Lower overfit)
        + `,keras.layers.Dropout(0.5)`
        + rate = keep_probability, lower for higher overfit
    + Activation layers : `kernel_initializer=tf.keras.initializers.` (Lower overfit)
        + he_normal() for `tanh` activations
        + GlorotNormal() for `relu` activations also called Xavier normal initializer
        + Both have normal and uniform versions

* Enabled Single Point precision (FP16) for better performance
* Installed `[tensorboard]`(https://www.tensorflow.org/tensorboard/get_started)
* used hParams 


### Some Random Notes on Dropout worsens performance

* Furthermore, be careful where you use dropout. It is usually ineffective in the convolutional layers, and very harmful to use right before the softmax layer.
* Dropout is a regularization technique, and is most effective at preventing overfitting. However, there are several places when dropout can hurt performance.
    + Right before the last layer. This is generally a bad place to apply dropout, because the network has no ability to "correct" errors induced by dropout before the classification happens. If I read correctly, you might have put dropout right before the softmax in the iris MLP.
    + When the network is small relative to the dataset, regularization is usually unnecessary. If the model capacity is already low, lowering it further by adding regularization will hurt performance. I noticed most of your networks were relatively small and shallow.
    + When training time is limited. It's unclear if this is the case here, but if you don't train until convergence, dropout may give worse results. Usually dropout hurts performance at the start of training, but results in the final ''converged'' error being lower. Therefore, if you don't plan to train until convergence, you may not want to use dropout.
    + Finally, I want to mention that as far as I know, dropout is rarely used nowaways, having been supplanted by a technique known as batch normalization. Of course, that's not to say dropout isn't a valid and effective tool to try out.



### Misc Issues / Notes [To be compiled]
* The .h5 file sizes are huge 
    + Solved by setting dtypye to uint
    + `file.create_dataset('set_x', data=set_x,dtype='uint8')`
    + 5.4GB reduced to 700MB
    + So far no reduction in model performance (accuracy 60%)

* Also reduced the numpy array that is holding x and y from `int64` to `uint8`
    + `set_xy = (np.empty(shape=(loop_range),dtype = 'uint8')
            ,np.empty(shape=(loop_range,IMG_SIZE,IMG_SIZE,3)))`
    + CRASHED

* Other issues faced as mentioned above :
    + Had to use all the 6 cores using multiprocessing pool
    + Same random number was getting generated when the process was called. I have to pass seed, but couldnt get it to work as passing 3 variables in pool.map / pool.starmap was becoming a challenge
    + Solved by using the Date variable in the file name    
    + Earlier I used 64x64 images but the model seemed to stuck at 50% accuracy. Maybe more epochs would have helped ? not sure


* Tensorflow started giving memory error :
    + `Unable to allocate array with shape (156816, 36, 53806) and data type uint8`
    + Fixed after increasing page size.. not sure how
    + Alternate solution : `echo 1 > /proc/sys/vm/overcommit_memory` [NOT TRIED]
    + Source : https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type


* More memory issues :
    + `print(set_x.dtype)`  gives `uint8` as I have stored the images in that format as it also takes less space but any operation like subtraction and division, converts it to float32 and thus bloating up the space and hence the memory error
    + `print(set_x.nbytes/1024/1024)` gives  3.5GB
    + Solution is 
        + 1. Convert the array to float32 or float16 (but many methods expect float32)
        + Use keras.layers.BatchNormalization()
    + Side note : Types are changing to float due to tf.image.resize_images.


* Mixed precision
    + https://www.tensorflow.org/guide/mixed_precision
    + As my GPU is RTX2060 with compute capability of 7.5 I used it
        + `from tensorflow.keras.mixed_precision import experimental as mixed_precision`
        + `policy = mixed_precision.Policy('mixed_float16')`
        + `mixed_precision.set_policy(policy)`
    + Did not notice any impact on the performance

* Initially, the accuracy was stuck at 50%. Below were some of the things that helped in increasing it
    + The objective was to overfit the model i.e get the training accuracy to 99% 
    + Increased the shape of the images from 64x64 to 192x192. Tried other sizes like 
        + 128 (was better than 64)
        + 256 (Was increasing the array size)
        + Settled with 192x192 for now
    + The set_x/255. step was filling up the memory. So used Batch normalisation
    + Increased the training set to over 30K images (out of the available 60K + images)
        + The DATE_WINDOW reduced from 20 to 15

* Reducing Overfitting
    + kernel_regularizer=keras.regularizers.l2(0.001))
    + kernel_initializer='GlorotNormal'
        + Default is GlorotUniform (which is Xavier uniform)
        + According to this course by Andrew Ng and the Xavier documentation, if you are using ReLU as activation function, better change the default weights initializer(which is Xavier uniform) to Xavier normal by
        + Interesting papers linked here : https://stats.stackexchange.com/questions/339054/what-values-should-initial-weights-for-a-relu-network-be
        + If you dug a little bit deeper, you’ve likely also found out that one should use Xavier / Glorot initialization if the activation function is a Tanh, and that He initialization is the recommended one if the activation function is a ReLU. Source: https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
    + kernel_initializer=tf.keras.initializers.he_normal()
        +  I will use this


* After adding `keras.layers.BatchNormalization()` after every Relu layer, the accuracy was initially very low as compared to without BN after every layer. But it is gradually increasing
    + After removing BN layers, the val acc for the first epoch increased from 25% to like 49% ????????


* Check memory occupied by int and float dtypes
```import numpy as np
def calcArrayMemorySize(array):
    return "Memory size is : " + str(array.nbytes/1024/1024) + " Mb"
    
print(calcArrayMemorySize(np.random.randint(0,255,size=(100,64,64,3))))
print(calcArrayMemorySize(np.random.random(size=(100,64,64,3))))
print(calcArrayMemorySize(np.random.random_sample(size=(100,64,64,3))))

Something wrong 
#Memory size is : 9.375 Mb
#Memory size is : 9.375 Mb
#Memory size is : 9.375 Mb
```

* Normalising Error
    + `set_x = set_x.astype("float32") / 255`
    + MemoryError: Unable to allocate 24.2 GiB for an array with shape (58842, 192, 192, 3) and data type float32



13th Jul 2020
* Cross checked data as validation performance not increasing
* Data set increased from 20 to 80 stocks, with start year of 2000

17th Jul 2020
* 15 layer model is giving average performance
* Try class weights with reg
* Add average model accuracy (predicting everything in majority class)



18th Jul 20
* For using the F1 metrics from tfa (tensorflowaddons), y should be one hot encoded
    + `metrics=['accuracy',tfa.metrics.FBetaScore(num_classes=3, average="micro", threshold=None )]`
    + And use `loss=tf.keras.losses.CategoricalCrossentropy()` 
* Changed class weights using `counts.sum()/counts/2`