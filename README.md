# Stock price movement prediction - Deep Learning

The project aims at predicting the stock price movements using deep learning models build on candlestick chart images

### Progress so far
* Data Preparation
    + Use of `mplfinance` package to create candlestick charts from OHLC CSV data 
    + Customisation using `matplotlib` [rcParams](https://matplotlib.org/3.2.1/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files)
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
* Installed `tensorboard`


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


