import tensorflow as tf
from tensorflow import keras

def createFullyConnected(structure):
    """Creates fully connected network
    """
    model = tf.keras.Sequential(name = structure['name'])
    model.add(tf.keras.layers.Flatten(input_shape=structure['inputShape']))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))

    for i in range(structure['layers']):        
        if('L1_regRate' in structure.keys()):
            model.add(tf.keras.layers.Dense(structure['units']
                    ,activation=structure['activation']
                    ,kernel_regularizer=tf.keras.regularizers.l1(structure['L1_regRate'])))
        elif('L2_regRate' in structure.keys()):
            model.add(tf.keras.layers.Dense(structure['units']
                    ,activation=structure['activation']
                    ,kernel_regularizer=tf.keras.regularizers.l2(structure['L2_regRate'])))
        else:
            model.add(tf.keras.layers.Dense(structure['units'],activation=structure['activation']))

        if('dropout' in structure.keys()):
                model.add(tf.keras.layers.Dropout(structure['dropout']))

    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    return model


def createCNN(structure):
    """Creates Convolutional Network
    """
    
    model = tf.keras.Sequential(name=structure['name'])
    model.add(keras.layers.Input(shape=structure['inputShape']))    

    if('batchnormalization' in structure.keys()):
                model.add(tf.keras.layers.BatchNormalization())            

    model.add(tf.keras.layers.BatchNormalization())            

    # model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))

    for f in structure['filters']:        
        for m in range(structure['convLayerMultiplier']):
            
            if('L1_regRate' in structure.keys()):
                model.add(tf.keras.layers.Conv2D(f,structure['kernelSize']
                        ,activation=structure['activation']
                        ,kernel_regularizer=tf.keras.regularizers.l1(structure['L1_regRate'])))
            elif('L2_regRate' in structure.keys()):
                model.add(tf.keras.layers.Conv2D(f,structure['kernelSize']
                    ,activation=structure['activation']
                    ,kernel_regularizer=tf.keras.regularizers.l2(structure['L2_regRate'])))
            else:
                model.add(tf.keras.layers.Conv2D(f,structure['kernelSize']
                            ,activation=structure['activation'],kernel_initializer='glorot_normal'
                            ,padding='same'))

            if('batchnormalization' in structure.keys()):
                model.add(tf.keras.layers.BatchNormalization())            
            
            if('poolingLayer' in structure.keys()):
                if structure['poolingLayer'] == 'AveragePooling2D':
                    model.add(tf.keras.layers.AveragePooling2D(pool_size = structure['poolSize']
                        ,padding = structure['padding']))
                elif structure['poolingLayer'] == 'MaxPooling2D':
                    model.add(tf.keras.layers.MaxPooling2D(pool_size = structure['poolSize']
                        ,padding = structure['padding']))
          
            if('dropout' in structure.keys()):
                model.add(tf.keras.layers.Dropout(structure['dropout']))


    model.add(tf.keras.layers.Flatten())    

    for i in range(structure['denseLayers']):        
        if('L1_regRate' in structure.keys()):
            model.add(tf.keras.layers.Dense(structure['units']
                    ,activation=structure['activation']                    
                    ,kernel_regularizer=tf.keras.regularizers.l1(structure['L1_regRate'])))
        elif('L2_regRate' in structure.keys()):
            model.add(tf.keras.layers.Dense(structure['units']
                    ,activation=structure['activation']
                    ,kernel_regularizer=tf.keras.regularizers.l2(structure['L2_regRate'])))
        else:
            model.add(tf.keras.layers.Dense(structure['units']
            ,activation=structure['activation'],kernel_initializer='glorot_normal'
            ))

        if('batchnormalization' in structure.keys()):
                model.add(tf.keras.layers.BatchNormalization())    
                
        if('dropout' in structure.keys()):
            model.add(tf.keras.layers.Dropout(structure['dropout']))
      

    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    
    return model

