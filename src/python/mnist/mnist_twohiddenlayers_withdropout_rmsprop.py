import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt

#------------------------------------------------------------------------
def loadMNIST():

	#Load
	mnist = keras.datasets.mnist
	(Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()
	print(Xtrain.shape[0], 'train samples')
	print(Xtest.shape[0], 'test samples')

	#Return
	return Xtrain,Ytrain,Xtest,Ytest

#------------------------------------------------------------------------
def normalize(Xtrain,Xtest):

	#Normalize
	Xtrain = Xtrain / 255.0
	Xtest = Xtest / 255.0

	#Return
	return Xtrain,Xtest

#------------------------------------------------------------------------
def flatten(Xtrain,Xtest):

	#Flatten
    flattened = Xtrain.shape[1] * Xtrain.shape[2]
    Xtrain = Xtrain.reshape(60000, flattened)
    Xtest = Xtest.reshape(10000, flattened)
    Xtrain = Xtrain.astype('float32')
    Xtest = Xtest.astype('float32')

	#Return
    return Xtrain,Xtest

#------------------------------------------------------------------------
def oneHotEncoding(Ytrain,Ytest,NB_CLASSES):
    
    Ytrain = tf.keras.utils.to_categorical(Ytrain, NB_CLASSES)
    Ytest = tf.keras.utils.to_categorical(Ytest, NB_CLASSES)

	#Return
    return Ytrain,Ytest

#------------------------------------------------------------------------
def buildModel(NB_CLASSES,N_HIDDEN,DROPOUT,inputSize):
    
    #Build Model
	model = tf.keras.models.Sequential()
	model.add(keras.layers.Dense(N_HIDDEN,input_shape=(inputSize,),name='dense_layer', activation='relu'))
	model.add(keras.layers.Dropout(DROPOUT))
	model.add(keras.layers.Dense(N_HIDDEN,name='dense_layer_2', activation='relu'))
	model.add(keras.layers.Dropout(DROPOUT))
	model.add(keras.layers.Dense(NB_CLASSES,name='dense_layer_3', activation='softmax'))
	model.summary()
    
    #Return
	return model

#------------------------------------------------------------------------
def compileModel(model):
    
    #Compile
    model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics=['accuracy'])

    #Return
    return model

#------------------------------------------------------------------------
if __name__ == '__main__':

    #Surpress warning
    os.system('clear')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #Network and training parameters
    EPOCHS = 50
    BATCH_SIZE = 128
    VERBOSE = 1
    NB_CLASSES = 10
    N_HIDDEN = 128
    VALIDATION_SPLIT=0.2
    DROPOUT = 0.3
    
	#For reproducibility
    np.random.seed(1671)

    #Loading MNIST dataset
    Xtrain,Ytrain,Xtest,Ytest = loadMNIST()
    
    #Reshape
    Xtrain,Xtest = flatten(Xtrain,Xtest)

    #Normalize
    Xtrain,Xtest = normalize(Xtrain,Xtest)

    #To One Hot
    Ytrain,Ytest = oneHotEncoding(Ytrain,Ytest,NB_CLASSES)

    #Build Model
    model = buildModel(NB_CLASSES,N_HIDDEN,DROPOUT,Xtrain.shape[1])

    #Compile Model
    model = compileModel(model)

    #Training the model
    history = model.fit(Xtrain, Ytrain,batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
    loss = history.history['loss']
    loss_val = history.history['val_loss']
    ac = history.history['accuracy']
    ac_val = history.history['val_accuracy']

    #Generate Figure
    plt.figure(figsize=(6,10))
    plt.subplot(2,1,1)
    plt.plot(ac,'g')
    plt.plot(ac_val,'r')
    plt.xlabel('epoch',fontsize=18)
    plt.ylabel('accuracy',fontsize=18)
    plt.grid()
    plt.subplot(2,1,2)
    plt.plot(loss,'g')
    plt.plot(loss_val,'r')
    plt.xlabel('epoch',fontsize=18)
    plt.ylabel('loss',fontsize=18)
    plt.grid()
    plt.savefig('src/python/mnist/mnist_twohiddenlayers_withdropout_rmsprop.png')

    #Evaluate the model
    testLoss, testAcc = model.evaluate(Xtest, Ytest)
    print('Test accuracy:', testAcc)

    #Making prediction
    predictions = model.predict(Xtest)