import os
import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, preprocessing

#------------------------------------------------------------------------
def loadData(max_len):
	
	#Load data
	(Xtrain, ytrain), (Xtest, ytest) = datasets.imdb.load_data(num_words=n_words)
	worddict = datasets.imdb.get_word_index(path='imdb_word_index.json')

	#Pad sequences with max_len
	Xtrain = preprocessing.sequence.pad_sequences(Xtrain, maxlen=max_len)
	Xtest = preprocessing.sequence.pad_sequences(Xtest, maxlen=max_len)

	#Return
	return (Xtrain, ytrain), (Xtest, ytest), worddict

#------------------------------------------------------------------------
def decipherReview(review,worddict):

	wordlist = []
	for item in review:
		if (item > 0):
			wordlist.append(worddict[item])
	print(wordlist)
	
#------------------------------------------------------------------------
def buildModel(n_words,dim_embedding):

	#Build  Model
	model = models.Sequential()
	model.add(layers.Embedding(n_words, dim_embedding, input_length=max_len))
	model.add(layers.Dropout(0.3))
	model.add(layers.GlobalMaxPooling1D())
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation='sigmoid'))
	model.summary()

	#Return
	return model

#------------------------------------------------------------------------
if __name__ == '__main__':

	#Clear
	os.system('clear')
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	#Parameters
	max_len = 200
	n_words = 1000
	dim_embedding = 256
	EPOCHS = 5
	BATCH_SIZE = 10

	#Load Data
	print('loading data...')
	(Xtrain, ytrain), (Xtest, ytest), worddict = loadData(max_len)

	#Word Dictionary to Index Dictionary
	indexdict = {}
	for item in worddict:
		indexdict[worddict[item]] = item

	#Decipher random review
	index = random.randint(0,Xtrain.shape[0])
	#decipherReview(Xtrain[index,:],indexdict)

	#Build Model
	model = buildModel(n_words,dim_embedding)

	#Compile Model
	model.compile(optimizer = "adam", loss = "binary_crossentropy",metrics = ["accuracy"])

	#Fit
	history = model.fit(Xtrain, ytrain, epochs= EPOCHS, batch_size = BATCH_SIZE, validation_data = (Xtest, ytest))
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
	plt.savefig('src/python/imdb_sentiment_analysis/imdb_sentiment_analysis.png')

	#Evaluate
	score = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
	print("\nTest score:", score[0])
	print('Test accuracy:', score[1])
