# -*- encoding:utf-8 -*-
import pandas as pd 
import numpy as np 
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense
from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers import concatenate
from keras.layers import MaxPooling1D
from keras.layers import GlobalMaxPooling1D
from keras.layers.convolutional import Convolution1D
from keras.layers.advanced_activations import PReLU


def ResBlock(x1, filterNum):
		x = Convolution1D(filterNum,3, padding='same')(x1)
		# x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Convolution1D(filterNum,3, padding='same')(x)
		# x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = concatenate([x, x1])
		return x

class VDCNN(object):
	"""docstring for CNN"""
	def __init__(self, sequence_len, nb_words, embedding_dim, embedding_matrix):
		super(VDCNN, self).__init__()
		self.sequence_len = sequence_len
		self.embedding_matrix = embedding_matrix
		self.embedding_dim = embedding_dim
		self.nb_words = nb_words

	def get_model(self):
		inputs_x = Input(shape=(self.sequence_len,), dtype='int32')
		embedded_sequences = Embedding(self.nb_words,self.embedding_dim,weights=[self.embedding_matrix],
			input_length=self.sequence_len,trainable=False)(inputs_x)
		conv_out = []
		# x = Convolution1D(128,3, padding='same')(embedded_sequences)
		x = ResBlock(embedded_sequences,256)
		x = MaxPooling1D(2)(x)
		x = ResBlock(x,256)
		x = GlobalMaxPooling1D()(x)
		x = Dropout(0.25)(x)
		x = Dense(256)(x) #128
		x = Activation('relu')(x)
		x = Dense(6)(x)

		outputs = Activation('sigmoid',name='outputs')(x)
		model = Model(inputs=[inputs_x], outputs=outputs)
		model.compile(loss='binary_crossentropy',optimizer= 'adamax')
		self.model = model
		return self.model

	def params(self):
		print 'Sequence  length: %d'%self.sequence_len
		print 'Word      Number: %d'%self.nb_words
		print 'Embedding    Dim: %d'%self.embedding_dim
	

