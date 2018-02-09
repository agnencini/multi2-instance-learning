import tensorflow as tf
import keras.backend as K
import keras
from bag_layer import *
from keras.layers import Conv2D, Input, Dense, MaxPooling2D, Flatten, Lambda, Dropout
from keras.models import Model, Sequential
from keras.utils import to_categorical

# Return a network model for multi-multi (2 levels of nesting) instance
# learning. Top-bags (images) are split in macropatches (sub-bags) and
# micropatches (instances).
# input_shape_ = (#big-patches, #small-patches, #pixels)
def get_model(input_shape_):

	model = Sequential()
	model.add(BagLayer(input_shape_[2], 150, input_shape=input_shape_))
	model.add(BagLayer(150, 300))

	model.add(Dense(1024, activation='relu', name='dense1'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax', name='out'))

	return model

# Return a network model for multi-multi-multi (3 levels of nesting) instance
# learning. Image are split in big, medium and small patches.
# input_shape_ = (#big-patches, #medium-patches, #small-patches, #pixels)
def get_multi3_model(input_shape_):

	model = Sequential()

	model.add(BagLayer(input_shape_[3], 100, input_shape=input_shape_))
	model.add(BagLayer(100, 200))
	model.add(BagLayer(200, 200))

	model.add(Dense(1024, activation='relu', name='dense1'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax', name='out'))

	return model

def lambda1(x):
	splits = []
	for i in range(80):
		for j in range(25):
			split = x[:, i, j]
			a = K.reshape(split, (-1,) + (5, 5,1))
			splits.append(a)
	return splits

def lambda2(convolved):
	outs = []
	for x in convolved:
		out = K.flatten(x)
		outs.append(out)
	return outs

def lambda3(outs):
	block = K.stack(outs)
	bps = []
	for i in range(80):
		bps.append(block[i * 25: (i+1)*25])
	return bps

def lambda4(bps):
	recomposed = K.stack(bps)
	recomposed = K.reshape(recomposed, (-1,) + (80, 25, 144))
	return recomposed

def get_conv_model(input_shape_):

	input_ = Input(shape=input_shape_)
	splits = Lambda(lambda1)(input_)
	conv1 = Conv2D(16, (3,3), padding='valid', activation='relu', input_shape=(5,5))

	convolved = []
	for s in splits:
		a = conv1(s)
		convolved.append(a)

	outs = Lambda(lambda2)(convolved)
	bps = Lambda(lambda3)(outs)
	recomposed = Lambda(lambda4)(bps)

	b1 = BagLayer(144, 100)(recomposed)
	b2 = BagLayer(100, 200)(b1)

	dense1 = Dense(1024, activation='relu')(b2)
	dropout = keras.layers.Dropout(0.5)(dense1)
	out1 = Dense(2, activation='softmax')(dropout)

	model = keras.models.Model(inputs=input_, outputs=out1)
	return model
