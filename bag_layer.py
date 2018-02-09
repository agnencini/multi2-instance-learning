import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.engine.topology import Layer

# A Keras layer that implements a bag layer, as defined in "A Network
# Architecture for Multi-Multi-Instance Learning" [Tibo, Frasconi, Jaeger].
class BagLayer(Layer):

	def __init__(self, input_dim, output_dim, **kwargs):
		self.input_dim = input_dim
		self.output_dim = output_dim
		super(BagLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		self.kernel = self.add_weight(name='kernel', shape=(self.input_dim, self.output_dim), initializer='uniform', trainable=True)
		self.bias = self.add_weight(name='bias', shape=(1, self.output_dim), initializer='uniform', trainable=True)

	# Here is where the bag layer logic lives. The layer takes as input a bag
	# of vectors of length n, then computes the product of each vector with a
	# weight matrix W(n,k), add biases and activation (ReLU). Then computes the
	# maximum component-wise of the resulted vectors. So the output is a single
	# vector of size k.
	def call(self, x):
		size_x = len(x.get_shape())
		feat_dim = x.get_shape()[-1].value
		nested_dim = [d.value for d in x.get_shape()[1:-1]]

		w1 = self.kernel.get_shape()[0].value
		w2 = self.kernel.get_shape()[1].value

		xxx = tf.reshape(x, [-1, feat_dim])
		xxx = tf.matmul(xxx, self.kernel) + self.bias
		xxx = tf.reshape(xxx, [-1] + nested_dim + [w2])
		xxx = tf.reduce_max(xxx, axis=-2)
		xxx = tf.nn.relu(xxx)

		return xxx

	# Return the output shape to allow Keras to perform shape inference
	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_dim)
