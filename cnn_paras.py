import tensorflow as tf
import numpy as np

from util import _p


def param_init_encoder(filter_shape, params, prefix='cnn_encoder'):
	W_enc = tf.random_uniform(filter_shape, minval=-0.01, maxval=0.01)
	b_enc = tf.zeros([filter_shape[3]])

	params[_p(prefix,'W_enc')] = tf.Variable(W_enc)
	params[_p(prefix,'b_enc')] = tf.Variable(b_enc)

	return params


def param_init_residual(shape,params,prefix='ResNet'):
	G = tf.random_uniform(shape, minval=-0.01, maxval=0.01)
	c = tf.zeros([shape[1]])
	W = tf.random_uniform([shape[1],shape[0]], minval=-0.01, maxval=0.01)
	b = tf.zeros([shape[0]])

	params[_p(prefix,'G')] = tf.Variable(G)
	params[_p(prefix,'c')] = tf.Variable(c)
	params[_p(prefix,'W')] = tf.Variable(W)
	params[_p(prefix,'b')] = tf.Variable(b)

	return params


def cnn_encoder_layer(layer_input,param,prefix='cnn_encoder'):


	"""
	encode a layer either from original input or output from
	other layers.
	"""
	# emb_sz = layer_input.get_shape().as_list()[1]
	filter_sz = param[_p(prefix,'W_enc')].get_shape().as_list()[0]
	sen_lz = layer_input.get_shape().as_list()[1]

	conv_layer = tf.nn.conv2d(layer_input, param[_p(prefix,'W_enc')], strides=[1, 1, 1, 1], padding="VALID")
	conv_tanh_layer = tf.tanh(tf.nn.bias_add(conv_layer,param[_p(prefix,'b_enc')]))

	max_pool_layer = tf.nn.max_pool(conv_tanh_layer, ksize=[1,sen_lz - filter_sz+1,1,1], strides=[1,1,1,1], padding="VALID")
	feat_maps = max_pool_layer.get_shape().as_list()[3]
	max_pool_layer = tf.reshape(max_pool_layer,[-1,feat_maps])
	return max_pool_layer



def ResNet_Unit(Unit_inputX,Unit_inputZ,params,prefix='ResNet'):
	Hlayer = tf.sigmoid(tf.matmul(Unit_inputZ,params[_p(prefix, 'G')])+ params[_p(prefix, 'c')])
	Unit_outputX = tf.matmul(Hlayer,params[_p(prefix , 'W')]) + Unit_inputX
	Unit_outputZ = tf.sigmoid(Unit_outputX + params[_p(prefix, 'b')])

	return Unit_outputX,Unit_outputZ






