import numpy as np
import tensorflow as tf
import logging
import time
import cPickle
import os
import argparse
from datetime import datetime
from collections import OrderedDict


from model_lstm import Model

from util import get_minibatches_idx, _p

logger = logging.getLogger('multi-zsl')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)


class Config:
	embedding_size = 300
	mhidden = 300
	rnn_hsize = 300
	rnncell = "GRU"
	patience = 3
	dropout_rate = 0.5
	beta = 2
	max_epochs = 30
	learn_rate = 0.0002
	batch_size = 20
	valid_size = 10
	dispFreq = 50
	
	# valid_batch_size = 10

	def __init__(self,ConfigInfo, data_type):
		self.n_words_sm = ConfigInfo['vocab_size']  # 38564 for clean data_64, 39367 for rpl data_64
		self.max_len_sm = ConfigInfo['max_len_sm']	 # 10 for clean data_64, 9 for rpl
		self.output_path = "./result_lm/" + data_type + "/{:%Y%m%d_%H%M%S}/".format(datetime.now())		
		self.model_path = self.output_path + "model.weights"



def init_parameters(Config,W):

	# W is initialized by the pretrained word embedding
	# otherwise, W will be initialized by random word embedding
	params = OrderedDict()
	params['Wemb'] = tf.Variable(W,trainable=False)
	# params['Wemb'] = tf.Variable(tf.random_uniform((W.shape), minval=-0.01, maxval=0.01))

	return params


def prepare_data(batch,Config):

	m_l = Config.max_len_sm
	n_w = Config.n_words_sm

	# f_h = Config.filters[-1]

	new_seqs = []
	mask_seqs = []
	for sen_input in batch:
		sen_mask = [True] * len(sen_input)

		# padding to reach a uniform length
		while len(sen_input) < m_l:  # + 2 * (f_h - 1):
			sen_input.append(n_w - 1)
			sen_mask.append(False)


		new_seqs.append(sen_input)
		mask_seqs.append(sen_mask)

	return new_seqs,mask_seqs



class ResCNNModel():

	def add_placeholders(self):
		sm_lz = self.Config.max_len_sm

		# sm_lz = self.Config.max_len_sm + 2 * (Config.filters[-1] - 1)
		self.seman_placeholder = tf.placeholder(dtype=tf.int32, shape= [None,sm_lz])
		self.mask_placeholder = tf.placeholder(dtype=tf.bool, shape= [None,sm_lz])
		self.dropout_placeholder = tf.placeholder(dtype=tf.float32)


	def create_feed_dict(self, seman, labelsR=None, mask=None, dropout_rate=1):
		feed_dict = {self.seman_placeholder:seman}
		feed_dict[self.dropout_placeholder] = dropout_rate
		if mask is not None:
			feed_dict[self.mask_placeholder] = mask

		return feed_dict


	def add_embedding(self,inputs):
		# emb_sz = self.Config.embedding_size
		# s_lz = self.Config.max_len + 2 * (Config.filters[-1] - 1)
		# setzero = tf.concat([tf.ones([self.Config.n_words-1,emb_sz]),tf.zeros([1,emb_sz])], axis=0)

		params = self.params
		x_emb = tf.nn.embedding_lookup(params['Wemb'],inputs)  # transform sentences into embeddings
		layer_input = tf.expand_dims(x_emb,-1)  	# transform batch of embedded sentences into convolutional form

		return layer_input



	def add_prediction_rnn(self,scope=None):
		dec_num = self.Config.n_words_sm
		mx_len = self.Config.max_len_sm
		h_size = self.Config.rnn_hsize
		scope = scope or type(self).__name__

		if self.Config.rnncell == "GRU":
			cell = tf.contrib.rnn.GRUCell(h_size)
		elif self.Config.rnncell == "LSTM":
			cell = tf.contrib.rnn.LSTMCell(h_size) 


		with tf.variable_scope(scope):
			O_h = tf.get_variable(name="O", shape=[h_size,dec_num], initializer= tf.contrib.layers.xavier_initializer())
			b_2 = tf.get_variable(name="b_2", shape=[dec_num,], initializer= tf.contrib.layers.xavier_initializer())

		with tf.variable_scope(scope):
			x = self.seman_placeholder
			x = self.add_embedding(x)
			x = tf.reshape(x,[-1,mx_len,self.Config.embedding_size])
			h_0 = x * 0
			curr_state = h_0[:,0,:]

			# import pdb;pdb.set_trace()

			states = []
			preds_rnn = []  # used for calculating the loss
			# preds_rnn_oh = []  # used for presenting the decoding result
			for time_step in range(mx_len):
				# import pdb; pdb.set_trace()
				if isinstance(cell.state_size,tuple):
					c_0 = tf.zeros([cell.state_size[0]])
					curr_state = (c_0,h_0)
				output,next_state = cell(x[:,time_step,:],curr_state)
				tf.get_variable_scope().reuse_variables()
				O_h = tf.get_variable(name="O")
				b_2 = tf.get_variable(name="b_2")
				pred_rnn = tf.matmul(output,O_h) + b_2

				# do smoe reshaping
				preds_rnn.append(pred_rnn)
				states.append(next_state)

				curr_state = next_state

			preds_rnn = tf.reshape(tf.concat(preds_rnn,axis=1),shape=[-1,mx_len,dec_num])

			return preds_rnn,states


	def add_loss_op(self,pred_rnn):  # mapping, seman

		loss_rnn_unmasked = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.seman_placeholder, logits=pred_rnn)
		loss_rnn = tf.reduce_mean(tf.boolean_mask(loss_rnn_unmasked,self.mask_placeholder))
		return loss_rnn


	def add_training_op(self,loss):
		lr = self.Config.learn_rate
		train_op = tf.train.AdamOptimizer(lr).minimize(loss)
		return train_op



	def train_on_batch(self, sess, seman, mask):  #
		feed_dict = self.create_feed_dict(seman=seman, mask=mask, dropout_rate=self.Config.dropout_rate)
		_,loss = sess.run([self.train_op,self.loss],feed_dict=feed_dict)

		return loss


	def predict_on_batch(self, sess, seman, mask):
		feed = self.create_feed_dict(seman=seman, mask=mask)
		rnn_encodings,pred_rnn = sess.run([self.rnn_encodings,self.pred_rnn], feed_dict=feed)

		return rnn_encodings,pred_rnn


	def run_epoch(self,sess,train_examples):
		iterator = get_minibatches_idx(len(train_examples),self.Config.batch_size,False)
		ce_score = 0
		dispFreq = self.Config.dispFreq
		# word_embeddings_old = sess.run(self.params['Wemb'])

		for i,idx in enumerate(iterator):
			sm = train_examples

			sm_batch = [sm[j] for j in idx]
			padded_sm_ipnuts, mask = prepare_data(sm_batch,self.Config)
			padded_sm_ipnuts = np.array(padded_sm_ipnuts)

			loss = self.train_on_batch(sess, padded_sm_ipnuts, mask)

			# print something about the loss
			if i % dispFreq == 0:
				logger.info("loss until batch_%d, : %f", i,loss)

			ce_score = ce_score + padded_sm_ipnuts.shape[0] * loss

		ce_score = ce_score / len(train_examples)
		logger.info("Evaluating on training data")
		logger.info("new updated AUC scores %.4f", ce_score)

		return ce_score


	def evaluate(self,sess,train_examples):
		iterator = get_minibatches_idx(len(train_examples),self.Config.batch_size,False)
		# word_embeddings_old = sess.run(self.params['Wemb'])

		masks = []
		rnn_encodings = []
		for i,idx in enumerate(iterator):
			sm = train_examples

			sm_batch = [sm[j] for j in idx]
			padded_sm_ipnuts, mask = prepare_data(sm_batch,self.Config)
			padded_sm_ipnuts = np.array(padded_sm_ipnuts)
			mask = np.array(mask)

			rnn_encoding,_ = self.predict_on_batch(sess,padded_sm_ipnuts,mask)

			rnn_encodings.append(rnn_encoding)
			masks.append(mask)


		rnn_encodings = np.concatenate(rnn_encodings,axis=0)
		masks = np.concatenate(masks,axis=0)

		label_embeddings = [rnn_encodings[i,:,:] for i in range(masks.shape[0])]

		lb_emd = []
		for i,emb in enumerate(label_embeddings):
			hstates = [emb[j,:] for j in range(emb.shape[0]) if mask[i,j]]
			lb_emd.append(hstates[-1])

		lb_emd = np.concatenate(lb_emd,axis=0)


		return lb_emd


	def __init__(self,Config,pretrained_embedding):
		self.Config = Config
		self.pretrained_embedding = tf.cast(pretrained_embedding, tf.float32)
		self.params = init_parameters(Config,self.pretrained_embedding)
		self.add_placeholders()
		self.pred_rnn,self.rnn_encodings = self.add_prediction_rnn()
		self.loss = self.add_loss_op(self.pred_rnn)
		self.train_op = self.add_training_op(self.loss)
		


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-lf','--label_freq', default='500', type=str)
	parser.add_argument('-ug','--using_glove', default=False, type=bool)
	parser.add_argument('-dt','--data',default='clean',type=str)

	args = parser.parse_args()

	# https://docs.python.org/2/howto/logging-cookbook.html
	logger = logging.getLogger('eval_tok64_cnn_res4')
	logger.setLevel(logging.INFO)
	fh = logging.FileHandler('eval_tok64_cnn_res4.log')
	fh.setLevel(logging.INFO)
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)
	logger.addHandler(fh)

	affx = ''
	if args.data == 'clean':
		affx = '_new'

	logger.info('loading data...')
	x = cPickle.load(open("./data/everything_lm" + affx + args.label_freq + ".p","rb"))
	train,test, W_g, W_m, i2w_lb, i2w_sm, ConfigInfo = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
	del x


	# whether use the glove data
	data_type = ''
	if args.using_glove:
		data_type = 'glove_'
		W = W_g
	else:
		data_type = 'mixed_'
		W = W_m

	n_classes = len(i2w_lb)
	n_words_sm = len(i2w_sm)
	config = Config(ConfigInfo, data_type)

	pred_acc = []
	ce_min = 10000

	GPU_config = tf.ConfigProto()
	GPU_config.gpu_options.per_process_gpu_memory_fraction = 0.2
	with tf.Graph().as_default():
		tf.set_random_seed(1234)
		logger.info("Building model")
		start = time.time()
		model = ResCNNModel(config, W)

		logger.info("time to build the model: %d", time.time() - start)
		logger.info("the output path: %s", model.Config.output_path)

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		if not os.path.exists(model.Config.output_path):
			os.makedirs(model.Config.output_path)	

		# if not os.path.exists(model.Config.output_path_results):
		#	os.makedirs(model.Config.output_path_results)	

		with tf.Session(config=GPU_config) as session:
			path = ''
			session.run(init)

			path = model.Config.model_path
			for epoch in range(Config.max_epochs):
				logger.info("running epoch %d", epoch)
				pred_acc.append(model.run_epoch(session,train))
				# import pdb; pdb.set_trace()
				if pred_acc[-1] < ce_min:
					logger.info("new best AUC score: %.4f", pred_acc[-1])
					ce_min = pred_acc[-1]
					saver.save(session, path)
				logger.info("BEST AUC SCORE: %.4f", ce_min)
			saver.restore(session, path)
			label_embeddings = model.evaluate(session,test)
			# make description of the configuration and test result
			with open(model.Config.output_path + "description.txt","w") as f:
				cfg = model.Config
				f.write("data_type: %s\nrnncel: %s\nlearn_rate: %f\nbatch_size: %d\n" % (args.data,cfg.rnncell,cfg.learn_rate,cfg.batch_size))
			f.close()

			# save all the results
			cPickle.dump(label_embeddings,open(model.Config.output_path + 'lb_emd_' + str(n_classes) + '.p',"wb"))
			