import numpy as np
import tensorflow as tf
import logging
import time
import cPickle
from datetime import datetime
from collections import OrderedDict

from sklearn.metrics import roc_auc_score

from model import Model
from cnn_paras import param_init_encoder
from cnn_paras import param_init_residual
from cnn_paras import cnn_encoder_layer
from cnn_paras import ResNet_Unit

from util import get_minibatches_idx, _p

logger = logging.getLogger('multi-zsl')
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.DEBUG)


class Config:
	embedding_size = 300
	feature_maps = 100
	filters = [3,4,5]
	filters_sm = [2,3,4]
	res_depth = 4
	mhidden = 300
	patience = 3
	dropout_rate = 0.5
	beta1 = 1
	beta2 = 0.1
	max_epochs = 50
	learn_rate = 0.0002
	batch_size = 50
	valid_size = 10
	dispFreq = 50

	# valid_batch_size = 10

	def __init__(self,ConfigInfo,n_label,nl_cls):
		self.nlabels = n_label
		self.nhidden = np.round(n_label * 3000 / (3000 + n_label))
		self.n_words = ConfigInfo['vocab_size']  # 38564 for clean data_64, 39367 for rpl data_64
		self.max_len = ConfigInfo['max_len']	 # 580 for clean data_64, 575 for rpl
		self.max_len_sm = ConfigInfo['max_len_sm']	 # 10 for clean data_64, 9 for rpl
		self.classes_in_nl = nl_cls
		self.output_path = "./result_new/{:%Y%m%d_%H%M%S}/".format(datetime.now())
		self.model_path = self.output_path + "model.weights"

	"""
	img_h = max_len + 2*(filter_hs[-1]-1)
   
    options = {}
    options['n_words'] = n_words
    options['img_w'] = img_w
    options['img_h'] = img_h
    options['feature_maps'] = feature_maps
    options['filter_hs'] = filter_hs
    options['patience'] = patience
    options['max_epochs'] = max_epochs
    options['lrate'] = lrate
    options['batch_size'] = batch_size
    options['valid_batch_size'] = valid_batch_size
    options['dispFreq'] = dispFreq
    options['validFreq'] = validFreq
    options['Sigma'] = Sigma
	"""


def init_parameters(Config,W):

	# W is initialized by the pretrained word embedding
	params = OrderedDict()
	params['Wemb'] = tf.Variable(tf.random_uniform(W.shape,minval=-0.01,maxval=0.01), trainable=False)
	# params['Wemb'] = tf.Variable(W,trainable=False)

	# otherwise, W will be initialized by random word embedding
	emb_sz = Config.embedding_size
	mhidden = Config.mhidden
	n_ft_map = Config.feature_maps
	filter_hs = len(Config.filters)
	filter_shape = []
	filter_shape_sm = []
	ResNet_shape = (Config.nlabels, Config.nhidden)
	trans_shape = (filter_hs*Config.feature_maps, Config.nlabels)

	for i in range(filter_hs):
		filter_shape.append((Config.filters[i],Config.embedding_size,1,n_ft_map))
		filter_shape_sm.append((Config.filters_sm[i],Config.embedding_size,1,n_ft_map))

	for idx in range(filter_hs):
		params = param_init_encoder(filter_shape[idx],params,prefix=_p('cnn_encoder', idx))
		params = param_init_encoder(filter_shape_sm[idx],params,prefix=_p('cnn_encoder_sm', idx))

	for idx in range(1,Config.res_depth+1):
		params = param_init_residual(ResNet_shape,params,prefix=_p('ResNet', idx))

	params['ResNet_0_W'] = tf.Variable(tf.random_uniform(trans_shape, minval=-0.01, maxval=0.01))
	params['ResNet_0_b'] = tf.Variable(tf.zeros(ResNet_shape[0]))

	params['W1_n2s'] = tf.Variable(tf.random_uniform((emb_sz,mhidden),minval=-0.01,maxval=0.01))
	params['b1_n2s'] = tf.Variable(tf.zeros(mhidden))
	params['W2_n2s'] = tf.Variable(tf.random_uniform((mhidden,emb_sz),minval=-0.01,maxval=0.01))
	params['b2_n2s'] = tf.Variable(tf.zeros(emb_sz))

	return params


def prepare_data(batch,Config,is_seman=False):
	n_w = Config.n_words
	m_l = Config.max_len
	if is_seman:
		m_l = Config.max_len_sm

	f_h = Config.filters[-1]

	new_seqs = []
	for sen in batch:
		# sen = [n_w - 1] + sen    # padding for the beginning of the sentence

		# padding to reach a uniform length

		while len(sen) < m_l:  # + 2 * (f_h - 1):
			sen.append(n_w - 1)
		new_seqs.append(sen)

	return new_seqs


class ResCNNModel(Model):

	def add_placeholders(self):
		s_lz = self.Config.max_len  # + 2 * (Config.filters[-1] - 1)
		n_lbls = self.Config.nlabels
		sm_lz = self.Config.max_len_sm

		# sm_lz = self.Config.max_len_sm + 2 * (Config.filters[-1] - 1)

		self.input_placeholder = tf.placeholder(dtype=tf.int32, shape= [None,s_lz])
		self.labels_placeholder = tf.placeholder(dtype=tf.float32, shape= [None,n_lbls])
		self.seman_placeholder = tf.placeholder(dtype=tf.int32, shape= [None,sm_lz])
		self.dropout_placeholder = tf.placeholder(dtype=tf.float32)
		# self.input_placeholder_sm = tf.placeholder(dtype=tf.float32, shape= [None,sm_lz])


	def create_feed_dict(self, inputs_batch, seman_batch=None, labels_batch=None, dropout_rate=1):
		feed_dict = {self.input_placeholder:inputs_batch}
		if labels_batch is not None:
			feed_dict[self.labels_placeholder] = labels_batch
		if seman_batch is not None:
			feed_dict[self.seman_placeholder] = seman_batch
		feed_dict[self.dropout_placeholder] = dropout_rate

		return feed_dict

	def add_embedding(self,is_seman = False):
		# emb_sz = self.Config.embedding_size
		# s_lz = self.Config.max_len + 2 * (Config.filters[-1] - 1)
		# setzero = tf.concat([tf.ones([self.Config.n_words-1,emb_sz]),tf.zeros([1,emb_sz])], axis=0)

		params = self.params
		Input = self.input_placeholder
		if is_seman:
			Input = self.seman_placeholder
		x_emb = tf.nn.embedding_lookup(params['Wemb'],Input)  # transform sentences into embeddings
		layer_input = tf.expand_dims(x_emb,-1)  	# transform batch of embedded sentences into convolutional form

		return layer_input


	def cnn_enc(self,is_seman= False):

			filter_hs = self.Config.filters
			params = self.params
			cnn_input = self.add_embedding(is_seman)
			# conv_input = tf.nn.dropout(layer_input_dropout, self.dropout_placeholder)

			prefix = 'cnn_encoder'
			if is_seman:
				prefix = 'cnn_encoder_sm'
			# CNN encoder
			conv_output = []
			for idx in range(len(filter_hs)):
				conv_layer = cnn_encoder_layer(cnn_input,params,prefix=_p(prefix, idx))
				conv_output.append(conv_layer)
			Encoded = tf.concat(conv_output,1)
			Encoded_dropout = tf.nn.dropout(Encoded, self.dropout_placeholder)

			return Encoded_dropout

	def add_mapping_op(self):

		params = self.params
		encoded_notes = self.cnn_enc()
		hidden_layer = tf.nn.relu(tf.matmul(encoded_notes,params['W1_n2s']) + params['b1_n2s'])
		notes2seman_layer = tf.nn.relu(tf.matmul(hidden_layer,params['W2_n2s']) + params['b2_n2s'])
		regularizer = tf.nn.l2_loss(params['W1_n2s']) + tf.nn.l2_loss(params['W1_n2s'])
		return notes2seman_layer,regularizer

	'''
	def add_prediction_train(self):  #,is_seman=False):

		# n_ft_map = self.Config.feature_maps
		# sen_lz = self.Config.max_len
		emb_sz = self.Config.embedding_size
		n_lb = self.Config.nlabels

		# params = self.params

		# with mapping structure


		if is_seman:
			encoded = self.cnn_enc(is_seman)
		else:
			encoded, _ = self.add_mapping_op()
		
		encoded = self.cnn_enc(True)

		with tf.variable_scope('predicting'):
			W_pred = tf.get_variable('W_pred',shape=(emb_sz,n_lb), initializer=tf.random_uniform_initializer(minval=-0.01,maxval=0.01))
			b_pred = tf.get_variable('b_pred',shape=(n_lb), initializer=tf.zeros_initializer())

			prediction = tf.matmul(encoded,W_pred)
			prediction_dropout = tf.sigmoid(tf.nn.bias_add(prediction,b_pred))

		
		prediction = tf.matmul(encoded,params['ResNet_0_W'])
		prediction_dropout = tf.sigmoid(tf.nn.bias_add(prediction,params['ResNet_0_b']))
		

		# Residual Network

		
		Reslayer1_X,Reslayer1_Z = ResNet_Unit(ResInput_X,ResInput_Z,params,prefix='ResNet_1')
		Reslayer2_X,Reslayer2_Z = ResNet_Unit(Reslayer1_X,Reslayer1_Z,params,prefix='ResNet_2')
		Reslayer3_X,Reslayer3_Z = ResNet_Unit(Reslayer2_X,Reslayer2_Z,params,prefix='ResNet_3')
		Reslayer4_X,Reslayer4_Z = ResNet_Unit(Reslayer3_X,Reslayer3_Z,params,prefix='ResNet_4')
		

		# pred = Reslayer4_Z
		pred = prediction_dropout
		assert pred.get_shape().as_list() == [None, self.Config.nlabels]

		return pred
	'''

	def add_loss_op(self,mapping,seman):  # ,pred):  # mapping,seman,
		loss_1 = tf.nn.l2_loss(mapping - seman)/self.Config.batch_size * 2 + self.regularizer * self.Config.beta1
		# loss_2 = - tf.reduce_mean(self.labels_placeholder * tf.log(pred) + (1 - self.labels_placeholder) * tf.log(1 - pred))
		loss = loss_1  # * self.Config.beta2 + loss_2
		return loss


	def add_training_op(self,loss):
		lr = self.Config.learn_rate
		train_op = tf.train.AdamOptimizer(lr).minimize(loss)
		return train_op


	def train_on_batch(self,sess,inputs_batch,seman_batch,labels_batch):  #
		feed_dict = self.create_feed_dict(inputs_batch=inputs_batch,seman_batch=seman_batch,labels_batch=labels_batch,dropout_rate=self.Config.dropout_rate)
		_, loss = sess.run([self.train_op, self.loss],feed_dict=feed_dict)

		return loss


	def add_prediction_test(self):  # ,is_seman=False):

		emb_sz = self.Config.embedding_size
		n_lb = self.Config.nlabels
		bch_sz = self.Config.batch_size

		encoded_o,_ = self.add_mapping_op()
		encoded_o = tf.reshape(tf.tile(encoded_o,[1,n_lb]),[-1,emb_sz,n_lb])

		standard_o = self.cnn_enc(True)
		standard_o = tf.tile(standard_o,[1,bch_sz])
		standard_o = tf.reshape(tf.transpose(standard_o),[-1, emb_sz, n_lb])

		assert encoded_o.get_shape().as_list() == standard_o.get_shape().as_list()

		prediction = tf.reduce_mean(tf.square(encoded_o - standard_o),axis=1)
		prediction_dropout = tf.sigmoid(prediction)


		# Residual Network

		'''
		Reslayer1_X,Reslayer1_Z = ResNet_Unit(ResInput_X,ResInput_Z,params,prefix='ResNet_1')
		Reslayer2_X,Reslayer2_Z = ResNet_Unit(Reslayer1_X,Reslayer1_Z,params,prefix='ResNet_2')
		Reslayer3_X,Reslayer3_Z = ResNet_Unit(Reslayer2_X,Reslayer2_Z,params,prefix='ResNet_3')
		Reslayer4_X,Reslayer4_Z = ResNet_Unit(Reslayer3_X,Reslayer3_Z,params,prefix='ResNet_4')
		'''

		# pred = Reslayer4_Z
		pred = prediction_dropout
		assert pred.get_shape().as_list() == [None, self.Config.nlabels]

		return pred


	def predict_on_batch(self, sess, inputs_batch,seman_batch):
		feed = self.create_feed_dict(inputs_batch=inputs_batch,seman_batch=seman_batch,dropout_rate=0)
		predictions = sess.run(self.pred_test, feed_dict=feed)

		return predictions




	def evaluate(self,sess,examples):
		iterator = get_minibatches_idx(len(examples[0]),self.Config.valid_size,True)	
		preds = []
		labels = []

		for i,idx in enumerate(iterator):
			nts,lb,sm = examples

			nts_batch = [nts[i] for i in idx]
			lb_batch = [lb[j] for j in idx]

			padded_nts_batch = prepare_data(nts_batch,self.Config,False)
			padded_nts_batch = np.array(padded_nts_batch)
			padded_nl_cls = prepare_data(self.Config.classes_in_nl,self.Config,True)
			padded_nl_cls = np.array(padded_nl_cls) 

			lb_batch = np.array(lb_batch)

			pred = self.predict_on_batch(sess,padded_nts_batch,padded_nl_cls)
			preds.append(pred.flatten())
			labels.append(lb_batch.flatten())

		preds = np.concatenate(preds,axis=0)
		labels = np.concatenate(labels,axis=0)

		auc = roc_auc_score(labels,preds)
		error = 1. - auc

		return error


	def run_epoch(self,sess,train_examples,dev_set):
		iterator = get_minibatches_idx(len(train_examples[0]),self.Config.batch_size,True)
		auc_score = 0
		dispFreq = self.Config.dispFreq

		for i,idx in enumerate(iterator):
			nts,lb,sm = train_examples

			nts_batch = [nts[j] for j in idx]
			lb_batch = [lb[j] for j in idx]
			sm_batch = [sm[j] for j in idx]

			padded_nts_batch = prepare_data(nts_batch,self.Config,False)
			padded_nts_batch = np.array(padded_nts_batch)
			padded_sm_batch = prepare_data(sm_batch,self.Config,True)
			padded_sm_batch = np.array(padded_sm_batch)
			lb_batch = np.array(lb_batch)

			loss = self.train_on_batch(sess,padded_nts_batch,padded_sm_batch,lb_batch)

			# print something about the loss
			if i % dispFreq == 0:
				logger.info("loss until batch_%d, : %f", i,loss)

		logger.info("Evaluating on devlopment data")
		auc_score = self.evaluate(sess,dev_set)
		logger.info("new updated AUC scores %.4f", auc_score)

		return auc_score


	def __init__(self,Config,pretrained_embedding):
		super(ResCNNModel, self).__init__()
		self.Config = Config
		self.pretrained_embedding = tf.cast(pretrained_embedding, tf.float32)
		self.params = init_parameters(Config,self.pretrained_embedding)
		self.build()


if __name__ == "__main__":
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


	logger.info('loading data...')
	x = cPickle.load(open("./data/everything64.p","rb"))
	train, dev, test, W, idx2word, word2idx, w2i_lb, i2w_lb, nl_cls, ConfigInfo = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
	del x

	n_classes = len(w2i_lb)
	config = Config(ConfigInfo,n_classes,nl_cls)

	# for debug
	train_debug = (train[0][:5000], train[1][:5000], train[2][:5000])


	# add another special token called <pad_zero>
	n_words = len(idx2word)
	idx2word[n_words] = '<pad_zero>'
	word2idx['<pad_zero>'] = n_words
	n_words = n_words + 1
	Wemb = np.zeros((n_words,300))
	Wemb[:n_words-1] = W
	del W

	pred_error = []
	err_min = 1

	with tf.Graph().as_default():
		logger.info("Building model")
		start = time.time()
		model = ResCNNModel(config, Wemb)
		logger.info("time to build the model: %d", time.time() - start)

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session() as session:
			session.run(init)
			for epoch in range(Config.max_epochs):
				logger.info("running epoch %d", epoch)
				pred_error.append(model.run_epoch(session,train,dev))
				if pred_error[-1] < err_min:
					logger.info("new best AUC score: %.4f", pred_error[-1])
					err_min = pred_error[-1]
				logger.info("BEST AUC SCORE: %.4f", err_min)
				saver.save(session, model.Config.model_path)

			test_error = model.evaluate(session,test)
			logger.info("TEST ERROR: %.4f", test_error)
			cPickle.dump([pred_error,test_error],open(model.Config.output_path + 'results' + str(n_classes) + '.p','wb'))


