import numpy as np
import tensorflow as tf
import logging
import time
import os
import cPickle
from datetime import datetime
from collections import OrderedDict

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import sklearn.preprocessing

from model_old import Model
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
	beta2 = 1
	max_epochs = 80
	learn_rate = 0.0002
	batch_size = 20
	valid_size = 10
	dispFreq = 50
	top_k = 15

	# valid_batch_size = 10

	def __init__(self,ConfigInfo,n_label):
		self.nlabels = n_label
		self.nhidden = np.round(n_label * 3000 / (3000+n_label))
		self.n_words = ConfigInfo['vocab_size']  # 38564 for clean data_64, 39367 for rpl data_64
		self.max_len = ConfigInfo['max_len']	 # 580 for clean data_64, 575 for rpl
		self.max_len_sm = ConfigInfo['max_len_sm']	 # 10 for clean data_64, 9 for rpl
		self.output_path = "./result_old/{:%Y%m%d_%H%M%S}/".format(datetime.now())
		self.output_path_results = "./result_old/results_only/{:%Y%m%d_%H%M%S}/".format(datetime.now())
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
	# otherwise, W will be initialized by random word embedding
	params = OrderedDict()
	params['Wemb'] = tf.Variable(W)
	# params['Wemb'] = tf.Variable(tf.random_uniform((W.shape), minval=-0.01, maxval=0.01))

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
		emb_sz = self.Config.embedding_size
		mhidden = self.Config.mhidden

		encoded_notes = self.cnn_enc()
		with tf.name_scope("mapping_to_seman"):
			W1_n2s = tf.Variable(tf.random_uniform((emb_sz,mhidden),minval=-0.01,maxval=0.01))
			W2_n2s = tf.Variable(tf.random_uniform((emb_sz,mhidden),minval=-0.01,maxval=0.01))
			hidden_layer = tf.nn.relu(tf.matmul(encoded_notes,W1_n2s))
			notes2seman_layer = tf.nn.relu(tf.matmul(hidden_layer,W2_n2s))
			regularizer = tf.nn.l2_loss(W1_n2s) + tf.nn.l2_loss(W2_n2s)
		return notes2seman_layer,regularizer


	def add_prediction(self,is_seman=False):
		'''
		n_ft_map = self.Config.feature_maps
		emb_sz = self.Config.embedding_size
		sen_lz = self.Config.max_len
		n_lb = self.Config.nlabels
		'''
		params = self.params

		# without mapping structure
		encoded = self.cnn_enc(is_seman)
		prediction = tf.matmul(encoded,params['ResNet_0_W'])
		
		# for multiclass case
		# prediction_dropout = tf.nn.bias_add(prediction,params['ResNet_0_b'])
		
		# for multilabel case
		prediction_dropout = tf.sigmoid(tf.nn.bias_add(prediction,params['ResNet_0_b']))
		
		'''
		# Residual Network
		Reslayer1_X,Reslayer1_Z = ResNet_Unit(prediction,prediction_dropout,params,prefix='ResNet_1')
		Reslayer2_X,Reslayer2_Z = ResNet_Unit(Reslayer1_X,Reslayer1_Z,params,prefix='ResNet_2')
		Reslayer3_X,Reslayer3_Z = ResNet_Unit(Reslayer2_X,Reslayer2_Z,params,prefix='ResNet_3')
		Reslayer4_X,Reslayer4_Z = ResNet_Unit(Reslayer3_X,Reslayer3_Z,params,prefix='ResNet_4')
		'''

		# pred = Reslayer4_Z
		pred = prediction_dropout
		assert pred.get_shape().as_list() == [None, self.Config.nlabels]

		return pred


	def add_loss_op(self,pred):  # mapping,seman,
		# for evaluation of auc
		# loss_data_seman = tf.nn.l2_loss(mapping - seman)/self.Config.batch_size * 2 + self.regularizer * self.Config.beta1
		loss_seman_label = - tf.reduce_mean(self.labels_placeholder * tf.log(pred + 1e-6) + (1 - self.labels_placeholder) * tf.log(1 - pred + 1e-6))
		loss = loss_seman_label  # + loss_data_seman * self.Config.beta2
		
		# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels_placeholder, logits= pred))

		return loss


	def add_training_op(self,loss):
		lr = self.Config.learn_rate
		train_op = tf.train.AdamOptimizer(lr).minimize(loss)
		return train_op



	def train_on_batch(self,sess,inputs_batch,seman_batch,labels_batch):
		feed_dict = self.create_feed_dict(inputs_batch=inputs_batch,seman_batch=seman_batch,labels_batch=labels_batch,dropout_rate=self.Config.dropout_rate)
		_,loss = sess.run([self.train_op,self.loss],feed_dict=feed_dict)

		return loss

	'''
	def predict_on_batch(self, sess, inputs_batch):
		feed = self.create_feed_dict(inputs_batch,dropout_rate=0)
		predictions = sess.run(self.pred, feed_dict=feed)

		return predictions
	'''



	def evaluate(self,sess,examples):
		iterator = get_minibatches_idx(len(examples[0]),self.Config.valid_size,False)	
		preds = []
		labels = []

		for i,idx in enumerate(iterator):
			nts,lb,sm = examples

			nts_batch = [nts[i] for i in idx]
			lb_batch = [lb[j] for j in idx]
			sm_batch = [sm[j] for j in idx]

			padded_nts_batch = prepare_data(nts_batch,self.Config,False)
			padded_nts_batch = np.array(padded_nts_batch)
			padded_sm_batch = prepare_data(sm_batch,self.Config,True)
			padded_sm_batch = np.array(padded_sm_batch)
			lb_batch = np.array(lb_batch)

			pred = self.predict_on_batch(sess,padded_nts_batch,padded_sm_batch)
			
			# multiclass version
			preds.append(pred)
			labels.append(lb_batch)
		
		preds = np.concatenate(preds,axis=0)
		labels = np.concatenate(labels,axis=0)

		'''
		# multiclass version to get class prediction for each sample
		preds_dense = np.argmax(preds,axis=1)
		preds_topk = [np.argpartition(preds,-k,axis=1) for k in range(1,self.Config.top_k+1)]
		labels_dense = np.argmax(labels,axis=1)
		
		label_binarizer = sklearn.preprocessing.LabelBinarizer()
		label_binarizer.fit(range(self.Config.nlabels))
		preds_peak = label_binarizer.transform(preds_dense)
		assert preds_peak.shape == labels.shape

		true_positive_cls = np.sum(preds_peak * labels,axis=0).astype('f')
		gold_cls = np.sum(labels,axis=0).astype('f')
		preds_cnn_cls = np.sum(preds_peak,axis=0).astype('f')

		# precision recall f1 scores
		precision_cls = [tp/gd if gd != 0 else 0 for (tp,gd) in zip(true_positive_cls,gold_cls)]
		recall_cls = [tp/prd if prd != 0 else 0 for (tp,prd) in zip(true_positive_cls,preds_cnn_cls)]
		f1_cls = [2 * p * r/(p+r) if p * r != 0 else 0 for (p,r) in zip(precision_cls,recall_cls)]

		# import pdb; pdb.set_trace()
		k_accuracy = []
		for k in range(self.Config.top_k):
			mask = [labels_dense[i] in preds_topk[k][i][-k-1:] for i in range(preds.shape[0])]
			tp = [msk for msk in mask if msk]
			acc = float(len(tp))/len(mask)
			k_accuracy.append(acc)
		'''

		# precision and recall for all the classes
		pr_rcl_cls = [average_precision_score(labels[:,i],preds[:,i]) for i in range(labels.shape[1])]
		
		# precision and recall for all instances
		# preds = preds.flatten()
		# labels = labels.flatten()
   		pr_rcl_cls_clean = [itm for itm in pr_rcl_cls if not np.isnan(itm)]
		pr = np.average(pr_rcl_cls_clean)
		
		# error = 1. - pr
		# auc = roc_auc_score(labels,preds)
		# error = 1. - auc

		# return k_accuracy,(precision_cls,recall_cls,f1_cls,pr_rcl_cls)
		return pr,pr_rcl_cls
	

	def run_epoch(self,sess,train_examples,dev_set):
		iterator = get_minibatches_idx(len(train_examples[0]),self.Config.batch_size,False)
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
		acc,_ = self.evaluate(sess,dev_set)
		logger.info("new updated AUC scores %.4f",acc)  # [self.Config.top_k-1])

		return acc

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
	x = cPickle.load(open("./data/everything500.p","rb"))
	# train, dev, test, W, idx2word, word2idx, i2w_lb, i2w_sm, dicts_mapping, ConfigInfo, lb_freq_list = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]
	
	train, dev, test, W, idx2word, word2idx, w2i_lb, i2w_lb, nl_clss, ConfigInfo = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]

	del x

	n_classes = len(i2w_lb)
	config = Config(ConfigInfo,n_classes)
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

	pred_acc = []
	acc_max = 0

	GPU_config = tf.ConfigProto()
	GPU_config.gpu_options.per_process_gpu_memory_fraction = 0.2
	with tf.Graph().as_default():
		tf.set_random_seed(1234)
		logger.info("Building model")
		start = time.time()
		model = ResCNNModel(config, Wemb)
		logger.info("time to build the model: %d", time.time() - start)

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session(config=GPU_config) as session:
			session.run(init)
			for epoch in range(Config.max_epochs):
				logger.info("running epoch %d", epoch)
				pred_acc.append(model.run_epoch(session,train,dev))
				# if pred_acc[-1][model.Config.top_k-1] > acc_max:
				if pred_acc[-1] > acc_max:
					logger.info("new best AUC score: %.4f", pred_acc[-1])  # [model.Config.top_k-1])
					acc_max = pred_acc[-1]  # [model.Config.top_k-1]
					saver.save(session, model.Config.model_path)
				logger.info("BEST AUC SCORE: %.4f", acc_max)
			saver.restore(session, model.Config.model_path)	
			test_acc,precision_recall_cls = model.evaluate(session,test)
			logger.info("TEST ERROR: %.4f", test_acc)  # [model.Config.top_k-1])

			# make description of the configuration and test result
			if not os.path.exists(model.Config.output_path_results):
				os.makedirs(model.Config.output_path_results)
			with open(model.Config.output_path_results + "description.txt","w") as f:
				cfg = model.Config
				f.write("feature_maps: %d\nfilters: [%d,%d,%d]\nlearn_rate: %f\nbatch_size: %d\nresult: %f" % (cfg.feature_maps,cfg.filters[0],cfg.filters[1],cfg.filters[2],cfg.learn_rate,cfg.batch_size,test_acc))  # [cfg.top_k-1]))
			f.close()

			# save all the results
			cPickle.dump([pred_acc,test_acc,precision_recall_cls],open(model.Config.output_path_results + 'results' + str(n_classes) + '.p','wb'))
