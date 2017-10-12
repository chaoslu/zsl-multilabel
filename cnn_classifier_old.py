import numpy as np
import tensorflow as tf
import logging
import time
import os
import cPickle
import argparse
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
	max_epochs = 40
	learn_rate = 0.0002
	batch_size = 20
	valid_size = 10
	dispFreq = 50
	top_k = 15
	rare_freq = 300

	# valid_batch_size = 10

	def __init__(self,ConfigInfo,n_label,data_type,label_type):
		self.nlabels = n_label
		self.nhidden = np.round(n_label * 3000 / (3000+n_label))
		self.n_words = ConfigInfo['vocab_size']  # 38564 for clean data_64, 39367 for rpl data_64
		self.max_len = ConfigInfo['max_len']	 # 580 for clean data_64, 575 for rpl
		self.max_len_sm = ConfigInfo['max_len_sm']	 # 10 for clean data_64, 9 for rpl
		self.label_type = label_type
		self.output_path = "./" + data_type + label_type + "_result_old/{:%Y%m%d_%H%M%S}/".format(datetime.now())
		self.output_path_results = "./" + data_type + label_type + "_result_old/results_only/{:%Y%m%d_%H%M%S}/".format(datetime.now())
		self.model_path = self.output_path + "model.weights"
		if label_type == 'single':
			self.max_epochs = 30
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
	params['Wemb'] = tf.Variable(W,trainable=False)
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


def rare_case_indices(rare_freq,freq,i2w):
	freq_lst = sorted([(itm,freq[itm]) for itm in freq],key = lambda t:t[1], reverse=True)
	indices = [i for i in range(len(i2w)) if freq[i2w[i]] < freq_lst[rare_freq][1]]
	return indices


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
		if self.Config.label_type == 'single':
			prediction_dropout = tf.nn.bias_add(prediction,params['ResNet_0_b'])

		# for multilabel case
		else:
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

		if self.Config.label_type == 'single':
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.labels_placeholder, logits= pred))

		else:
			loss = - tf.reduce_mean(self.labels_placeholder * tf.log(pred + 1e-6) + (1 - self.labels_placeholder) * tf.log(1 - pred + 1e-6)) 

		return loss


	def add_training_op(self,loss):
		lr = self.Config.learn_rate
		train_op = tf.train.AdamOptimizer(lr).minimize(loss)
		return train_op



	def train_on_batch(self,sess,inputs_batch,seman_batch,labels_batch):
		feed_dict = self.create_feed_dict(inputs_batch=inputs_batch,seman_batch=seman_batch,labels_batch=labels_batch,dropout_rate=self.Config.dropout_rate)
		_,loss = sess.run([self.train_op,self.loss],feed_dict=feed_dict)

		return loss


	def predict_on_batch(self, sess, inputs_batch, seman_batch):
		feed = self.create_feed_dict(inputs_batch,seman_batch)
		predictions,cnn_ecoded = sess.run([self.pred,self.encoded], feed_dict=feed)

		return predictions,cnn_ecoded

	'''
	def predict_on_batch(self, sess, inputs_batch):
		feed = self.create_feed_dict(inputs_batch,dropout_rate=0)
		predictions = sess.run(self.pred, feed_dict=feed)

		return predictions
	'''

	def eval_label(self,sess,train_examples):
		iterator = get_minibatches_idx(len(train_examples),self.Config.batch_size,False)
		# word_embeddings_old = sess.run(self.params['Wemb'])

		cnn_encodings = []
		for i,idx in enumerate(iterator):
			sm = train_examples

			sm_batch = [sm[j] for j in idx]
			padded_sm_ipnuts = prepare_data(sm_batch,self.Config)
			padded_sm_ipnuts = np.array(padded_sm_ipnuts)

			# import pdb;pdb.set_trace()

			_,cnn_encoding = self.predict_on_batch(sess,padded_sm_ipnuts,padded_sm_ipnuts)

			cnn_encodings.append(cnn_encoding)
		# import pdb;pdb.set_trace()

		cnn_encodings = np.concatenate(cnn_encodings,axis=0)

		# import pdb;pdb.set_trace()
		lb_emd = np.array(cnn_encodings)
		lb_emd = np.transpose(lb_emd)


		return lb_emd


	def evaluate(self,sess,examples,rci=None,only_encoding=False):
		iterator = get_minibatches_idx(len(examples[0]),self.Config.valid_size,False)	
		preds = []
		labels = []
		cnn_encodeds = []

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

			pred,cnn_encoded = self.predict_on_batch(sess,padded_nts_batch,padded_sm_batch)

			# multiclass version
			preds.append(pred)
			labels.append(lb_batch)
			cnn_encodeds.append(cnn_encoded)

		preds = np.concatenate(preds,axis=0)
		labels = np.concatenate(labels,axis=0)
		preds_dense = np.argmax(preds,axis=1)
		labels_dense = np.argmax(labels,axis=1)
		cnn_encodeds = np.concatenate(cnn_encodeds,axis=0)

		if only_encoding:
			return cnn_encodeds,labels_dense

		if self.Config.label_type == 'multi': 
			# precision and recall for all the classes
			pr_rcl_cls = [average_precision_score(labels[:,i],preds[:,i]) for i in range(labels.shape[1])]
			pr_rcl_cls_clean = [itm for itm in pr_rcl_cls if not np.isnan(itm)]
			pr = np.average(pr_rcl_cls_clean)

			# should return the embbeding before the classification step and the labels
			return pr,pr_rcl_cls,cnn_encodeds,labels_dense

		else:

			# multiclass version to get class prediction for each sample

			preds_rare = [preds[i] for i in range(preds.shape[0]) if labels_dense[i] in rci]
			labels_dense_rare = [labels_dense[i] for i in range(preds.shape[0]) if labels_dense[i] in rci]
			preds_rare = np.concatenate(preds_rare,axis=0)
			preds_rare = np.reshape(preds_rare,(-1,preds.shape[1]))
			preds_topk = [np.argpartition(preds,-k,axis=1) for k in range(1,self.Config.top_k+1)]
			preds_topk_rare = [np.argpartition(preds_rare,-k,axis=1) for k in range(1,self.Config.top_k+1)]

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
			pr_rcl_cls = [average_precision_score(labels[:,i],preds[:,i]) for i in range(labels.shape[1])]


			def scores_gen(preds_topk,labels_dense):
				# import pdb; pdb.set_trace()
				k_accuracy = []
				mask = []
				for k in range(self.Config.top_k):
					mask = [labels_dense[i] in preds_topk[k][i][-k-1:] for i in range(len(labels_dense))]
					tp = [msk for msk in mask if msk]
					acc = float(len(tp))/len(mask)
					k_accuracy.append(acc)

				return k_accuracy,mask

			k_acc,mask = scores_gen(preds_topk,labels_dense)
			k_acc_rare,_ = scores_gen(preds_topk_rare,labels_dense_rare)

			incorrect_classified = [(labels_dense[i],preds_dense[i]) for i,c_p in enumerate(mask) if not c_p]

			return (k_acc,k_acc_rare),(precision_cls,recall_cls,f1_cls,pr_rcl_cls),cnn_encodeds,(zip(labels_dense,preds_dense),incorrect_classified)


	def run_epoch(self,sess,train_examples,dev_set,rci=None,is_test=False):
		iterator = get_minibatches_idx(len(train_examples[0]),self.Config.batch_size,False)
		dispFreq = self.Config.dispFreq
		cnn_encodings = []

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
		acc,_,_,_ = self.evaluate(sess,dev_set,rci)
		if self.Config.label_type == 'multi':
			logger.info("new updated AUC scores %.4f",acc)  # [self.Config.top_k-1])
		else:
			logger.info("new updated AUC scores %.4f",acc[0][self.Config.top_k-1])
		return acc

	def __init__(self,Config,pretrained_embedding):
		super(ResCNNModel, self).__init__()
		self.Config = Config
		self.pretrained_embedding = tf.cast(pretrained_embedding, tf.float32)
		self.params = init_parameters(Config,self.pretrained_embedding)
		self.build()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-lf','--label_freq', default='500', type=str)
	parser.add_argument('-lt','--label_type', default='single', type=str)
	parser.add_argument('-ug','--using_glove', default=False, type=bool)
	parser.add_argument('-it','--is_train',default='train', type=str)
	parser.add_argument('-mp','--model_path',default='',type=str)
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
	x = cPickle.load(open("./data/everything" + affx + args.label_type + args.label_freq + ".p","rb"))
	train, dev, test, nl_clss, W_g, W_m, idx2word, word2idx, w2i_lb, i2w_lb, ConfigInfo, lb_freq = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]
	del x

	# data to generate label embeddings
	x = cPickle.load(open("./data/everything_lm" + affx + args.label_freq + ".p","rb"))
	label_data = x[1]
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
	config = Config(ConfigInfo,n_classes,data_type,args.label_type)

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
		if args.label_type == 'single':
			rci = rare_case_indices(model.Config.rare_freq,lb_freq[1],i2w_lb)
		else:
			rci = None
		logger.info("time to build the model: %d", time.time() - start)
		logger.info("the output path: %s", model.Config.output_path)
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		if not os.path.exists(model.Config.output_path):
			os.makedirs(model.Config.output_path)

		if not os.path.exists(model.Config.output_path_results):
			os.makedirs(model.Config.output_path_results)

		with tf.Session(config=GPU_config) as session:
			path = ''
			session.run(init)
			if args.is_train == 'train':
				path = model.Config.model_path
				for epoch in range(Config.max_epochs):
					logger.info("running epoch %d", epoch)
					pred_acc.append(model.run_epoch(session,train,dev,rci))
					if args.label_type == 'single':
						if pred_acc[-1][0][model.Config.top_k-1] > acc_max:
							logger.info("new best AUC score: %.4f", pred_acc[-1][0][model.Config.top_k-1])  # [model.Config.top_k-1])
							acc_max = pred_acc[-1][0][model.Config.top_k-1]  # [model.Config.top_k-1]
							saver.save(session, path)
						logger.info("BEST AUC SCORE: %.4f", acc_max)
					else:
						if pred_acc[-1] > acc_max:
							logger.info("new best AUC score: %.4f", pred_acc[-1])  # [model.Config.top_k-1])
							acc_max = pred_acc[-1]  # [model.Config.top_k-1]
							saver.save(session, path)
						logger.info("BEST AUC SCORE: %.4f", acc_max)
				saver.restore(session, path)	
			else:
				path = args.model_path
				saver.restore(session, path)
			# cnn_encodings_train,labels_train = model.evaluate(session,train,rci,True)
			# test_acc,precision_recall_cls,cnn_encodings,labels = model.evaluate(session,test,rci)

			# the label embedding channel
			cnn_encodings = model.eval_label(session,nl_clss)
			cPickle.dump(cnn_encodings,open("./result_lm/{:%Y%m%d_%H%M%S}/".format(datetime.now()) + 'lb_emd_' + args.label_freq + '.p',"wb"))

			# classification channel
			'''
			# make description of the configuration and test result
			if args.label_type == 'single':
				test_result = test_acc[0][-1]
			else:
				test_result = test_acc

			logger.info("TEST ERROR: %.4f", test_result)  # [model.Config.top_k-1])
			# make description of the configuration and test result
			with open(model.Config.output_path_results + "description.txt","w") as f:
				cfg = model.Config
				f.write("train_or_test: %s\nmodel_path: %s\ndata_version: %s\nfeature_maps: %d\nfilters: [%d,%d,%d]\n \
					learn_rate: %f\nbatch_size: %d\nresult: %f" % (args.is_train,path,args.data,cfg.feature_maps,
					cfg.filters[0],cfg.filters[1],cfg.filters[2],cfg.learn_rate,cfg.batch_size,test_result))  # [cfg.top_k-1]))
			f.close()

			# save all the results
			cPickle.dump([pred_acc,test_acc,precision_recall_cls,(labels_train,labels),i2w_lb,lb_freq],open(model.Config.output_path_results + 'results' + str(n_classes) + '.p','wb'))
			cPickle.dump((cnn_encodings_train,cnn_encodings),open(model.Config.output_path_results + 'encodings' + str(n_classes) + '.p','wb'))
			'''
