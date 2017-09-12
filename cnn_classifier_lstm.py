import numpy as np
import tensorflow as tf
import logging
import time
import cPickle
import os
import argparse
from datetime import datetime
from collections import OrderedDict

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import sklearn.preprocessing

from model_lstm import Model

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
	top_k = 15

	# valid_batch_size = 10

	def __init__(self,ConfigInfo,n_label,n_words_sm,dicts_mapping, data_type):
		self.nlabels = n_label
		self.n_words_sm = n_words_sm
		self.nhidden = np.round(n_label * 3000 / (3000+n_label))
		self.n_words = ConfigInfo['vocab_size']  # 38564 for clean data_64, 39367 for rpl data_64
		self.dicts_mapping = dicts_mapping
		self.max_len = ConfigInfo['max_len']	 # 580 for clean data_64, 575 for rpl
		self.max_len_sm = ConfigInfo['max_len_sm']	 # 10 for clean data_64, 9 for rpl
		self.output_path = "./" + data_type + "result_lstm/{:%Y%m%d_%H%M%S}/".format(datetime.now())
		self.output_path_results = "./" + data_type + "result_lstm/results_only/{:%Y%m%d_%H%M%S}/".format(datetime.now())
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
		n_w = Config.n_words_sm

	# f_h = Config.filters[-1]




	new_seqs = []
	target_seqs = []
	mask_seqs = []
	for sen_input in batch:
		if is_seman:
			# n_w -1 for 'pad_zero', n_w - 2 for 'EOS', n_w - 3 for 'GO'
			sen_target = sen_input + [n_w - 2]
			sen_input = [n_w - 3] + sen_input    # padding for the beginning of the sentence
			sen_mask = [True] * len(sen_input)

		# padding to reach a uniform length
		while len(sen_input) < m_l:  # + 2 * (f_h - 1):
			sen_input.append(n_w - 1)
			if is_seman:
				sen_mask.append(False)
				sen_target.append(n_w - 1)

		new_seqs.append(sen_input)
		if is_seman:
			target_seqs.append(sen_target)
			mask_seqs.append(sen_mask)

	return new_seqs,target_seqs,mask_seqs


def idxs_to_sentences(classified,i2w,i2w_sm,cfg):
	n = cfg.n_words 
	n_sm = cfg.n_words_sm
	idx_original = [itm[0] for itm in classified]
	idx_decoded = [itm[1] for itm in classified]
	idx_nts = [itm[2] for itm in classified]
	sen_original = []
	sen_decoded = []
	# lb_freq = []
	nts = []

	# import pdb; pdb.set_trace()
	for sm in idx_original:
		sen_o = [i2w_sm[i] for i in sm if i < n_sm - 2]
		sen_o = " ".join(sen_o)
		sen_original.append(sen_o)
	# import pdb; pdb.set_trace()
	for sm_d in idx_decoded:
		sen_d = [i2w[i] for i in sm_d if i < n - 2]
		sen_d = " ".join(sen_d)
		sen_decoded.append(sen_d)
	# import pfb; pdb.set_trace()
	for note in idx_nts:
		sen_n = [i2w[i] for i in note]
		sen_n = " ".join(sen_n)
		nts.append(sen_n)
	# import pdb; pdb.set_trace()
	return (sen_original,sen_decoded,nts)


'''
def scores_summary(scores,num_buckets,w2i_lb,lb_lst):
	precision = scores[0]
	recall = scores[1]
	f1 = scores[2]

	precision = [precision[w2i_lb[lb]] for lb in lb_lst]
	recall = [recall[w2i_lb[lb]] for lb in lb_lst]
	f1 = [f1[w2i_lb[lb]] for lb in lb_lst]

	return (precision,recall,f1)
'''


class ResCNNModel(Model):

	def add_placeholders(self):
		s_lz = self.Config.max_len  # + 2 * (Config.filters[-1] - 1)
		n_lbls = self.Config.nlabels
		sm_lz = self.Config.max_len_sm

		# sm_lz = self.Config.max_len_sm + 2 * (Config.filters[-1] - 1)

		self.input_placeholder = tf.placeholder(dtype=tf.int32, shape= [None,s_lz])
		self.labelsC_placeholder = tf.placeholder(dtype=tf.float32, shape= [None,n_lbls])
		self.labelsR_placeholder = tf.placeholder(dtype = tf.int32, shape= [None,sm_lz])
		self.seman_placeholder = tf.placeholder(dtype=tf.int32, shape= [None,sm_lz])
		self.mask_placeholder = tf.placeholder(dtype=tf.bool, shape= [None,sm_lz])
		self.dropout_placeholder = tf.placeholder(dtype=tf.float32)


	def create_feed_dict(self, inputs, labelsC=None, labelsR=None, seman=None, mask=None, dropout_rate=1):
		feed_dict = {self.input_placeholder:inputs}
		feed_dict[self.dropout_placeholder] = dropout_rate

		if labelsC is not None:
			feed_dict[self.labelsC_placeholder] = labelsC
		if labelsR is not None:
			feed_dict[self.labelsR_placeholder] = labelsR
		if seman is not None:
			feed_dict[self.seman_placeholder] = seman
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


	def cnn_enc(self,inputs):

			filter_hs = self.Config.filters
			params = self.params
			cnn_input = self.add_embedding(inputs)
			# conv_input = tf.nn.dropout(layer_input_dropout, self.dropout_placeholder)

			prefix = 'cnn_encoder'

			# CNN encoder
			conv_output = []
			for idx in range(len(filter_hs)):
				conv_layer = cnn_encoder_layer(cnn_input,params,prefix=_p(prefix, idx))
				conv_output.append(conv_layer)
			Encoded = tf.concat(conv_output,1)
			Encoded_dropout = tf.nn.dropout(Encoded, self.dropout_placeholder)

			return Encoded_dropout


	def add_prediction_cnn(self):
		'''
		n_ft_map = self.Config.feature_maps
		emb_sz = self.Config.embedding_size
		sen_lz = self.Config.max_len
		n_lb = self.Config.nlabels
		'''
		params = self.params

		# CNN encoding for classification
		encoded = self.encoded
		
		# RNN encoding for classification
		# encoded = self.encoded_R

		# without mapping structure
		prediction = tf.nn.bias_add(tf.matmul(encoded,params['ResNet_0_W']),params['ResNet_0_b'])

		'''
		# Residual Network
		Reslayer1_X,Reslayer1_Z = ResNet_Unit(prediction,prediction_dropout,params,prefix='ResNet_1')
		Reslayer2_X,Reslayer2_Z = ResNet_Unit(Reslayer1_X,Reslayer1_Z,params,prefix='ResNet_2')
		Reslayer3_X,Reslayer3_Z = ResNet_Unit(Reslayer2_X,Reslayer2_Z,params,prefix='ResNet_3')
		Reslayer4_X,Reslayer4_Z = ResNet_Unit(Reslayer3_X,Reslayer3_Z,params,prefix='ResNet_4')
		'''

		# pred = Reslayer4_Z
		preds_cnn = prediction
		assert preds_cnn.get_shape().as_list() == [None, self.Config.nlabels]

		return preds_cnn

	def add_prediction_rnn(self,scope=None):
		dict_emb = self.Config.dicts_mapping
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
			h_0 = self.encoded

		with tf.variable_scope(scope):
			#import pdb; pdb.set_trace()			
			x = self.seman_placeholder[:,0]
			x = tf.nn.embedding_lookup(dict_emb,x)
			x = self.add_embedding(x)
			x = tf.reshape(x,[-1,self.Config.embedding_size])
			curr_state = h_0

			states = []
			preds_rnn = []  # used for calculating the loss
			preds_rnn_oh = []  # used for presenting the decoding result
			for time_step in range(mx_len):
				# import pdb; pdb.set_trace()
				if isinstance(cell.state_size,tuple):
					c_0 = tf.zeros([cell.state_size[0]])
					curr_state = (c_0,h_0)
				output,next_state = cell(x,curr_state)
				tf.get_variable_scope().reuse_variables()
				O_h = tf.get_variable(name="O")
				b_2 = tf.get_variable(name="b_2")
				pred_rnn = tf.matmul(output,O_h) + b_2
				pred_rnn_oh = tf.argmax(pred_rnn,axis=1)
				pred_rnn_oh = tf.nn.embedding_lookup(dict_emb,pred_rnn_oh)  # change the indice to indice of the whole vocabulary


				# do smoe reshaping
				preds_rnn.append(pred_rnn)
				preds_rnn_oh.append(pred_rnn_oh)
				states.append(next_state)

				# feed the current output to next input
				x = self.add_embedding(pred_rnn_oh)  # should be batch_size * embedding_size
				x = tf.reshape(x,[-1,self.Config.embedding_size])
				curr_state = next_state

			preds_rnn = tf.reshape(tf.concat(preds_rnn,axis=1),shape=[-1,mx_len,dec_num])
			preds_rnn_oh = tf.concat(preds_rnn_oh,axis=1)

			return preds_rnn,preds_rnn_oh #,states[-1]


	def add_loss_op(self,pred_cnn,pred_rnn):  # mapping, seman
		loss_cnn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labelsC_placeholder, logits=pred_cnn))
		loss_rnn_unmasked = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labelsR_placeholder, logits=pred_rnn)
		loss_rnn = tf.reduce_mean(tf.boolean_mask(loss_rnn_unmasked,self.mask_placeholder))
		loss = loss_cnn + self.Config.beta * loss_rnn 
		return loss


	def add_training_op(self,loss):
		lr = self.Config.learn_rate
		train_op = tf.train.AdamOptimizer(lr).minimize(loss)
		return train_op



	def train_on_batch(self, sess, inputs, labels, seman_t, seman, mask):  #
		feed_dict = self.create_feed_dict(inputs=inputs, labelsC=labels, labelsR=seman_t, seman=seman, mask=mask, dropout_rate=self.Config.dropout_rate)
		_,loss = sess.run([self.train_op,self.loss],feed_dict=feed_dict)

		return loss


	def predict_on_batch(self, sess, inputs,labels,seman_t,seman,mask):
		feed = self.create_feed_dict(inputs=inputs, labelsC=labels, labelsR=seman_t, seman=seman, mask=mask)
		predictions_cnn, predictions_rnn, cnn_encodings = sess.run([self.pred_cnn,self.pred_rnn_oh,self.encoded], feed_dict=feed)

		return predictions_cnn, predictions_rnn, cnn_encodings


	def run_epoch(self,sess,train_examples,dev_set):
		iterator = get_minibatches_idx(len(train_examples[0]),self.Config.batch_size,False)
		score = 0
		dispFreq = self.Config.dispFreq
		# word_embeddings_old = sess.run(self.params['Wemb'])

		for i,idx in enumerate(iterator):
			nts,lb,sm = train_examples

			nts_batch = [nts[j] for j in idx]
			lb_batch = [lb[j] for j in idx]
			sm_batch = [sm[j] for j in idx]

			padded_nts_batch, _, _ = prepare_data(nts_batch,self.Config,False)
			padded_nts_batch = np.array(padded_nts_batch)
			padded_sm_ipnuts, padded_sm_target, mask = prepare_data(sm_batch,self.Config,True)
			padded_sm_ipnuts = np.array(padded_sm_ipnuts)
			padded_sm_target = np.array(padded_sm_target)
			lb_batch = np.array(lb_batch)

			loss = self.train_on_batch(sess, padded_nts_batch, lb_batch, padded_sm_target, padded_sm_ipnuts, mask)

			# print something about the loss
			if i % dispFreq == 0:
				logger.info("loss until batch_%d, : %f", i,loss)
				
				# to see if the word embeddings are being trained.
				'''
				word_embeddings_new = sess.run(self.params['Wemb'])
				if not np.allclose(word_embeddings_new,word_embeddings_old):
					logger.info("The word embeddings are being trained")
					word_embeddings_old = word_embeddings_new
				'''
		logger.info("Evaluating on devlopment data")
		score,_,_,_,_,_ = self.evaluate(sess,dev_set)
		logger.info("new updated AUC scores %.4f", score[self.Config.top_k-1])

		return score

	def evaluate(self,sess,examples):
		iterator = get_minibatches_idx(len(examples[0]),self.Config.valid_size,False)	
		preds_cnn = []
		labels = []
		cnn_encodings = []

		preds_rnn = []
		sm_target = []
		masks = []
		nts = []
		for i,idx in enumerate(iterator):
			nts,lb,sm = examples

			nts_batch = [nts[i] for i in idx]
			lb_batch = [lb[j] for j in idx]
			sm_batch = [sm[j] for j in idx]

			padded_nts_batch, _, _ = prepare_data(nts_batch,self.Config,False)
			padded_nts_batch = np.array(padded_nts_batch)
			padded_sm_ipnuts, padded_sm_target, mask = prepare_data(sm_batch,self.Config,True)
			padded_sm_target = np.array(padded_sm_target)
			padded_sm_ipnuts = np.array(padded_sm_ipnuts)
			mask = np.array(mask)
			lb_batch = np.array(lb_batch)

			predictions_cnn, predictions_rnn, cnn_encoded = self.predict_on_batch(sess, padded_nts_batch, lb_batch, padded_sm_target, padded_sm_ipnuts, mask)

			preds_cnn.append(predictions_cnn)
			labels.append(lb_batch)
			cnn_encodings.append(cnn_encoded)

			preds_rnn.append(predictions_rnn)
			sm_target.append(padded_sm_target)
			masks.append(mask)
			nts = nts + nts_batch

		# all results are transformed into num_dev * ?
		preds_cnn = np.concatenate(preds_cnn,axis=0)  
		labels = np.concatenate(labels,axis=0)
		cnn_encodings = np.concatenate(cnn_encodings,axis=0)
		preds_rnn = np.concatenate(preds_rnn,axis=0)
		sm_target = np.concatenate(sm_target,axis=0)
		masks = np.concatenate(masks,axis=0)

		# transform preds_cnn into densely representation (top k)
		preds_cnn_topk = [np.argpartition(preds_cnn,-k,axis=1) for k in range(1,self.Config.top_k+1)]
		preds_cnn_dense = np.argmax(preds_cnn, axis=1)
		preds_cnn_dense_k = preds_cnn_topk[self.Config.top_k - 1][:,-self.Config.top_k - 1:]
		labels_dense = np.argmax(labels,axis=1)


		# import pdb; pdb.set_trace()
		# transform preds_cnn to one_hot form
		label_binarizer = sklearn.preprocessing.LabelBinarizer()
		label_binarizer.fit(range(self.Config.nlabels))
		preds_cnn_peak = label_binarizer.transform(preds_cnn_dense)
		assert preds_cnn_peak.shape == labels.shape

		# import pdb; pdb.set_trace()
		true_positive_cls = np.sum(preds_cnn_peak * labels,axis=0).astype('f')
		gold_cls = np.sum(labels,axis=0).astype('f')
		preds_cnn_cls = np.sum(preds_cnn_peak,axis=0).astype('f')

		# precision recall and f1 score
		precision_cls = [tp / gd if gd != 0 else 0 for (tp,gd) in zip(true_positive_cls,gold_cls)] 
		recall_cls = [tp / prd if prd != 0 else 0 for (tp,prd) in zip(true_positive_cls,preds_cnn_cls)]
		f1_cls = [2 * p * r / (p+r) if p * r != 0 else 0 for (p,r) in zip(precision_cls,recall_cls)]

		# precicion and recall for all the classes		
		pr_rcl_cls = [average_precision_score(labels[:,i],preds_cnn[:,i]) for i in range(labels.shape[1])]

		# error for all choices of k
		acc_list = []
		classified_result_k = []
		for k in range(self.Config.top_k):
			true_positive = [labels_dense[i] for i in range(labels_dense.shape[0]) if labels_dense[i] in preds_cnn_topk[k][i][-k-1:]]
			acc_list.append(float(len(true_positive))/labels_dense.shape[0])

			# for k, get the boolean mask of the classification result
			if k == self.Config.top_k - 1:
				classified_result_k = [labels_dense[i] in preds_cnn_topk[k][i][-k-1:] for i in range(labels_dense.shape[0])]

		# auc = roc_auc_score(labels,preds)
		# error = 1. - auc

		# for those incorrectly classified, show its decoding result
		masked_decoded = []
		masked_sm_target = []
		for i in range(preds_rnn.shape[0]):
			rnn_decoded = [preds_rnn[i][j] for j in range(preds_rnn.shape[1]) if masks[i][j]]
			sm_original = [sm_target[i][j] for j in range(sm_target.shape[1]) if masks[i][j]]
			masked_sm_target.append(sm_original)
			masked_decoded.append(rnn_decoded)
		
		incorrect_cls_decoded = [(masked_sm_target[i],masked_decoded[i],nts[i]) for i,c_p in enumerate(classified_result_k) if not c_p]  # if not correctly classified
		all_decoded = [(masked_sm_target[i],masked_decoded[i],nts[i]) for i in range(len(classified_result_k))]
		# import pdb; pdb.set_trace()
		return acc_list, (precision_cls,recall_cls,f1_cls,pr_rcl_cls), incorrect_cls_decoded, all_decoded, cnn_encodings, labels


	def __init__(self,Config,pretrained_embedding):
		super(ResCNNModel, self).__init__()
		self.Config = Config
		self.pretrained_embedding = tf.cast(pretrained_embedding, tf.float32)
		self.params = init_parameters(Config,self.pretrained_embedding)
		self.build()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-lf','--label_freq', default='500', type=str)
	parser.add_argument('-ug','--using_glove', default=False, type=bool)
	parser.add_argument('-it','--is_train',default='train', type=str)
	parser.add_argument('-mp','--model_path',default='',type=str)

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
	logger.info('loading data...')
	x = cPickle.load(open("./data/lstm_everything_new" + args.label_freq + ".p","rb"))
	train, dev, test, W_g, W_m, idx2word, word2idx, i2w_lb, i2w_sm, ConfigInfo, dicts_mapping, lb_freq = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]
	
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
	config = Config(ConfigInfo, n_classes, n_words_sm, dicts_mapping, data_type)

	pred_acc = []
	acc_max = 0
	
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

		if not os.path.exists(model.Config.output_path_results):
			os.makedirs(model.Config.output_path_results)	
		
		with tf.Session(config=GPU_config) as session:
			path = ''
			session.run(init)
			if args.is_train == 'train':
				path = model.Config.model_path
				for epoch in range(Config.max_epochs):
					logger.info("running epoch %d", epoch)
					pred_acc.append(model.run_epoch(session,train,dev))
					# import pdb; pdb.set_trace()
					if pred_acc[-1][model.Config.top_k-1] > acc_max:
						logger.info("new best AUC score: %.4f", pred_acc[-1][model.Config.top_k-1])
						acc_max = pred_acc[-1][model.Config.top_k-1] 
						saver.save(session, path)
					logger.info("BEST AUC SCORE: %.4f", acc_max)

				saver.restore(session, path)
				test_score,precision_recall_cls,incorrectly_decoded,all_decoded,cnn_encodings,labels = model.evaluate(session,test)
				incorrectly_decoded = idxs_to_sentences(incorrectly_decoded,idx2word,i2w_sm,model.Config)
				all_decoded = idxs_to_sentences(all_decoded,idx2word,i2w_sm,model.Config)
				logger.info("TEST ERROR: %.4f",test_score[model.Config.top_k-1])

			else:
				path = args.model_path
				saver.restore(session, path)
				test_score,precision_recall_cls,incorrectly_decoded,all_decoded,cnn_encodings,labels = model.evaluate(session,test)
				incorrectly_decoded = idxs_to_sentences(incorrectly_decoded,idx2word,i2w_sm,model.Config)
				all_decoded = idxs_to_sentences(all_decoded,idx2word,i2w_sm,model.Config)
				logger.info("TEST ERROR: %.4f",test_score[model.Config.top_k-1])

			# make description of the configuration and test result
			with open(model.Config.output_path_results + "description.txt","w") as f:
				cfg = model.Config
				f.write("train_or_test: %s\nmodel_path: %s\nfeature_maps: %d\nfilters: [%d,%d,%d]\nrnncel: %s\nlearn_rate: %f\nbatch_size: %d\nbeta: %f\nresult: %f" % (args.is_train,path,cfg.feature_maps,cfg.filters[0],cfg.filters[1],cfg.filters[2],cfg.rnncell,cfg.learn_rate,cfg.batch_size,cfg.beta,test_score[cfg.top_k-1]))
			f.close()

			# save all the results
			cPickle.dump([pred_acc ,test_score ,precision_recall_cls,cnn_encodings,labels,incorrectly_decoded,all_decoded,lb_freq,i2w_lb],open(model.Config.output_path_results + 'results' + str(n_classes) + ".p","wb"))
