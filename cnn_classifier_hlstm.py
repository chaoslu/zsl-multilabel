import numpy as np
import tensorflow as tf
import logging
import time
import cPickle
import os
from datetime import datetime
from collections import OrderedDict

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
import sklearn.preprocessing

from model_hlstm import Model

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
	dropout_rate = 0.5
	beta = 1
	max_epochs = 80
	learn_rate = 0.0002
	batch_size = 20
	valid_size = 10
	dispFreq = 50
	top_k = 15

	# valid_batch_size = 10

	def __init__(self,ConfigInfo,n_label,n_words_sm,dicts_mapping):
		self.nlabels = n_label
		self.n_words_sm = n_words_sm
		self.nhidden = np.round(n_label * 3000 / (3000+n_label))
		self.dicts_mapping = dicts_mapping
		self.n_words = ConfigInfo['vocab_size']
		self.n_words_sm = len(dicts_mapping)
		self.max_len = ConfigInfo['max_len']	
		self.max_len_sm = ConfigInfo['max_len_sm']
		self.n_diags = ConfigInfo['n_diagnosis']	 
		self.output_path = "./result_hlstm/{:%Y%m%d_%H%M%S}/".format(datetime.now())
		self.output_path_results = "./result_hlstm/results_only/{:%Y%m%d_%H%M%S}/".format(datetime.now())
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


def prepare_data(batch,Config):
	n_w = Config.n_words
	m_l = Config.max_len
	
	new_seqs = []
	for sen_input in batch:
		sen_in = [itm for itm in sen_input]
		while len(sen_in) < m_l:
			sen_in.append(n_w-1)
		new_seqs.append(sen_in)

	return new_seqs

def prepare_data_sm(batch,Config):
	m_l = Config.max_len_sm
	n_w = Config.n_words_sm

	# f_h = Config.filters[-1]

	new_seqs = []
	target_seqs = []
	mask_seqs = []
	for sen_inputs in batch:
		new_labels = []
		target_labels = []
		mask_labels = []
		# import pdb;pdb.set_trace()
		for sen_input in sen_inputs:
			sen_in = [itm for itm in sen_input]
			sen_target = []
			sen_mask = []
			if len(sen_input) > 0:	# ignore the padded labels
				# n_w -1 for 'pad_zero', n_w - 2 for 'EOS', n_w - 3 for 'GO'
				sen_target = sen_in + [n_w - 2]
				sen_in = [n_w - 3] + sen_in    # padding for the beginning of the sentence
				sen_mask = [True] * len(sen_in)

			# padding to reach a uniform length
			while len(sen_in) < m_l:  # + 2 * (f_h - 1):
				sen_in.append(n_w - 1)
				sen_mask.append(False)
				sen_target.append(n_w - 1)
			new_labels.append(sen_in)
			target_labels.append(sen_target)
			mask_labels.append(sen_mask)

		new_seqs.append(new_labels)
		target_seqs.append(target_labels)
		mask_seqs.append(mask_labels)

	return new_seqs,target_seqs,mask_seqs


def idxs_to_sentences(classified,i2w,i2w_sm,cfg):
	n = cfg.n_words 
	n_sm = cfg.n_words_sm
	idx_original = [itm[0] for itm in classified]
	idx_decoded = [itm[1] for itm in classified]
	idx_nts = [itm[2] for itm in classified]
	sen_original = []
	sen_decoded = []
	lb_freq = []
	nts = []
	# import pdb;pdb.set_trace()
	for sm in idx_original:
		sen_ori = ""
		for sm_diag in sm:
			sen_o = [i2w_sm[i] for i in sm_diag if i < n_sm - 2]
			sen_o = " ".join(sen_o)
			sen_o = sen_o + '. '	
			sen_ori = sen_ori + sen_o
		sen_original.append(sen_ori)
	# import pdb;pdb.set_trace()
	for sm_d in idx_decoded:
		sen_dec = ""
		for sm_decoded_diag in sm_d:
			sen_d = [i2w[i] for i in sm_decoded_diag if i < n - 2]
			sen_d = " ".join(sen_d)
			sen_d = sen_d + '. '
			sen_dec = sen_dec + sen_d
		sen_decoded.append(sen_dec)
	# import pdb;pdb.set_trace()
	for note in idx_nts:
		sen_n = [i2w[i] for i in note]
		sen_n = " ".join(sen_n)
		nts.append(sen_n)
	# import pdb; pdb.set_trace()
	return (sen_original,sen_decoded,nts)


class ResCNNModel(Model):

	def add_placeholders(self):
		s_lz = self.Config.max_len  # + 2 * (Config.filters[-1] - 1)
		n_lbls = self.Config.nlabels
		sm_lz = self.Config.max_len_sm
		n_diags = self.Config.n_diags

		# sm_lz = self.Config.max_len_sm + 2 * (Config.filters[-1] - 1)

		self.input_placeholder = tf.placeholder(dtype=tf.int32, shape= [None,s_lz])
		self.labelsC_placeholder = tf.placeholder(dtype=tf.float32, shape= [None,n_lbls])
		self.labelsR_placeholder = tf.placeholder(dtype = tf.int32, shape= [None,n_diags,sm_lz])
		self.seman_placeholder = tf.placeholder(dtype=tf.int32, shape= [None,n_diags,sm_lz])
		self.mask_placeholder = tf.placeholder(dtype=tf.bool, shape= [None,n_diags,sm_lz])
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
		encoded = self.encoded

		# without mapping structure
		prediction = tf.nn.bias_add(tf.matmul(encoded,params['ResNet_0_W']),params['ResNet_0_b'])
		preds_cnn = tf.sigmoid(prediction)
		'''
		# Residual Network
		Reslayer1_X,Reslayer1_Z = ResNet_Unit(prediction,prediction_dropout,params,prefix='ResNet_1')
		Reslayer2_X,Reslayer2_Z = ResNet_Unit(Reslayer1_X,Reslayer1_Z,params,prefix='ResNet_2')
		Reslayer3_X,Reslayer3_Z = ResNet_Unit(Reslayer2_X,Reslayer2_Z,params,prefix='ResNet_3')
		Reslayer4_X,Reslayer4_Z = ResNet_Unit(Reslayer3_X,Reslayer3_Z,params,prefix='ResNet_4')
		'''

		# pred = Reslayer4_Z
		assert preds_cnn.get_shape().as_list() == [None, self.Config.nlabels]

		return preds_cnn

	def add_prediction_rnn(self,scope=None):
		dict_emb = self.Config.dicts_mapping
		dec_num = self.Config.n_words_sm
		mx_len = self.Config.max_len_sm
		n_diags = self.Config.n_diags
		h_size = self.Config.rnn_hsize
		scope = scope or type(self).__name__

		if self.Config.rnncell == "GRU":
			cell = tf.contrib.rnn.GRUCell(h_size)
		elif self.Config.rnncell == "LSTM":
			cell = tf.contrib.rnn.LSTMCell(h_size) 

		preds = []
		preds_oh = []
		for diag in range(n_diags):
			diag_mark = str(diag)
			with tf.variable_scope(scope + diag_mark):
				O_h = tf.get_variable(name="O"+diag_mark, shape=[h_size,dec_num], initializer= tf.contrib.layers.xavier_initializer())
				b_2 = tf.get_variable(name="b_2"+diag_mark, shape=[dec_num,], initializer= tf.contrib.layers.xavier_initializer())
				h_0 = self.encoded
				
			with tf.variable_scope(scope+diag_mark):
	
				preds_rnn = []  # used for calculating the loss
				preds_rnn_oh = []  # used for presenting the decoding result

				x = self.seman_placeholder[:,diag,0]
				x = tf.nn.embedding_lookup(dict_emb,x)
				x = self.add_embedding(x)
				x = tf.reshape(x,[-1,self.Config.embedding_size])
				curr_state = h_0

				for time_step in range(mx_len):

					# import pdb; pdb.set_trace()
					if isinstance(cell.state_size,tuple):
						c_0 = tf.zeros([cell.state_size[0]])
						curr_state = (c_0,h_0)
					output,next_state = cell(x,curr_state)
					tf.get_variable_scope().reuse_variables()
					O_h = tf.get_variable(name="O"+diag_mark)
					b_2 = tf.get_variable(name="b_2"+diag_mark)
					pred_rnn = tf.matmul(output,O_h) + b_2
					pred_rnn_oh = tf.arg_max(pred_rnn,dimension=1)
					pred_rnn_oh = tf.nn.embedding_lookup(dict_emb,pred_rnn_oh)  # change the indice to indice of the whole vocabulary

					# import pdb;pdb.set_trace()
					# do smoe reshaping
					preds_rnn.append(pred_rnn)
					preds_rnn_oh.append(pred_rnn_oh)

					# feed the current output to next input
					x = self.add_embedding(pred_rnn_oh)  # should be batch_size * embedding_size
					x = tf.reshape(x,[-1,self.Config.embedding_size])
					curr_state = next_state

				preds_rnn = tf.concat(preds_rnn,axis=1)
				preds_rnn_oh = tf.concat(preds_rnn_oh,axis=1)

			preds.append(preds_rnn)
			preds_oh.append(preds_rnn_oh)

		preds = tf.reshape(tf.concat(preds,axis=1), shape=[-1,n_diags,mx_len,dec_num])	
		preds_oh = tf.reshape(tf.concat(preds_oh,axis=1), shape=[-1,n_diags,mx_len])
		return preds,preds_oh


	def add_loss_op(self,pred_cnn,pred_rnn):  # mapping, seman
		loss_cnn = -tf.reduce_mean(self.labelsC_placeholder * tf.log(pred_cnn + 1e-6) + (1. - self.labelsC_placeholder) * tf.log(1 - pred_cnn + 1e-6))
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
		predictions_cnn, predictions_rnn = sess.run([self.pred_cnn,self.pred_rnn_oh], feed_dict=feed)

		return predictions_cnn, predictions_rnn


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
		        # import pdb; pdb.set_trace()
			padded_nts_batch = prepare_data(nts_batch,self.Config)
			padded_nts_batch = np.array(padded_nts_batch)
			assert type(padded_nts_batch).__name__ == 'ndarray'

			padded_sm_inputs, padded_sm_target, mask = prepare_data_sm(sm_batch,self.Config)
			padded_sm_inputs = np.array(padded_sm_inputs)
			padded_sm_target = np.array(padded_sm_target)
			lb_batch = np.array(lb_batch)
			assert type(padded_sm_inputs).__name__ == 'ndarray'
			assert type(padded_sm_target).__name__ == 'ndarray'
			assert type(lb_batch).__name__ == 'ndarray'
			# import pdb;pdb.set_trace()
			loss = self.train_on_batch(sess, padded_nts_batch, lb_batch, padded_sm_target, padded_sm_inputs, mask)

			# print something about the loss

			if i % dispFreq == 0:
				logger.info("loss until batch_%d, : %f", i,loss)


		# import pdb; pdb.set_trace()
		logger.info("Evaluating on devlopment data")
		score,_,_ = self.evaluate(sess,dev_set)
		logger.info("new updated AUC scores %.4f", score)
		# import pdb; pdb.set_trace()

		return score

	def evaluate(self,sess,examples):
		iterator = get_minibatches_idx(len(examples[0]),self.Config.valid_size,False)	
		preds_cnn = []
		labels = []

		preds_rnn = []
		sm_target = []
		masks = []
		notes = []
		for i,idx in enumerate(iterator):
			nts,lb,sm = examples

			nts_batch = [nts[i] for i in idx]
			lb_batch = [lb[j] for j in idx]
			sm_batch = [sm[j] for j in idx]

			padded_nts_batch = prepare_data(nts_batch,self.Config)
			padded_nts_batch = np.array(padded_nts_batch)	

			assert type(padded_nts_batch).__name__ == 'ndarray'
			# import pdb; pdb.set_trace()
			padded_sm_inputs, padded_sm_target, mask = prepare_data_sm(sm_batch,self.Config)
			padded_sm_target = np.array(padded_sm_target)
			padded_sm_inputs = np.array(padded_sm_inputs)
			assert type(padded_sm_inputs).__name__ == 'ndarray'
			mask = np.array(mask)
			lb_batch = np.array(lb_batch)

			predictions_cnn, predictions_rnn = self.predict_on_batch(sess, padded_nts_batch, lb_batch, padded_sm_target, padded_sm_inputs, mask)

			preds_cnn.append(predictions_cnn)
			labels.append(lb_batch)

			preds_rnn.append(predictions_rnn)
			sm_target.append(padded_sm_target)
			masks.append(mask)
			notes = notes + nts_batch

		# all results are transformed into num_dev * ?
		preds_rnn = np.concatenate(preds_rnn,axis=0)
		sm_target = np.concatenate(sm_target,axis=0)
		masks = np.concatenate(masks,axis=0)

		# transform preds_cnn into densely representation (top k)
		preds_cnn = np.concatenate(preds_cnn,axis=0)  
		labels = np.concatenate(labels,axis=0)
		# preds_cnn_flat = preds_cnn.flatten()
		# labels_flatten = labels.flatten()		
		
		# precicion and recall for all the classes		
		pr_rcl_cls = [average_precision_score(labels[:,i],preds_cnn[:,i]) for i in range(labels.shape[1])]
		pr_rcl_cls_clean = [itm for itm in pr_rcl_cls if not np.isnan(itm)]
		pr_rcl_flat = np.mean(pr_rcl_cls_clean)

		# for those incorrectly classified, show its decoding result
		masked_decoded = []
		masked_sm_target = []
		for i in range(preds_rnn.shape[0]):
			masked_i = []
			masked_sm_i = []
			for k in range(preds_rnn.shape[1]):
				rnn_decoded = [preds_rnn[i][k][j] for j in range(preds_rnn.shape[2]) if masks[i][k][j]]
				sm_original = [sm_target[i][k][j] for j in range(sm_target.shape[2]) if masks[i][k][j]]
				# import pdb;pdb.set_trace()
				masked_i.append(sm_original)
				masked_sm_i.append(rnn_decoded)
			masked_sm_target.append(masked_i)
			masked_decoded.append(masked_sm_i)
			# import pdb; pdb.set_trace()
		all_decoded = [(masked_sm_target[i],masked_decoded[i],notes[i]) for i in range(preds_rnn.shape[0])]
		# all_decoded = idxs_to_sentences(all_decoded,idx2word,i2w_sm,model.Config)
		# import pdb; pdb.set_trace()
		return pr_rcl_flat, pr_rcl_cls,all_decoded


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
	x = cPickle.load(open("./data/hlstm_everything1000.p","rb"))
	train, dev, test, W, idx2word, word2idx, i2w_lb, i2w_sm, dicts_mapping, ConfigInfo, lb_freq = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]
	del x
	train_debug = (train[0][:70],train[1][:70],train[2][:70])

	# import pdb;pdb.set_trace()
	n_classes = len(i2w_lb)
	n_words_sm = len(i2w_sm)
	config = Config(ConfigInfo,n_classes,n_words_sm,dicts_mapping)

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

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		with tf.Session(config=GPU_config) as session:
			session.run(init)
			for epoch in range(Config.max_epochs):
				logger.info("running epoch %d", epoch)
				pred_acc.append(model.run_epoch(session,train,dev))
				# import pdb; pdb.set_trace()
				if pred_acc[-1] > acc_max:
					logger.info("new best AUC score: %.4f", pred_acc[-1])
					acc_max = pred_acc[-1] 
					saver.save(session,model.Config.model_path)
				logger.info("BEST AUC SCORE: %.4f", acc_max)

			saver.restore(session,model.Config.model_path)
			test_score,precision_recall_cls,all_decoded = model.evaluate(session,test)
			all_decoded = idxs_to_sentences(all_decoded,idx2word,i2w_sm,model.Config)
			logger.info("TEST ERROR: %.4f",test_score)

			# make description of the configuration and test result
			if not os.path.exists(model.Config.output_path_results):
				os.makedirs(model.Config.output_path_results)
			with open(model.Config.output_path_results + "description.txt","w") as f:
				cfg = model.Config
				f.write("feature_maps: %d\nfilters: [%d,%d,%d]\nrnncel: %s\nlearn_rate: %f\nbatch_size: %d\nbeta: %f\nresult: %f" % (cfg.feature_maps,cfg.filters[0],cfg.filters[1],cfg.filters[2],cfg.rnncell,cfg.learn_rate,cfg.batch_size,cfg.beta,test_score))
			f.close()

			# save all the results
			cPickle.dump([pred_acc ,test_score ,precision_recall_cls,all_decoded,lb_freq,i2w_lb],open(model.Config.output_path_results + 'results' + str(n_classes) + ".p","wb"))
