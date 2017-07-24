import cPickle
from collections import OrderedDict

from data_utils import make_idx_sentences,make_idx_data,split_train,load_bin_vec
from data_utils import add_unknown_words,get_vocab_emb,diag_narrow,build_vocab


if __name__ == "__main__":

	loc = './data/'
	freq_lbd_idx = 30
	w2v_file = 'GoogleNews-vectors-negative300.bin'
	train,train_labels,lb_lst = cPickle.load(open(loc + 'pre_clipped.p','rb'))

	# only use the notes and diagnoses more than certain amount
	lb_freq = OrderedDict(lb_lst)
	freq_lbd = lb_lst[freq_lbd_idx - 1][1]
	train_notes,train_labels = diag_narrow(train,train_labels,lb_freq,freq_lbd)
	cPickle.dump([train_notes,train_labels,lb_lst],open(loc + 'clipped_data_' + str(freq_lbd_idx) + '.p',"wb"))

	# make labels natural language
	train_seman = []
	for labels in train_labels:
		seman = ' '.join(labels)
		seman = seman.split()
		train_seman.append(seman)

	# the corpus of both text and semantic space 
	text_and_seman = train_notes + train_seman   

	# build dictionary for semantic along and the whole
	word2idx, idx2word, _ = build_vocab(text_and_seman)
	w2i_lb, i2w_lb, _ = build_vocab(train_labels)

	# lb_vocab, lb_wd_fq = label_vocab(train_labels,valid_ngram)

	train = make_idx_data(train_notes, train_labels,train_seman, word2idx, w2i_lb,word2idx)
	train,test = split_train(train)
	train,dev = split_train(train)

	# build the word embedding for dataset
	print "loading word2vec vectors...",
	w2v = load_bin_vec( loc + w2v_file, word2idx)

	# add_unknown_words(w2v_sm, w2i_sm)
	add_unknown_words(w2v, word2idx)

	W = get_vocab_emb(w2v,idx2word)

	everything = [train, dev, test, W, idx2word, word2idx, w2i_lb, i2w_lb]
	cPickle.dump(everything, open(loc + 'everything' + str(freq_lbd_idx) + '.p', "wb"))
#    print "dataset created!"
