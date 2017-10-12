import numpy as np
import tensorflow as tf
import cPickle
import nltk
import argparse
from nltk.tokenize import word_tokenize

from collections import defaultdict
from collections import OrderedDict




def load_data(loc,locs):

    notes = []
    diagonosis = []

    with open(loc + locs[0], 'rb') as f:
        for line in f:
            notes.append(line.strip())

    with open(loc + locs[1], 'rb') as f:
        for line in f:
            diagonosis.append(line.strip())

    return notes,diagonosis


def prepare_data(text,label):

        labels = [t.split('@') for t in label]
        labels = [[d.strip() for d in diag if len(d) > 0] for diag in labels]
        # labels = [[d.replace(" ","_") for d in diag] for diag in labels]

        X = [t.split() for t in text]

        return X,labels


def preprocess(text):
    """
    Preprocess text for encoder
    """
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for t in text:
        sents = sent_detector.tokenize(t)
        result = ''
        for s in sents:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    return X


def build_vocab(text):
    vocab_freq = defaultdict(float)
    for sent in text:
        # words = sent.split()
        for word in sent:
            if word in vocab_freq:
                vocab_freq[word] += 1
            else:
                vocab_freq[word] = 1

    word2idx = defaultdict(int)
    idx2word = defaultdict(int)
    count = 0

    for word in vocab_freq:
        word2idx[word] = count
        idx2word[count] = word
        count += 1

    return word2idx, idx2word, vocab_freq


def diag_narrow(train,train_labels,vocab_freq,freq_lbd):
    remain_list = [word for word in vocab_freq if vocab_freq[word] >= freq_lbd]

    train_set = zip(train,train_labels)
    new_set = []

    for i,itm in enumerate(train_set):
        removed = []
        for wd in itm[1]:
            if wd in remain_list:
                removed.append(wd)
        if len(removed) != 0:
            new_set.append((train_set[i][0],removed))
    tp = zip(*new_set)
    train = list(tp[0])
    train_labels = list(tp[1])

    return train,train_labels


def make_idx_sentences(text,word2idx):
    id_sentence = []
    for sent in text:
        # words = sent.split()
        id_sentence.append([word2idx[word] for word in sent])
    return id_sentence


def make_idx_data(train,train_labels,train_seman,nl_lb_dict,word2idx,w2i_lb,w2i_sm):
    n_lb = len(w2i_lb)
    
    tr_id = make_idx_sentences(train,word2idx)
    tr_id_l = make_idx_sentences(train_labels,w2i_lb)
    tr_id_sm = make_idx_sentences(train_seman,w2i_sm)
    tr_id_cls = make_idx_sentences(nl_lb_dict,word2idx)
    
    tr_lb = []
    # transfer labels from indice format to multi-hot format

    for lbs in tr_id_l:
        lb = [0] * n_lb
        for l in lbs:
            lb[l] = 1
        tr_lb.append(lb)

    tr = (tr_id,tr_lb,tr_id_sm)

    return tr,tr_id_cls


def split_train(train,dev_portion=0.1):
    lz = len(train[0])
    np.random.seed(123)
    idx_dev = np.random.permutation(lz)
    n_train = int(np.round(lz * (1 - dev_portion)))
    
    train_sent = [train[0][s] for s in idx_dev[:n_train]]
    train_label = [train[1][s] for s in idx_dev[:n_train]]
    train_seman = [train[2][s] for s in idx_dev[:n_train]]
    
    dev_sent = [train[0][s] for s in idx_dev[n_train:]]
    dev_label = [train[1][s] for s in idx_dev[n_train:]]
    dev_seman = [train[2][s] for s in idx_dev[n_train:]]

    return (train_sent,train_label,train_seman), (dev_sent,dev_label,dev_seman)


def label_frequency(data,i2w):
    lb_freq_list = {}
    for lb_oh in data:
        lb = np.argmax(lb_oh)
        if i2w[lb] in lb_freq_list:
            lb_freq_list[i2w[lb]] += 1
        else:
            lb_freq_list[i2w[lb]] = 1
    return lb_freq_list


def load_bin_vec(fname,  vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    # word_vecs = {}
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word = word.lower()
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
#          word_vecs,
    return word_vecs


def load_text_vec(fname,  vocab):
    vectors = {}
    with open(fname, "r") as f:
        for line in f:
            vals = line.rstrip().split(' ')
            if vals[0] in vocab:
                vectors[vals[0]] = [float(x) for x in vals[1:]]

    return vectors


def add_unknown_words(word_vecs,word2idx,k=300):
    for word in word2idx:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def get_vocab_emb(word_vecs,idx2word,k=300):
    vocab_size = len(word_vecs)
    W = np.zeros([vocab_size,k])
    for idx in range(vocab_size):
        W[idx] = word_vecs[idx2word[idx]]
    # W = tf.constant(W)
    return W



if __name__ == "__main__":   

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_freq', default=500, type=int)
    parser.add_argument('--label_type', default='multi', type=str)
    parser.add_argument('--data', default='clean', type=str)
    args = parser.parse_args()

    loc = './data/'
    
    w2v_file_m = 'GoogleNews-vectors-negative300.bin'
    w2v_file_g = 'vectors_my.txt'
    affx = ''  # about the saved adat filename affix

    if args.data == 'clean':
        notes_path = 'tok_hpi_clean'
        dx_path = 'tok_dx_clean'
        affx = '_new'
    else:
        notes_path = 'tok_hpi_rpl'
        dx_path = 'tok_dx_rpl'
    

    freq_lbd_idx = args.label_freq

    print "preparing data...",  
    notestext, labeltext = load_data(loc,(notes_path,dx_path))
    notestext = preprocess(notestext)

    train_notes, train_labels = prepare_data(notestext,labeltext)
    del notestext, labeltext

    # build the dictionary for labels
    w2i_lb, i2w_lb, lb_freq = build_vocab(train_labels)
    lb_vb = OrderedDict(sorted(lb_freq.items(), key = lambda t:t[1], reverse = True))
    lb_lst = [(wd,lb_vb[wd]) for wd in lb_vb]

    # get rid of those single cases
    sc_lb = [wd for wd in lb_freq if lb_freq[wd] == 1]
    train_set = zip(train_notes,train_labels)
    for i,itm in enumerate(train_set):
        if len(set(sc_lb + itm[1])) != len(sc_lb + itm[1]):
            train_set.remove(itm)
    tp = zip(*train_set)
    train_notes = list(tp[0])
    train_labels = list(tp[1])

    if args.label_type != 'multi':
        # only remian the first diagnosis for each patient
        train_labels = [[labels[0]] for labels in train_labels]

        # update the label dictionary and the frequency list
        w2i_lb, i2w_lb, lb_freq = build_vocab(train_labels)  
        lb_vb = OrderedDict(sorted(lb_freq.items(), key = lambda t:t[1], reverse = True))
        lb_lst = [(wd,lb_vb[wd]) for wd in lb_vb]

    # only use the notes and diagnoses more than certain amount
    freq_lbd = lb_lst[freq_lbd_idx - 1][1]
    train_notes,train_labels = diag_narrow(train_notes,train_labels,lb_freq,freq_lbd)

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

    # make labels dictionary natural language
    nl_lb_dict = []
    for clss in w2i_lb:
        cls_in_wds = clss.split()
        nl_lb_dict.append(cls_in_wds)


    train,nl_clss = make_idx_data(train_notes, train_labels,train_seman, nl_lb_dict, word2idx, w2i_lb,word2idx)
    train,test = split_train(train)
    train,dev = split_train(train)
    lb_freq_train = label_frequency(train[1],i2w_lb)
    lb_freq_test = label_frequency(test[1],i2w_lb)


    # get the hyperparameters
    max_len = max([len(nts) for nts in train[0] + test[0] + dev[0]])
    max_len_sm = max([len(sm) for sm in train[2] + test[2] + dev[2]])
    vocab_size = len(word2idx)
    ConfigInfo = {}
    ConfigInfo['max_len'] = max_len
    ConfigInfo['max_len_sm'] = max_len_sm
    ConfigInfo['vocab_size'] = vocab_size

    # build the word embedding for dataset
    print "loading word2vec vectors...",
    w2v_g = load_text_vec(loc + w2v_file_g, word2idx)
    w2v_m = load_bin_vec(loc + w2v_file_m, word2idx)

    add_unknown_words(w2v_g, word2idx)
    add_unknown_words(w2v_m, word2idx)

    W_g = get_vocab_emb(w2v_g,idx2word)
    W_m = get_vocab_emb(w2v_m,idx2word)
    # W_sm = get_vocab_emb(w2v_sm,i2w_sm)

    lb_lst = dict(lb_lst)
    everything = [train, dev, test, nl_clss, W_g, W_m, idx2word, word2idx, w2i_lb, i2w_lb,ConfigInfo,(lb_lst,lb_freq_train,lb_freq_test)]
    cPickle.dump(everything, open('./data/everything' + affx + args.label_type + str(freq_lbd_idx) + '.p', "wb"))
#    print "dataset created!"
