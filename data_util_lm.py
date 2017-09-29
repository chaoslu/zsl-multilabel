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


def add_special_token(w2i,i2w):
    # add special token called <pad_zero> and <GO> and <EOS>
    n_words = len(i2w)

    i2w[n_words] = '<pad_zero>'
    w2i['<pad_zero>'] = n_words
    n_words = n_words + 1

    return w2i,i2w


def diag_narrow(train_labels,vocab_freq,freq_lbd):
    remain_list = [word for word in vocab_freq if vocab_freq[word] >= freq_lbd]

    new_labels = []

    for i,itm in enumerate(train_labels):
        removed = []
        for wd in itm:
            if wd in remain_list:
                removed.append(wd)
        if len(removed) != 0:
            new_labels.append(removed)

    return new_labels


def make_idx_sentences(text,word2idx):
    id_sentence = []
    for sent in text:
        # words = sent.split()
        id_sentence.append([word2idx[word] for word in sent])
    return id_sentence


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
    unk_words = []
    for word in word2idx:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)
            unk_words.append(word)
    return unk_words


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
    parser.add_argument('--data', default='clean', type=str)
    args = parser.parse_args()

    loc = './data/'

    w2v_file_m = 'GoogleNews-vectors-negative300.bin'
    w2v_file_g = 'vectors_my.txt'
    affx = ''  # about the saved adat filename affix

    if args.data == 'clean':
        dx_path = 'tok_dx_clean'
        affx = '_new'
    else:
        dx_path = 'tok_dx_rpl'

    freq_lbd_idx = args.label_freq

    print "preparing data...",  
    notestext,labeltext = load_data(loc,(dx_path,dx_path))

    _,train_labels = prepare_data(notestext,labeltext)
    del notestext, labeltext

    # build the dictionary for labels
    w2i_lb, i2w_lb, lb_freq = build_vocab(train_labels)
    lb_vb = OrderedDict(sorted(lb_freq.items(), key = lambda t:t[1], reverse = True))
    lb_lst = [(wd,lb_vb[wd]) for wd in lb_vb]

    # get rid of those single cases
    sc_lb = [wd for wd in lb_freq if lb_freq[wd] == 1]
    for i,itm in enumerate(train_labels):
        if len(set(sc_lb + itm)) != len(sc_lb + itm):
            train_labels.remove(itm)

    # only remian the first diagnosis for each patient
    train_labels = [[labels[0]] for labels in train_labels]  

    # update the label dictionary and the frequency list
    w2i_lb, i2w_lb, lb_freq = build_vocab(train_labels)  
    lb_vb = OrderedDict(sorted(lb_freq.items(), key = lambda t:t[1], reverse = True))
    lb_lst = [(wd,lb_vb[wd]) for wd in lb_vb]

    # only use the notes and diagnoses more than certain amount
    freq_lbd = lb_lst[freq_lbd_idx - 1][1]
    train_labels = diag_narrow(train_labels,lb_freq,freq_lbd)

    # make labels natural language
    def expand_labels(train_labels):
        train_seman = []
        for labels in train_labels:
            seman = ' '.join(labels)
            seman = seman.split()
            train_seman.append(seman)
        return train_seman


    train_seman = expand_labels(train_labels)
    # build dictionary for semantic along and the whole

    w2i_lb, i2w_lb, _ = build_vocab(train_labels)
    w2i_sm, i2w_sm, _ = build_vocab(train_seman)

    # transform sentences into indices sequence
    train = make_idx_sentences(train_seman,w2i_sm)
    all_labels = [[lb] for lb in w2i_lb]
    test_seman = expand_labels(all_labels)
    all_labels_idx = make_idx_sentences(test_seman,w2i_sm)


    # build the word embedding for dataset
    print "loading word2vec vectors...",

    w2v_g = load_text_vec(loc + w2v_file_g, w2i_sm)
    w2v_m = load_bin_vec(loc + w2v_file_m, w2i_sm)

    unk_g = add_unknown_words(w2v_g, w2i_sm)
    unk_m = add_unknown_words(w2v_m, w2i_sm)

    W_g = get_vocab_emb(w2v_g,i2w_sm)
    W_m = get_vocab_emb(w2v_m,i2w_sm)
    # W_sm = get_vocab_emb(w2v_sm,i2w_sm)

    # words in labels set as unknown
    unk_lb_word_g = [word for word in unk_g if word in w2i_sm]
    unk_lb_word_m = [word for word in unk_m if word in w2i_sm]

    # add special token to both the notes vocabulary and the semantic vocabulary

    w2i_sm, i2w_sm = add_special_token(w2i_sm,i2w_sm)
    vocab_size_sm = len(w2i_sm)

    Wemb_g = np.random.uniform(-0.25,0.25,(vocab_size_sm,300))
    Wemb_m = np.random.uniform(-0.25,0.25,(vocab_size_sm,300))
    Wemb_g[:vocab_size_sm-1] = W_g
    Wemb_m[:vocab_size_sm-1] = W_m


    # get the hyperparameters
    max_len_sm = max([len(sm) for sm in train])   # 'GO' or 'EOS' takes up a postition
    ConfigInfo = {}
    ConfigInfo['max_len_sm'] = max_len_sm
    ConfigInfo['vocab_size'] = vocab_size_sm

    everything = [train,all_labels_idx,Wemb_g, Wemb_m,i2w_lb,i2w_sm,ConfigInfo]
    cPickle.dump(everything, open('./data/everything_lm' + affx + str(freq_lbd_idx) + '.p', "wb"))
    # cPickle.dump(everything[5:] + [unk_lb_word_m,unk_lb_word_g,unk_word_m,unk_word_g], open('./data/lstm_no_dt' + affx + str(freq_lbd_idx) + '.p', "wb"))
    # print "dataset created!"
