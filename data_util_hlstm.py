import numpy as np
import tensorflow as tf
import cPickle
import nltk
from nltk.tokenize import word_tokenize

from collections import defaultdict
from collections import OrderedDict

notes_path = 'tok_hpi_clean_glove'
dx_path = 'tok_dx_clean_glove'
vector_path = 'vectors_my.txt'
using_glove = True


def load_data(loc='./data/'):

    notes = []
    diagonosis = []

    with open(loc + notes_path, 'rb') as f:
        for line in f:
            notes.append(line.strip())

    with open(loc + dx_path, 'rb') as f:
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

    i2w[n_words] = '<GO>'
    w2i['<GO>'] = n_words
    n_words = n_words + 1

    i2w[n_words] = '<EOS>'
    w2i['<EOS>'] = n_words
    n_words = n_words + 1

    i2w[n_words] = '<pad_zero>'
    w2i['<pad_zero>'] = n_words
    n_words = n_words + 1

    return w2i,i2w


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


'''
def label_vocab(text,valid_ngram):
    vocab = []
    vocab_dict = {}
    for labels in text:
        for label in labels:
            if label not in valid_ngram:
                splitted = label.split('_')

                for lb in splitted:
                    if lb not in vocab:
                        vocab.append(lb)
                        vocab_dict[lb] = 0
                    else:
                        vocab_dict[lb] += 1
    for uni in vocab:
        if '-' in uni:
            uni_k = np.str(uni)
            uni_k.replace('-','_')
            if uni not in valid_ngram:
                vocab_dict.pop(uni)
                vocab.remove(uni)

                splitted = label.split('_')

                for lb in splitted:
                    if lb not in vocab:
                        vocab.append(lb)
                        vocab_dict[lb] = 0
                    else:
                        vocab_dict[lb] += 1
    w2i = OrderedDict(sorted(vocab_dict.items(), key= lambda t:t[1], reverse=True))
    return vocab,w2i
'''


'''
def make_labels_idx(labels):
    d = {}
    setlabels = set(labels)
    count = 0
    for l in setlabels:
        d[l] = count
        count += 1
    idx_lables = [d[l] for l in labels]

    return idx_lables,d
'''


def make_idx_sentences(text,word2idx,is_seman=False):
    id_sentence = []
    if is_seman:
        id_sentence = [make_idx_sentences(sen,word2idx)[0] for sen in text]
        sm_max = max(make_idx_sentences(sen,word2idx)[1] for sen in text)
    else:
        for sent in text:
            id_sentence.append([word2idx[word] for word in sent])
        sm_max = max([len(s) for s in text])
    return id_sentence,sm_max


def make_idx_data(train,train_labels,train_seman,word2idx,w2i_lb,w2i_sm):

    n_lb = len(w2i_lb)

    tr_id,mx_len = make_idx_sentences(train,word2idx)
    tr_id_l,_ = make_idx_sentences(train_labels,w2i_lb)
    tr_id_sm,mx_len_sm = make_idx_sentences(train_seman,w2i_sm,True)

    tr_lb = []
    # transfer labels from indice format to multi-hot format
    for lbs in tr_id_l:
        lb = [0] * n_lb
        for l in lbs:
            lb[l] = 1
        tr_lb.append(lb)

    tr = (tr_id,tr_lb,tr_id_sm)

    return tr,mx_len,mx_len_sm


def split_train(train,dev_portion=0.1):
    lz = len(train[0])
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
    for line in fname:
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

    loc = './data/'
    freq_lbd_idx = 1000
    w2v_file = 'GoogleNews-vectors-negative300.bin'
    # valid_ngram = cPickle.load(open('./valid_gram.p','rb'))

    use_glove = ''
    if using_glove:
        use_glove = 'glove_'
        w2v_file = vector_path

    print "preparing data...",  
    notestext, labeltext = load_data()
    notestext = preprocess(notestext)

    train_notes, train_labels = prepare_data(notestext,labeltext)
    del notestext, labeltext


    # test_labels,lb_test = make_labels_idx(test_labels)
    # text = train + test
    # labels = lb_train + lb_test

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
    cPickle.dump([train_notes,train_labels,lb_lst],open('./data/' + use_glove + 'pre_clipped_hlstm.p',"wb"))

    # only use the notes and diagnoses more than certain amount
    freq_lbd = lb_lst[freq_lbd_idx - 1][1]
    train_notes,train_labels = diag_narrow(train_notes,train_labels,lb_freq,freq_lbd)
    cPickle.dump([train_notes,train_labels,lb_lst],open('./data/' + use_glove + 'clipped_data_hlstm_' + str(freq_lbd_idx) + '.p',"wb"))


    label_lexon = []
    for labels in train_labels:
        lexon = ' '.join(labels)
        lexon = lexon.split()
        label_lexon.append(lexon)

    # the corpus of both text and semantic space 
    text_and_seman = train_notes + label_lexon   

    # build dictionary for semantic along and the whole
    word2idx, idx2word, _ = build_vocab(text_and_seman)
    w2i_lb, i2w_lb, _ = build_vocab(train_labels)
    w2i_sm, i2w_sm, _ = build_vocab(label_lexon)

    # do the first level padding for the labels, which pads all the patient to
    # uniform number of labels
    max_diags = max([len(itm) for itm in train_labels])
    train_seman = []

    for lbs in train_labels:
        labels_to_pad = [itm.split() for itm in lbs]
        while len(labels_to_pad) < max_diags:
            labels_to_pad.append([])
        train_seman.append(labels_to_pad)

    # make data ids and split
    train,mx_len,mx_len_sm = make_idx_data(train_notes, train_labels,train_seman, word2idx, w2i_lb,w2i_sm)
    train,test = split_train(train)
    train,dev = split_train(train)

    # labels frequency from trianing and test
    lb_freq_train = label_frequency(train[1],i2w_lb)
    lb_freq_test = label_frequency(test[1],i2w_lb)

    # build the word embedding for dataset
    print "loading word2vec vectors...",

    if using_glove:
        w2v = load_text_vec(loc + w2v_file, word2idx)
    else:
        w2v = load_bin_vec(loc + w2v_file, word2idx)

    add_unknown_words(w2v, word2idx)
    W = get_vocab_emb(w2v,idx2word)

    # add special token to both the notes vocabulary and the semantic vocabulary
    word2idx,idx2word = add_special_token(word2idx,idx2word)
    w2i_sm, i2w_sm = add_special_token(w2i_sm,i2w_sm)

    vocab_size = len(word2idx)
    vocab_size_sm = len(w2i_sm)

    Wemb = np.zeros((vocab_size,300))
    Wemb[:vocab_size-3] = W

    # build mapping between notes dictionary and the seman dictionary
    dict_s2n = np.zeros((vocab_size_sm,1),dtype=np.int32)
    for idx in i2w_sm:
        dict_s2n[idx] = word2idx[i2w_sm[idx]]

    ConfigInfo = {}
    ConfigInfo['n_diagnosis'] = max_diags
    ConfigInfo['max_len'] = mx_len
    ConfigInfo['max_len_sm'] = mx_len_sm + 1  # for the 'GO' or 'EOS'
    ConfigInfo['vocab_size'] = vocab_size

    lb_lst = dict(lb_lst)
    everything = [train, dev, test, Wemb, idx2word, word2idx, i2w_lb, i2w_sm,dict_s2n,ConfigInfo,(lb_lst,lb_freq_train,lb_freq_test)]
    cPickle.dump(everything, open('./data/' + use_glove + 'hlstm_everything' + str(freq_lbd_idx) + '.p', "wb"))
#    print "dataset created!"
