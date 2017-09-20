#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 11:20:22 2017

@author: shuchaolu
"""

import cPickle
import itertools, operator
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == "__main__":
    
    output_path_500_mixed = "./mixed_single_result_old/results_only/20170915_191447/results506.p"
    output_path_500_lstm_mixed = "./mixed_result_lstm/results_only/20170918_131633/results506.p"
    output_path_500 = "./glove_single_result_old/results_only/20170915_191331/results506.p"
    output_path_500_lstm = "./glove_result_lstm/results_only/20170918_131022/results506.p"

    x_mixed_freq_500 = cPickle.load(open(output_path_500_mixed,"rb"))
    x_mixed_freq_500_lstm =cPickle.load(open(output_path_500_lstm_mixed,"rb"))
    x_for_freq_500 = cPickle.load(open(output_path_500,"rb"))
    x_for_freq_500_lstm = cPickle.load(open(output_path_500_lstm,"rb"))
    
    cnn_encodings,labels,i2w_lb,lb_freq = x_for_freq_500[3],x_for_freq_500[4],x_for_freq_500[5],x_for_freq_500[6]
    cnn_encodings_lstm,labels_lstm,lb_freq_lstm,i2w_lb_lstm = x_for_freq_500_lstm[4],x_for_freq_500_lstm[5],x_for_freq_500_lstm[8],x_for_freq_500_lstm[9]
    
    def do_tsne_plot(enc,lb,i2w,freq,model_name,n_cluster=9):
        cnn_encodings_train = enc[0]
        cnn_encodings_test = enc[1]
        labels_train =lb[0]
        labels_test = lb[1]
        
        lb_freq_all500,lb_freq_train500,lb_freq_test500 = freq
        labels_dense = np.argmax(labels_train,axis=1)
        
        def get_top_words_in_labels(lb_freq,n_clusters):
            stop_words = ['acute','pain','of','with','leg','unspecified','and','finger','and','mal','in','status','hand','follow','up','accidental']
            vocab_sm = {}
            for label in lb_freq:
                label_words = label.split()
                for word in label_words:
                    if word in vocab_sm:
                        vocab_sm[word] += lb_freq[label]
                    else:
                        vocab_sm[word] = lb_freq[label]
            vocab_list = [(itm,vocab_sm[itm]) for itm in vocab_sm if not itm in stop_words]
            vocab_list = sorted(vocab_list, key = lambda t:t[1], reverse = True)
            vocab_list = [word for (word,freq) in vocab_list[:n_clusters] if word not in stop_words]
            
            return vocab_list
     
        def get_the_label_encodings(label_encodings,labels,i2w,lb_vocab):
            label_cluster = []
            label_cluster_name = []
            label_words = []
            for i in range(len(labels)):
                label_words.append(i2w[labels_dense[i]])
                label_cluster.append(0)
                label_cluster_name.append('other')
                for j in range(len(lb_vocab)):
                    if lb_vocab[j] in i2w[labels_dense[i]]:
                        label_cluster[i] = j + 1
                        label_cluster_name[i] = lb_vocab[j]
            return label_encodings,label_words,label_cluster,label_cluster_name
    
        
        lb_vocab = get_top_words_in_labels(lb_freq_all500,n_cluster)
        
        
        label_encodings = get_the_label_encodings(cnn_encodings_train[:7000,:],labels_dense[:7000],i2w,lb_vocab)
        label_tSNE = TSNE(n_components=2).fit_transform(label_encodings[0])
        label_tSNE = [ label_tSNE[i] for i in range(label_tSNE.shape[0])]
        
        label_encodings_tuples = zip(label_tSNE,label_encodings[1],label_encodings[2],label_encodings[3])
        label_encodings_tuples = sorted(label_encodings_tuples, key = lambda t:t[2])
        
        # plot the t-SNE of the label encodings:
        groups = []
        group = []
        for i in range(len(label_encodings_tuples)):
            if i==0:
                group = [label_encodings_tuples[i]]
            else:
                if label_encodings_tuples[i][2] != label_encodings_tuples[i-1][2]:
                    groups.append(group)
                    group = [label_encodings_tuples[i]]
                else:
                    group.append(label_encodings_tuples[i])
        groups.append(group)
        groups = groups[1:n_cluster+1]
        
        fig = plt.figure(figsize=(12,10))
        scats = []
        draw = [True,True,True,True,True,True,True,True,True]
        group_names = ['chest','abd','fever','back','closed','laceration','headache','viral','contusion']
        using_group_names = [name for i,name in enumerate(group_names) if draw[i]]
        
        def draw_tsne(grp):
            tsne,_,_,_ = zip(*grp)
            tsne = list(tsne)
            tsne_x,tsne_y = zip(*tsne)
            #tsne_x,tsne_y,tsne_z = zip(*tsne)
            scat = plt.scatter(list(tsne_x),list(tsne_y),alpha=0.3)#,list(tsne_z))
            
            return scat
        
        for i,grp in enumerate(groups):
            if draw[i]:
                scat = draw_tsne(grp)
                scats.append(scat)
                
        plt.title("CNN-encoding for top " + str(n_cluster) + " groups of diagnosis -- " + model_name)
        leg = plt.legend(scats,using_group_names,bbox_to_anchor=(1.05, 1), loc=2,prop={'size':15}, borderaxespad=0.)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
        plt.show()
    
    
    do_tsne_plot(cnn_encodings,labels,i2w_lb,lb_freq,"CNN")
    do_tsne_plot(cnn_encodings_lstm,labels_lstm,i2w_lb_lstm,lb_freq_lstm,"CNN_RNN")
    
     # plot scatter for top k test accuracy
    
    acc_500 = x_mixed_freq_500[1]
    acc_500_lstm = x_mixed_freq_500_lstm[1]
    acc_500_glove = x_for_freq_500[1]
    acc_500_glove_lstm = x_for_freq_500_lstm[1]
   
    
    fig1 = plt.figure()
    line_500_CNN, = plt.plot(range(1,len(acc_500)+1),acc_500)
    line_500_CNN-RNN, = plt.plot(range(1,len(acc_500_lstm)+1),acc_500_lstm)
    line_500_glove_CNN, = plt.plot(range(1,len(acc_500_glove)+1),acc_500_glove)
    line_500_glove_CNN-RNN, = plt.plot(range(1,len(acc_500_glove_lstm)+1),acc_500_glove_lstm)
   
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('accuracy of predicting within the top k classes over top500 diagnosis')
    plt.legend([line_500,line_500_CNN,line_1000,line_1000_CNN],['CNN-WordVec','CNN-RNN-WordVec','CNN-Glove','CNN-RNN-Glove'])
    
    
    plt.show()
    