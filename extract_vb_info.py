#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:02:14 2017

@author: shuchao
"""

import cPickle
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == "__main__":
    
    #output_path_500_mixed = "./mixed_single_result_old/results_only/20170915_191447/results506.p"
    #output_path_500_lstm_mixed = "./mixed_result_lstm/results_only/20170918_131633/results506.p"
    path = "./data/lstm_everything_new500.p"
    #output_path_500_lstm = "./glove_result_lstm/results_only/20170918_131022/results506.p"

    #x_mixed_freq_500 = cPickle.load(open(output_path_500_mixed,"rb"))
    #x_mixed_freq_500_lstm =cPickle.load(open(output_path_500_lstm_mixed,"rb"))
    data = cPickle.load(open(path,"rb"))
    cPickle.dump(data[5:],open("./data/lstm_no_dt500.p","wb"))