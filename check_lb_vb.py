#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:20:36 2017

@author: shuchao
"""

import cPickle
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == "__main__":
    
    path = "./data/lstm_no_dt500.p"
    #output_path_500_lstm = "./glove_result_lstm/results_only/20170918_131022/results506.p"

    #x_mixed_freq_500 = cPickle.load(open(output_path_500_mixed,"rb"))
    #x_mixed_freq_500_lstm =cPickle.load(open(output_path_500_lstm_mixed,"rb"))
    data = cPickle.load(open(path,"rb"))