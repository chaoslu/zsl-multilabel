import cPickle
import itertools, operator
import numpy as np

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



if __name__ == "__main__":
    output_path_500 = "./result_lstm/20170816_214154/results500.p"
    output_path_1000 = "./result_lstm/20170822_145007/results1149.p"
    output_path_500_CNN = "./result_old/20170817_145548/results500.p"
    output_path_1000_CNN = "./result_old/20170822_152510/results1149.p"
    output_path_label_encodings = "./result_new/20170817_233522/results1149.p"
    path_for_freq_500 = output_path_500
    path_for_freq_1000 = output_path_1000
    summary_path = "./summary/"
    
    bin_size = 10
    n_cluster = 9
    
    x_for_freq_500 = cPickle.load(open(path_for_freq_500,"rb"))
    x_for_freq_1000 = cPickle.load(open(path_for_freq_1000,"rb"))
    labels_freqs_500,i2w_lb_500 = x_for_freq_500[5],x_for_freq_500[6]
    labels_freqs_1000,i2w_lb_1000 = x_for_freq_1000[5],x_for_freq_1000[6]
    
    
    lb_freq_all500,lb_freq_train500,lb_freq_test500 = labels_freqs_500
    lb_freq_all1000,lb_freq_train1000,lb_freq_test1000 = labels_freqs_1000
    
    
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
        vocab_list = [(itm,vocab_sm[itm]) for itm in vocab_sm]
        vocab_list = sorted(vocab_list, key = lambda t:t[1], reverse = True)
        vocab_list = [word for (word,freq) in vocab_list[:n_clusters] if word not in stop_words]
        
        return vocab_list
    
    
    def make_prevalance_tuples(data,lb_freq,i2w_lb):
        dt = np.array([data[i] for i in range(len(i2w_lb)) if (data[i] != 0) & (not np.isnan(data[i]))])
        lb = [i2w_lb[i] for i in range(len(i2w_lb)) if (data[i] != 0) & (not np.isnan(data[i]))]
        fq = [lb_freq[i2w_lb[i]] for i in range(len(i2w_lb)) if (data[i] != 0) & (not np.isnan(data[i]))]
        return (dt,lb,fq)
    
    
    def aggregate_classes(prev_tuple,bin_size):
        fq = prev_tuple[2]
        dt = prev_tuple[0]
        
        fq_bin_no = [freq/bin_size if freq < 500 else 10 for freq in fq]
        fq_bin_no = [freq * bin_size for freq in fq_bin_no]
        data_by_freq = {}
        for i,itm in enumerate(zip(dt,fq_bin_no)):
            if itm[1] in data_by_freq:
                data_by_freq[itm[1]].append(itm[0])
            else:
                data_by_freq[itm[1]] = [itm[0]]
        # import pdb; pdb.set_trace()
        data_by_prev = [(prev,np.mean(data_by_freq[prev])) for i,prev in enumerate(data_by_freq)]
        prevalance = [itm[0] for itm in data_by_prev]
        data = [itm[1] for itm in data_by_prev]
        return (prevalance,data)
    
    
    def make_label_frequency(labels,lb_freq):
        lb_freq_decoded = []
        for lb in labels:
            if lb in lb_freq:
                lb_freq_decoded.append(lb_freq[lb])
            else:
                lb_freq_decoded.append(0)
        return lb_freq_decoded
    
    
    def get_the_label_encodings(output_path,i2w,lb_vocab):
        x = cPickle.load(open(output_path,"rb"))
        label_encodings = x[2]
        label_cluster = []
        label_cluster_name = []
        label_words = []
        for i in range(len(i2w)):
            label_words.append(i2w[i])
            label_cluster.append(0)
            label_cluster_name.append('other')
            for j in range(len(lb_vocab)):
                if lb_vocab[j] in i2w[i]:
                    label_cluster[i] = j + 1
                    label_cluster_name[i] = lb_vocab[j]
        return label_encodings,label_words,label_cluster,label_cluster_name


    def get_data_for_visualisation(output_path,bin_size,lb_freq,i2w_lb):
        x = cPickle.load(open(output_path,"rb"))
    
        test_acc,pr_rcl_cls = x[1],x[2]
        prec_cls,rcl_cls,f1_cls,pr_rcl_crv = pr_rcl_cls
    
        precision_prev = make_prevalance_tuples(prec_cls,lb_freq,i2w_lb)
        recall_prev = make_prevalance_tuples(rcl_cls,lb_freq,i2w_lb)
        f1_prev = make_prevalance_tuples(f1_cls,lb_freq,i2w_lb)
        crv_prev = make_prevalance_tuples(pr_rcl_crv,lb_freq,i2w_lb)
        all_prev = (precision_prev,recall_prev,f1_prev,crv_prev)
    
        # set the bin of the prevalance
        precision_prev_agg = aggregate_classes(precision_prev,bin_size)
        recall_prev_agg = aggregate_classes(recall_prev,bin_size)
        f1_prev_agg = aggregate_classes(f1_prev,bin_size)
        crv_prev_agg = aggregate_classes(crv_prev,bin_size)
        all_prev_agg = (precision_prev_agg,recall_prev_agg,f1_prev_agg,crv_prev_agg)
        
        return test_acc, all_prev,all_prev_agg
    
    
    def get_data_old(output_path,bin_size,lb_freq,i2w_lb):
        x = cPickle.load(open(output_path,"rb"))
    
        k_accuracy,precision_recall_cls = x[2],x[3]
        prec_cls,rcl_cls,f1_cls,pr_rcl_crv = precision_recall_cls
    
        precision_prev = make_prevalance_tuples(prec_cls,lb_freq,i2w_lb)
        recall_prev = make_prevalance_tuples(rcl_cls,lb_freq,i2w_lb)
        f1_prev = make_prevalance_tuples(f1_cls,lb_freq,i2w_lb)
        crv_prev = make_prevalance_tuples(pr_rcl_crv,lb_freq,i2w_lb)
        all_prev = (precision_prev,recall_prev,f1_prev,crv_prev)
    
        # set the bin of the prevalance
        precision_prev_agg = aggregate_classes(precision_prev,bin_size)
        recall_prev_agg = aggregate_classes(recall_prev,bin_size)
        f1_prev_agg = aggregate_classes(f1_prev,bin_size)
        crv_prev_agg = aggregate_classes(crv_prev,bin_size)
        all_prev_agg = (precision_prev_agg,recall_prev_agg,f1_prev_agg,crv_prev_agg)
        return k_accuracy,all_prev,all_prev_agg
    
    test_acc_500, all_prev500, all_prev_agg500 = get_data_for_visualisation(output_path_500,bin_size,lb_freq_train500,i2w_lb_500)
    test_acc_1000, all_prev1000, all_prev_agg1000 = get_data_for_visualisation(output_path_1000,bin_size,lb_freq_train1000,i2w_lb_1000)
    test_acc_500_CNN, all_prev500_CNN, all_prev_agg500_CNN = get_data_old(output_path_500_CNN,bin_size,lb_freq_train500,i2w_lb_500)
    test_acc_1000_CNN, all_prev1000_CNN, all_prev_agg1000_CNN = get_data_old(output_path_1000_CNN,bin_size,lb_freq_train1000,i2w_lb_1000)
    
   
    

    
    
   
    # plot the boxplot for all the classes
    box_prec = [all_prev500[0][0],all_prev1000[0][0]]
    box_rcl = [all_prev500[1][0],all_prev1000[1][0]] 
    box_f1 = [all_prev500[2][0],all_prev1000[2][0]] 
    box_crv = [all_prev500[3][0],all_prev1000[3][0]] 
    datasets = [box_prec,box_rcl,box_f1,box_crv]
    label_names = ['precision','recall','f1','precision_recall_auc']
    

    fig1,axes = plt.subplots(ncols=4, sharey = True, figsize=(10,6))
    fig1.subplots_adjust(wspace=0)
    
    for ax,data,name in zip(axes,datasets,label_names):
        ax.boxplot(data)
        ax.set(xticklabels = ['500','1000'], xlabel=name)
    
    plt.show()
    
    
    # plot scatters for classes by prevalance
    fig2,axes2 = plt.subplots(nrows=2, ncols=2, figsize=(14,10), sharey=True)
    axes2[0,0].scatter(all_prev_agg500[0][0],all_prev_agg500[0][1])
    axes2[0,0].set_title('precision')
    axes2[0,1].scatter(all_prev_agg500[1][0],all_prev_agg500[1][1])
    axes2[0,1].set_title('recall')
    axes2[1,0].scatter(all_prev_agg500[2][0],all_prev_agg500[2][1])
    axes2[1,0].set_title('f1')
    axes2[1,1].scatter(all_prev_agg500[3][0],all_prev_agg500[3][1])
    axes2[1,1].set_title('precision_recall_auc')
    
    for ax in axes2.flatten():
        ax.set_xlabel('prevalence')
    fig2.suptitle("scores by prevalences of top 500 diaganosis")
    plt.show()
    
    # plot scatters for classes by prevalance
    fig3,axes3 = plt.subplots(nrows=2, ncols=2, figsize=(14,10), sharey=True)
    axes3[0,0].scatter(all_prev_agg1000[0][0],all_prev_agg1000[0][1])
    axes3[0,0].set_title('precision')
    axes3[0,1].scatter(all_prev_agg1000[1][0],all_prev_agg1000[1][1])
    axes3[0,1].set_title('recall')
    axes3[1,0].scatter(all_prev_agg1000[2][0],all_prev_agg1000[2][1])
    axes3[1,0].set_title('f1')
    axes3[1,1].scatter(all_prev_agg1000[3][0],all_prev_agg1000[3][1])
    axes3[1,1].set_title('precision_recall_auc')
    
    for ax in axes2.flatten():
        ax.set_xlabel('prevalence')
    fig3.suptitle("scores by prevalences of top 1000 diaganosis")
    plt.show()
    
    
    # plot scatter for top k test accuracy
    fig4 = plt.figure()
    

    line_500, = plt.plot(range(1,len(test_acc_500)+1),test_acc_500)
    line_1000, = plt.plot(range(1,len(test_acc_1000)+1),test_acc_1000)
    line_500_CNN, = plt.plot(range(1,len(test_acc_500_CNN)+1),test_acc_500_CNN)
    line_1000_CNN, = plt.plot(range(1,len(test_acc_1000_CNN)+1),test_acc_1000_CNN)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('accuracy of predicting within the top k classes')
    plt.legend([line_500,line_500_CNN,line_1000,line_1000_CNN],['acc top500','acc top500_CNN','acc top1000','acc top1000_CNN'])
    
    
    plt.show()

    
    # heat map
    
    
    
    
    '''
    gold_labels = incorrect_cls_decoded[0]
    decoded_labels = incorrect_cls_decoded[1]
    notes = incorrect_cls_decoded[2]
    
    nts = []
    # get rid of '<pad_zero>'
    for nt in notes:
        nt = nt.split()
        nt_word_only = [wd for wd in nt if wd != '<pad_zero>']
        nt_word_only = " ".join(nt_word_only)
        nts.append(nt_word_only)
    
    

 
    lb_freqT_gold = make_label_frequency(gold_labels,)
        

    incorrect_cls_decoded = zip(gold_labels,lb_freq_gold,decoded_labels,lb_freq_decoded,nts)
    incorrect_cls_decoded = sorted(incorrect_cls_decoded,key = lambda x:(x[1],x[0],x[3]))
          
    with open(summary_path + "RNN_Decoder_result.txt","w") as f:
        f.write("original label" + "\t\t" + "occurence" + "\t\t" + "decoded label" + "\t\t" + "occurence" + "\n")
        for i in range(len(notes)):
            f.write(incorrect_cls_decoded[i][0] + "\t\t" + str(incorrect_cls_decoded[i][1]) + "\t\t" + incorrect_cls_decoded[i][2]
            + "\t\t" + str(incorrect_cls_decoded[i][3]) + "\n")
    f.close()
    
    with open(summary_path + "RNN_Decoder_result_RareCases.txt","w") as f:
        f.write("original label" + "\t\t" + "occurence" + "\t\t" + "decoded label" + "\t\t" + "occurence" + "\n")
        for i in range(101):
            f.write(incorrect_cls_decoded[i][0] + "\t\t" + str(incorrect_cls_decoded[i][1]) + "\t\t" + incorrect_cls_decoded[i][2]
            + "\t\t" + str(incorrect_cls_decoded[i][3]) + "\n")
            f.write(incorrect_cls_decoded[i][4] + "\n\n\n")
    f.close()
    '''
    
    
    
    # get t-SNE of the CNN encoded labels
    '''
    lb_vocab = get_top_words_in_labels(lb_freq_all1000,n_cluster)
    label_encodings = get_the_label_encodings(output_path_label_encodings,i2w_lb_1000,lb_vocab)
    label_tSNE = TSNE(n_components=3).fit_transform(label_encodings[0])
    label_tSNE = [ label_tSNE[i] for i in range(label_tSNE.shape[0])]
    
    label_encodings_tuples = zip(label_tSNE,label_encodings[1],label_encodings[2],label_encodings[3])
    label_encodings_tuples = sorted(label_encodings_tuples, key = lambda t:t[3])
    
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
    # groups.append(group)
    
    fig0 = plt.figure()
    colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8']
    for i,grp in enumerate(groups):
            tsne,_,_,_ = zip(*grp)
            tsne = list(tsne)
            tsne_x,tsne_y,tsne_z = zip(*tsne)
            plt.scatter(list(tsne_x),list(tsne_y),list(tsne_z))
    plt.show()
    '''