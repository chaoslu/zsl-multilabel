#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:59:59 2017

@author: shuchaolu
"""
import cPickle

def build_replace_list(fname):
    
    replst_uni = {}
    replst_ngm = {}

    with open(fname,'rb') as f:
        for line in f:
            item = line.split(':')
            item[0] = item[0].strip()
            item[1] = item[1].strip()
            if ' ' not in item[0]:
                replst_uni[item[0]] = item[1]
            else:
                replst_ngm[item[0]] = item[1]
    return replst_uni,replst_ngm
    
            
if __name__=="__main__":
    
    valid_ngram = cPickle.load(open('./valid_gram.p','rb'))
    
    notes = []
    diagnose = []
    
    fdiag_v = './abbrev_vld'
    fdiag_iv = './abbrev_ivld'
    fnts_v = './abbrev_vld_data'
    fnts_iv = './abbrev_ivld_data'
    loc = './data'
    
    rpl_uni_nts_v,rpl_ngm_nts_v = build_replace_list(fnts_v)
    rpl_uni_diag_v,rpl_ngm_diag_v = build_replace_list(fdiag_v)
    rpl_uni_nts_iv,_ = build_replace_list(fnts_iv)
    rpl_uni_diag_iv,_ = build_replace_list(fdiag_iv)
    
    rpl_uni_nts = dict(rpl_uni_nts_v)
    rpl_ngm_nts = dict(rpl_ngm_nts_v)
    rpl_uni_diag = dict(rpl_uni_diag_v)
    rpl_ngm_diag = dict(rpl_ngm_diag_v)
    
    rpl_uni_nts.update(rpl_uni_nts_iv)
    rpl_ngm_nts.update(valid_ngram)
    rpl_uni_diag.update(rpl_uni_diag_iv)
    rpl_ngm_diag.update(valid_ngram)
    
    cPickle.dump([rpl_uni_nts,rpl_ngm_nts,rpl_uni_diag,rpl_ngm_diag], open('rpl_all.p','w'))
    
    
    with open(loc + 'tok_hpi_clean','rb') as f:
        for line in f:
            line = line.strip()
            for phrs in rpl_ngm_nts:
                line = line.replace(phrs,rpl_ngm_nts[phrs])
            for phrs in rpl_uni_nts:
                line = line.replace(' ' + phrs + ' ', ' ' + rpl_uni_diag[phrs] + ' ')
            notes.append(line)
    
    
    with open(loc + 'tok_dx_clean','rb') as f:
        for line in f:
            line = line.strip()
            for phrs in rpl_ngm_diag:
                line = line.replace(phrs,rpl_ngm_diag[phrs]) 
            splitted = line.split('@')
            splitted = [x.strip() for x in splitted if x != '']
            for i,lb in enumerate(splitted):
                s_lb = lb.split()
                for j,slb in enumerate(s_lb):
                    if slb in rpl_uni_diag:
                        s_lb[j] = rpl_uni_diag[slb]
                new_subline = ' '.join(s_lb)
                splitted[i] = new_subline
            newline = ' @ '.join(splitted)
              
            diagnose.append(newline)

    with open(loc + 'tok_hpi_rpl','w') as f:
        for nt in notes:
            f.write(nt + '\n')
    
    with open(loc + 'tok_dx_rpl','w') as f:
        for dg in diagnose:
            f.write(dg + '\n')
