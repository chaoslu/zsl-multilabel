#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:19:34 2017

@author: shuchaolu
"""

import numpy as np

notes = []
diagonosis = []
loc = './'

with open(loc + 'tok_hpi', 'rb') as f:
	for line in f:
		notes.append(line.strip())

with open(loc + 'tok_dx', 'rb') as f:
	for line in f:
         if not line.strip() == 'DX':
             diagonosis.append(line.strip())

new_notes = []
new_diag = []
for nt,dg in zip(notes,diagonosis):
    if (len(nt) != 0) & ('?' not in dg):
        new_notes.append(nt)
        new_diag.append(dg)
        
        
with open(loc + 'tok_hpi_clean','w') as f:
    for nt in new_notes:
        f.write(nt+'\n')
    
with open(loc + 'tok_dx_clean','w') as f:
    for dg in new_diag:
        f.write(dg+'\n')
        