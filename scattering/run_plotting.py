#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:44:29 2020
The code that produces the plots in https://doi.org/10.1039/D0FD00106F
still requires some enhanced generalization
will do in due course.

@author: afouda
"""

import plotting
import numpy as np
import matplotlib.pyplot as plt

energy      = ["5", "9", "24"]
state       = {"Neutral":44, "C3":43, "C4":43, "C5":43}# key = state label, item = number of electrons in state
method      = ["HF", "DFT"]
data        = {}
method_diff = True # plot the difference in scattering between methods, only works for 2 methods (method[1] - method[0])
normalise   = True

def datread(filename,z):
    data = np.loadtxt(filename)
    return data[:,z]

for k in method:
    data[k] = {}
    for i in energy:
       data[k][i] = {}
       for j in state.keys():                 # file name structure "state"_"method"_"energy"Kev.txt
               data[k][i][j] = datread('data_files/'+str(j)+"_"+str(k)+"_"+str(i)+"Kev.txt",2)
         
# Normalise SCF data
if normalise == True:
    for k in method:
        for i in energy:
            for j,val in state.items():
                 data[k][i][j]  = (data[k][i][j] / data[k][i][j].max()) * (val ** 2)

plotting.plot_detector(data, len(energy), len(state)).log_data()

# Method[1] - Method[0]
if method_diff == True:
    method_diff = {"DFT-HF":{}}
    for i in energy:
        method_diff["DFT-HF"][i]     = {} 
        for j in state:
            method_diff["DFT-HF"][i][j]     = data["DFT"][i][j] - data["HF"][i][j]

    plotting.plot_detector(method_diff, len(energy), len(state)).method_diff()
        


