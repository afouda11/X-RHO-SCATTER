#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:44:29 2020

@author: afouda
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib import ticker

def plot(data, row, col, name):
    
    key_min = []
    key_max = []
    for i in data:
        for j in data[i]:
            
            key_min.append(min(data[i][j].keys(), key=(lambda k: data[i][j][k].all())))
            key_max.append(max(data[i][j].keys(), key=(lambda k: data[i][j][k].all())))
    
    vmin_array = np.zeros(len(key_min))
    vmax_array = np.zeros(len(key_max))
    
    count = 0
    for i in data:
        for j in data[i]:
                vmin_array[count] = data[i][j][key_min[count]].min()
                vmax_array[count] = data[i][j][key_max[count]].max()
                            
                count += 1
                
    vmin = vmin_array.min()
    vmax = vmax_array.max()
    
    X = (x * deg).reshape(100,100)
    Y = (y * deg).reshape(100,100)
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    levels = 1000
    count = 0
    for k in data: 
    
        f,ax = plt.subplots(row, col, sharey=True, sharex=True, figsize=(16,8))

        for i,vali in enumerate(data[k]):
            for j,valj in enumerate(data[k][vali]):
                Z = data[k][vali][valj].reshape(100,100)
                cf = ax[i,j].contourf(X, Y, Z, levels=levels, cmap=cmap,
                            vmin=vmin, vmax=vmax)

        cbar_ax = f.add_axes([0.91, 0.17, 0.015, 0.65])
        f.colorbar(cf,cax=cbar_ax)
    
        for i in range(row):
            ax[i,0].set_ylabel('$\Phi$', fontsize=15)
        
        for i in range(col):
            ax[row-1,i].set_xlabel('$\Theta$', fontsize=15)

        plt.subplots_adjust(wspace=0.06, hspace=0.1)
    
        f.savefig("plots/"+str(name)+"_"+str(k)+".png",dpi=300)
        count +=1 
    
def plot_detector(data, row, col, name):

    key_min = []
    key_max = []
    
    for i in data:
        for j in data[i]:
            
            key_min.append(min(data[i][j].keys(), key=(lambda k: data[i][j][k].all())))
            key_max.append(max(data[i][j].keys(), key=(lambda k: data[i][j][k].all())))
    
    vmin_array = np.zeros(len(key_min))
    vmax_array = np.zeros(len(key_max))
    
    count = 0
    for i in data:
        for j in data[i]:
                vmin_array[count] = data[i][j][key_min[count]].min()
                vmax_array[count] = data[i][j][key_max[count]].max()
                
                count += 1
                
    vmin = vmin_array.min()
    vmax = vmax_array.max()
        
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    X = (np.tan(x)*np.cos(y)).reshape(100,100)
    Y = (np.tan(x)*np.sin(y)).reshape(100,100)

    if name == "data_log":
            levels = np.logspace(0,4,100)
            
            count = 0
            for k in data: 
                f,ax = plt.subplots(row, col, sharey=True, sharex=True, figsize=(12,8))
                # z = 
                for i,vali in enumerate(data[k]):
                    for j,valj in enumerate(data[k][vali]):
                
                        Z = (data[k][vali][valj] * (np.cos(x) ** 3)).reshape(100,100)
                        for l in range(100):
                            for m in range(100):
                                if Z[l,m] < 1:
                                    Z[l,m] = 1
                        cf = ax[i,j].contourf(X, Y, Z, levels=levels, cmap=cmap,
                                      locator=ticker.LogLocator(),
                                        norm=colors.LogNorm(vmin=1, vmax=2000))

                        ax[i,j].set_xlim(-1, 1)
                        ax[i,j].set_ylim(-1, 1)
                
                cbar_ax = f.add_axes([0.91, 0.17, 0.015, 0.65])
                formatter = LogFormatterSciNotation(10, labelOnlyBase=False)
                f.colorbar(cf,cax=cbar_ax, ticks=[1,10,100,1000, 10000], format=formatter)
    
                for i in range(row):
                    ax[i,0].set_ylabel('y/z', fontsize=15)
        
                for i in range(col):
                    ax[row-1,i].set_xlabel('x/z', fontsize=15)

                plt.subplots_adjust(wspace=0.085, hspace=0.1)
    
                f.savefig("plots_detector/"+str(name)+"_"+str(k)+".png",dpi=300)
                count += 1

    if name == "method_diff":
        
        vmin = [-30, -30, -1.0, -15.0]
        vmax = [50,   50,    2.5,   8]

        levels = 100
        count = 0
        for k in data:
            
           f,ax = plt.subplots(row, col, sharey=True, sharex=True, figsize=(12,8))
                               
                          
           for i,vali in enumerate(data[k]):
               for j,valj in enumerate(data[k][vali]):
                                                              

                    Z = (data[k][vali][valj] * (np.cos(x) ** 3)).reshape(100,100)
                    
                    if row == 1 and col == 1:
                        cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap,
                                         vmin=Z.min(), vmax=Z.max())
                    else:
                        cf = ax[i,j].contourf(X, Y, Z, levels=levels, cmap=cmap,
                                          vmin=vmin[count], vmax=vmax[count])
                                      
                    if row == 1 and col == 1:
                        ax.set_xlim(-1, 1)
                        ax.set_ylim(-1, 1)
                    else:
                        ax[i,j].set_xlim(-1, 1)
                        ax[i,j].set_ylim(-1, 1)
                    
           cbar_ax = f.add_axes([0.91, 0.17, 0.015, 0.65])
           f.colorbar(cf,cax=cbar_ax)
            
           for i in range(row):
               if row == 1 and col == 1:
                   ax.set_ylabel('y/z', fontsize=15)
            
               
                   ax[i,0].set_ylabel('y/z', fontsize=15)
        
           for i in range(col):
               if row == 1 and col == 1:
                   ax.set_xlabel('x/z', fontsize=15)
               else:
                   ax[row-1,i].set_xlabel('x/z', fontsize=15)
               
           plt.subplots_adjust(wspace=0.085, hspace=0.1)
    
           f.savefig("plots_detector/"+str(name)+"_"+str(k)+".png",dpi=300)
           count += 1
            
def datread(filename,z):
    data = np.loadtxt(filename)

    return data[:,z]
    
######## PLOTTING ##############

E_list      = ["5", "9", "24"]
state_list  = ["Neutral", "C5", "C4", "C3"]
state_list1 = ["C5", "C4", "C3"]
method_list = ["atom", "HF", "DFT", "DFTopt"]
method_list1 = ["HF", "DFT", "DFTopt"]

## quality assement###
# method_list2 = ["atom", "HF", "HFfine", "HFfineQVZ", "DFT", "DFTfine"]
# method_list3 = ["HF", "HFfine", "HFfineQVZ", "DFT", "DFTfine"]
# state_list2  = ["C5"]
# E_list1      = ["5"]

data        = {}
state_diff  = {}
method_diff = {"HF-atom":{}, "DFT-atom":{}, "DFT-HF":{}, "DFT-DFTopt":{}}
# method_diff1 = {"HF-atom":{}, "HFfine-atom":{}, "HFfineQVZ-atom":{}, "DFT-atom":{}, "DFTfine-atom":{},  "DFT-HF":{}, "DFTfine-HFfine":{}}


xydata = np.loadtxt('xy.txt')
deg = 57.2958
x = xydata[:,0] 
y = xydata[:,1]

cmap = plt.get_cmap('rainbow')  

for k in method_list:
    data[k] = {}
    for i in E_list:
       data[k][i] = {}
       for j in state_list:

           data[k][i][j] = datread(str(k)+'_data/'+str(j)+"_"+str(k)+"_"+str(i)+"Kev.txt",2)

# #### Normalise HF and DFT data

for k in method_list1:
    for i in E_list:
        for j in state_list1:
            if j == "Neutra;l":
                data[k][i][j] = (data[k][i][j] / data[k][i][j].max()) * 1936
            else:
                data[k][i][j]  = (data[k][i][j] / data[k][i][j].max()) * 1849

#Neutral - SCH

for k in method_list:
    state_diff[k] = {}
    for i in E_list:
        state_diff[k][i] = {}
        for j in state_list1:
            state_diff[k][i][j] = data[k][i]["Neutral"] - data[k][i][j]
    
# Method - Method

for i in E_list:
      method_diff["HF-atom"][i]    = {}
      method_diff["DFT-atom"][i]   = {} 
      method_diff["DFT-HF"][i]     = {} 
      method_diff["DFT-DFTopt"][i] = {} 
      for j in state_list:
        method_diff["HF-atom"][i][j]    = data["HF"][i][j]  - data["atom"][i][j]
        method_diff["DFT-atom"][i][j]   = data["DFT"][i][j] - data["atom"][i][j]
        method_diff["DFT-HF"][i][j]     = data["DFT"][i][j] - data["HF"][i][j]
        method_diff["DFT-DFTopt"][i][j] = data["DFT"][i][j] - data["DFTopt"][i][j]
      
plot_detector(data, 3, 4, "data_log")
plot_detector(method_diff, 3, 4, "method_diff")

# for i in E_list1:
#     method_diff1["HF-atom"][i]        = {}
#     method_diff1["HFfine-atom"][i]    = {} 
#     method_diff1["HFfineQVZ-atom"][i] = {} 
#     method_diff1["DFT-atom"][i]       = {}
#     method_diff1["DFTfine-atom"][i]       = {}
#     method_diff1["DFT-HF"][i]     = {} 
#     method_diff1["DFTfine-HFfine"][i]     = {} 
    
#     for j in state_list2:
    
#         method_diff1["HF-atom"][i][j]     = data["HF"][i][j]  - data["atom"][i][j]
#         method_diff1["HFfine-atom"][i][j] = data["HFfine"][i][j]  - data["atom"][i][j]
#         method_diff1["HFfineQVZ-atom"][i][j] = data["HFfineQVZ"][i][j]  - data["atom"][i][j]
#         method_diff1["DFT-atom"][i][j] = data["DFT"][i][j]  - data["atom"][i][j]
#         method_diff1["DFTfine-atom"][i][j] = data["DFTfine"][i][j]  - data["atom"][i][j]
#         method_diff1["DFT-HF"][i][j] = data["DFT"][i][j]  - data["HF"][i][j]
#         method_diff1["DFTfine-HFfine"][i][j] = data["DFTfine"][i][j]  - data["HFfine"][i][j]
        



