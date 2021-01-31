#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:44:29 2020
The code that produces the plots in https://doi.org/10.1039/D0FD00106F
This whole project is dire need of some generalization

@author: afouda
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib import ticker

class plot_detector:
    def __init__(self,*args):
        self.data = args[0]
        self.row  = args[1]
        self.col  = args[2]
        xydata    = np.loadtxt('xy.txt')
        self.x    = xydata[:,0] 
        self.y    = xydata[:,1]
        self.cmap  = plt.get_cmap('rainbow')  

    def log_data(self): 
        key_min = []
        key_max = []
        
        for i in self.data:
            for j in self.data[i]:
                
                key_min.append(min(self.data[i][j].keys(), key=(lambda k: self.data[i][j][k].all())))
                key_max.append(max(self.data[i][j].keys(), key=(lambda k: self.data[i][j][k].all())))
        
        vmin_array = np.zeros(len(key_min))
        vmax_array = np.zeros(len(key_max))
        
        count = 0
        for i in self.data:
            for j in self.data[i]:
                    vmin_array[count] = self.data[i][j][key_min[count]].min()
                    vmax_array[count] = self.data[i][j][key_max[count]].max()
                    
                    count += 1
                    
        vmin = vmin_array.min()
        vmax = vmax_array.max()
            
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        
        X = (np.tan(self.x)*np.cos(self.y)).reshape(100,100)
        Y = (np.tan(self.x)*np.sin(self.y)).reshape(100,100)

        levels = np.logspace(0,4,100)
        
        count = 0
        for k in self.data: 
            f,ax = plt.subplots(self.row, self.col, sharey=True, sharex=True, figsize=(12,8))
            for i,vali in enumerate(self.data[k]):
                for j,valj in enumerate(self.data[k][vali]):
            
                    Z = (self.data[k][vali][valj] * (np.cos(self.x) ** 3)).reshape(100,100)
                    for l in range(100):
                        for m in range(100):
                            if Z[l,m] < 1:
                                Z[l,m] = 1
                    cf = ax[i,j].contourf(X, Y, Z, levels=levels, cmap=self.cmap,
                              locator=ticker.LogLocator(),
                                norm=colors.LogNorm(vmin=1, vmax=2000))

                    ax[i,j].set_xlim(-1, 1)
                    ax[i,j].set_ylim(-1, 1)
            
            cbar_ax = f.add_axes([0.91, 0.17, 0.015, 0.65])
            formatter = LogFormatterSciNotation(10, labelOnlyBase=False)
            f.colorbar(cf,cax=cbar_ax, ticks=[1,10,100,1000, 10000], format=formatter)

            for i in range(self.row):
                ax[i,0].set_ylabel('y/z', fontsize=15)
    
            for i in range(self.col):
                ax[self.row-1,i].set_xlabel('x/z', fontsize=15)

            plt.subplots_adjust(wspace=0.085, hspace=0.1)

            f.savefig("plots/log_data_"+str(k)+"_detector.png",dpi=300)

            count += 1

    def method_diff(self): 
        key_min = []
        key_max = []
        
        for i in self.data:
            for j in self.data[i]:
                
                key_min.append(min(self.data[i][j].keys(), key=(lambda k: self.data[i][j][k].all())))
                key_max.append(max(self.data[i][j].keys(), key=(lambda k: self.data[i][j][k].all())))
        
        vmin_array = np.zeros(len(key_min))
        vmax_array = np.zeros(len(key_max))
        
        count = 0
        for i in self.data:
            for j in self.data[i]:
                    vmin_array[count] = self.data[i][j][key_min[count]].min()
                    vmax_array[count] = self.data[i][j][key_max[count]].max()
                    
                    count += 1
                    
        vmin = vmin_array.min()
        vmax = vmax_array.max()
            
        plt.rcParams['xtick.labelsize'] = 14
        plt.rcParams['ytick.labelsize'] = 14
        
        X = (np.tan(self.x)*np.cos(self.y)).reshape(100,100)
        Y = (np.tan(self.x)*np.sin(self.y)).reshape(100,100)
    
        #Here the color scale is set
        vmin = [-1.0]
        vmax = [2.5]

        levels = 100
        count = 0
        for k in self.data:
            
           f,ax = plt.subplots(self.row, self.col, sharey=True, sharex=True, figsize=(12,8))
               
           for i,vali in enumerate(self.data[k]):
               for j,valj in enumerate(self.data[k][vali]):
                                                              
                    Z = (self.data[k][vali][valj] * (np.cos(self.x) ** 3)).reshape(100,100)
                    cf = ax[i,j].contourf(X, Y, Z, levels=levels, cmap=self.cmap,
                                     vmin=vmin[count], vmax=vmax[count])
                    
                    ax[i,j].set_xlim(-1, 1)
                    ax[i,j].set_ylim(-1, 1)
                    
           cbar_ax = f.add_axes([0.91, 0.17, 0.015, 0.65])
           f.colorbar(cf,cax=cbar_ax)
            
           for i in range(self.row):
                ax[i,0].set_ylabel('y/z', fontsize=15)
        
           for i in range(self.col):
                ax[self.row-1,i].set_xlabel('x/z', fontsize=15)

           plt.subplots_adjust(wspace=0.085, hspace=0.1)
    
           f.savefig("plots/method_diff_"+str(k)+"_detector.png",dpi=300)
           count += 1
            

