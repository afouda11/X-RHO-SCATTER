#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:34:07 2020

@author: afouda
"""

import time
from orbkit import read,core,grid,options,extras,display,output
import numpy as np
from cubature import cubature
import itertools
import multiprocessing
from functools import partial

def dot(x, y):
    return sum(x_i*y_i for x_i, y_i in zip(x, y))

def Qvector(kin, Theta, Phi):
    vec = np.array([np.cos(Phi) * np.cos(Theta / 2),
                    np.sin(Phi) * np.cos(Theta / 2), 
                    np.sin(Theta / 2)])
    Q = (2 * np.absolute(kin) * np.sin(Theta / 2)) * vec
    return Q
    
def func(x_array,*args):
    
    x_array = x_array.reshape((-1,3))
    grid.x = np.array(x_array[:,0],copy=True)
    grid.y = np.array(x_array[:,1],copy=True)
    grid.z = np.array(x_array[:,2],copy=True)

    densa = core.rho_compute(qca, 
                            slice_length=args[2],
                            drv=None,
                            laplacian=False,
                            numproc=args[1])
    
    densb = core.rho_compute(qcb, 
                            slice_length=args[2],
                            drv=None,
                            laplacian=False,
                            numproc=args[1])
    
    dens = (densa + densb)
    
    rvec = np.array([   x_array[:,0],    x_array[:,1],     x_array[:,2]])
    qvec = np.array([1 * args[0][0], 1 * args[0][2],   -1 * args[0][1]])
    
    dotp = dot(qvec,rvec)
    
    npt = x_array.shape[0]
    out = np.zeros((npt, 2))
    out[:, 0] = np.real(dens * np.exp(1j * dotp))
    out[:, 1] = np.imag(dens * np.exp(1j * dotp))

    return out

def thomson(a, Theta, Phi):
     
    # return (a ** 4) * ( ((np.cos(Theta) ** 2)*(np.cos(Phi) ** 2)) + (np.sin(Phi) ** 2) )
    return ((np.cos(Theta) ** 2)*(np.cos(Phi) ** 2)) + (np.sin(Phi) ** 2)


def crosssection(w, params):
    numproc = 1
    slice_length = 1e2
    ndim = 3
    fdim = 2
    xmin = np.array([-9.00,-9.00,-9.00],dtype=float)
    xmax = np.array([9.00,9.00,9.00],dtype=float)
    abserr = 1e-2
    relerr = 1e-2
    a    = 1/137
    kin = a * w
    Theta = params[0]
    Phi   = params[1]

    Q = Qvector(kin, Theta, Phi)
    F,error_F = cubature(func, ndim, fdim, xmin, xmax,
                                args=[Q, numproc, slice_length],
                                adaptive='h', abserr=abserr, relerr=relerr,
                                norm=0, maxEval=0, vectorized=True)
    Fcomp = F[0] + 1j * F[1]
    Fsq = np.real(np.absolute(Fcomp) ** 2)
    dsigma = Fsq * thomson(a, Theta, Phi)

    return dsigma

# def run(name):
    
   
#     start_time = time.time()
#     DifScat = 
#     elapsed_time = time.time() - start_time                  
#     print(elapsed_time)

#     data[name] = DifScat
    
#     return data[name]
    
Theta  = np.linspace(0, (80 * np.pi/180), 5)
Phi = np.linspace(0, (2 * np.pi), 5)
paramlist = list(itertools.product(Theta,Phi))
pool = multiprocessing.Pool(16)

e_dict      = {"5":5600, "9":9000, "24":24000}
state_list  = ["Neutral", "C5", "C4", "C3"]
method_list = ["HF", "DFT", "DFTopt"]

data   = {}

options.quiet  = True
options.no_log = True

######## RUN ##############
paramlist = list(itertools.product(Theta,Phi))
array = np.asarray(paramlist)
for k in method_list:
    data[k] = {}
    for i,val in e_dict.items():
       data[k][i] = {}
       for j in state_list:
           
           w   = val / 27.2114
           qca = read.main_read('molden_files/'+str(j)+'_'+str(k)+'.molden',itype='molden',all_mo=False,spin='alpha')
           qcb = read.main_read('molden_files/'+str(j)+'_'+str(k)+'.molden',itype='molden',all_mo=False,spin='beta')
           
           grid.is_initialized = True
           
           pool = multiprocessing.Pool(16)
           start_time = time.time()
           funcpool = partial(crosssection, w)
           data[k][i][j] = pool.map(funcpool,paramlist)
           
           elapsed_time = time.time() - start_time                  
           print(elapsed_time)
           
           np.savetxt(str(k)+'_data/'+str(j)+"_"+str(k)+"_"+str(i)+"Kev.txt", np.column_stack((array[:,0],array[:,1],data[k][i][j])))


