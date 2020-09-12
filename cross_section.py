#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:34:07 2020

@author: afouda
"""
import os
import time
from orbkit import read,core,grid,options
import numpy as np
from cubature import cubature
import itertools
import multiprocessing
from functools import partial


class crosssection:
    def __init__(self,*args):
        self.Theta       = args[0]
        self.Phi         = args[1]
        self.e_dict      = args[2]
        self.state_list  = args[3]
        self.method_list = args[4]
        self.nprocs      = args[5]
        self.precision   = args[6]
        self.extent      = args[7]
        
#Public        
    def Coherent_Elastic(self):

        data   = {}

        options.quiet  = True
        options.no_log = True
        
        paramlist = list(itertools.product(self.Theta,self.Phi))
        array = np.asarray(paramlist)
       
        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        for k in self.method_list:
            data[k] = {}
            for i,val in self.e_dict.items():
                data[k][i] = {}
                for j in self.state_list:
           
                    w   = val / 27.2114
                    qca = read.main_read('molden_files/'+str(j)+'_'+str(k)+'.molden',itype='molden',all_mo=False,spin='alpha')
                    qcb = read.main_read('molden_files/'+str(j)+'_'+str(k)+'.molden',itype='molden',all_mo=False,spin='beta')
                
                    grid.is_initialized = True
           
                    pool = multiprocessing.Pool(self.nprocs)
                    start_time = time.time()
                    funcpool = partial(Mol_Form_Factor, w, qca, qcb, self.precision, self.extent)
                    data[k][i][j] = pool.map(funcpool,paramlist)
           
                    elapsed_time = time.time() - start_time                  
                    print(elapsed_time)
           
                    np.savetxt(str(data_dir)+"/"+str(j)+"_"+str(k)+"_"+str(i)+"Kev.txt", np.column_stack((array[:,0],array[:,1],data[k][i][j])))
#Private

def Mol_Form_Factor(w, qca, qcb, precision, extent, params):
    numproc = 1
    slice_length = 1e3
    ndim = 3
    fdim = 2
    xmin = np.array([-1 * extent,-1 * extent,-1 *extent],dtype=float)
    xmax = np.array([extent,extent,extent],dtype=float)
    abserr = precision
    relerr = precision 
    a    = 1/137
    kin = a * w
    Theta = params[0]
    Phi   = params[1]
    e_class = True
    Q = Qvector(kin, Theta, Phi)
    F,error_F = cubature(func, ndim, fdim, xmin, xmax,
                                args=[Q, numproc, slice_length, qca, qcb],
                                adaptive='h', abserr=abserr, relerr=relerr,
                                norm=0, maxEval=0, vectorized=True)
    Fcomp = F[0] + 1j * F[1]
    Fsq = np.real(np.absolute(Fcomp) ** 2)
    dsigma = Fsq * Thomson(a, Theta, Phi, e_class)

    return dsigma

def func(x_array,*args):
    
    x_array = x_array.reshape((-1,3))
    grid.x = np.array(x_array[:,0],copy=True)
    grid.y = np.array(x_array[:,1],copy=True)
    grid.z = np.array(x_array[:,2],copy=True)
    
    densa = core.rho_compute(args[3], 
                            slice_length=args[2],
                            drv=None,
                            laplacian=False,
                            numproc=args[1])

    
    densb = core.rho_compute(args[4], 
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

def Thomson(a, Theta, Phi, e_class):
     
    if e_class == True:
        return ((np.cos(Theta) ** 2)*(np.cos(Phi) ** 2)) + (np.sin(Phi) ** 2)
    if e_class == False:
        return (a ** 4) * ( ((np.cos(Theta) ** 2)*(np.cos(Phi) ** 2)) + (np.sin(Phi) ** 2) )
 
def Qvector(kin, Theta, Phi):
    vec = np.array([np.cos(Phi) * np.cos(Theta / 2),
                    np.sin(Phi) * np.cos(Theta / 2), 
                    np.sin(Theta / 2)])
    Q = (2 * np.absolute(kin) * np.sin(Theta / 2)) * vec
    return Q
    
def dot(x, y):
    return sum(x_i*y_i for x_i, y_i in zip(x, y))

   


    