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
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
import itertools
import multiprocessing

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
    
    dens = 0.5 * (densa + densb)
    
    rvec = np.array([   x_array[:,0],    x_array[:,1],     x_array[:,2]])
    qvec = np.array([1 * args[0][0], 1 * args[0][2],   -1 * args[0][1]])
    
    dotp = dot(qvec,rvec)
    
    npt = x_array.shape[0]
    out = np.zeros((npt, 2))
    out[:, 0] = np.real(dens * np.exp(1j * dotp))
    out[:, 1] = np.imag(dens * np.exp(1j * dotp))

    return out

def thomson(a, Theta, Phi):
     
    return (a ** 4) * ( ((np.cos(Theta) ** 2)*(np.cos(Phi) ** 2)) + (np.sin(Phi) ** 2) )


def crosssection(params):
    numproc = 1
    slice_length = 1e3
    ndim = 3
    fdim = 2
    xmin = np.array([-9.00,-9.00,-9.00],dtype=float)
    xmax = np.array([9.00,9.00,9.00],dtype=float)
    abserr = 1e-3
    relerr = 1e-3
    a    = 1/137
    w   = 5600 / 27.2114
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

def run(name):

    paramlist = list(itertools.product(Theta,Phi))
    pool = multiprocessing.Pool(16)
    start_time = time.time()
    DifScat = pool.map(crosssection,paramlist)
    elapsed_time = time.time() - start_time                  
    print(elapsed_time)

    griddata = np.zeros((len(Theta),len(Phi)))
    count = 0
    for i in range(len(Phi)):
        for j in range(len(Theta)):
            griddata[j][i] = DifScat[count]
            count += 1
    
    gridbarn[name] = griddata / (57106 * (10 ** -8))
    levels[name] = MaxNLocator(nbins=1000).tick_values(gridbarn[name].min(), gridbarn[name].max())
    
    return gridbarn[name], levels[name]

def plot(ax, grid, levels):
    deg = 57.2958
    return ax.contourf(Theta*deg, Phi*deg, grid,
                       levels=levels, cmap=cmap)

def plotlog(ax, grid, levels):
    deg = 57.2958
    return ax.contourf(Theta*deg, Phi*deg, grid, 
                       norm=colors.LogNorm(vmin=grid.min(), vmax=grid.max()),
                       levels=levels, cmap=cmap)
    
Theta  = np.linspace(0, (80 * np.pi/180), 5)
Phi = np.linspace(0, (2 * np.pi), 5)

gridbarn = {"Neutral":[], "C0":[], "C1":[], "C2":[], "C3":[], "C4":[], "C5":[]}
levels   = {"Neutral":[], "C0":[], "C1":[], "C2":[], "C3":[], "C4":[], "C5":[]}

options.quiet  = True
options.no_log = True
######## Neutral ##############
qca = read.main_read('Neutral.molden',itype='molden',spin='alpha')
qcb = read.main_read('Neutral.molden',itype='molden',spin='beta')
grid.is_initialized = True
gridbarn["Neutral"], levels["Neutral"] = run("Neutral")

######## C0 ##############
qca = read.main_read('C0.molden',itype='molden',spin='alpha')
qcb = read.main_read('C0.molden',itype='molden',spin='beta')
grid.is_initialized = True
gridbarn["C0"], levels["C0"] = run("C0")

######## C1 ##############
qca = read.main_read('C1.molden',itype='molden',spin='alpha')
qcb = read.main_read('C1.molden',itype='molden',spin='beta')
grid.is_initialized = True
gridbarn["C1"], levels["C1"] = run("C0")

######## C2 ##############
qca = read.main_read('C2.molden',itype='molden',spin='alpha')
qcb = read.main_read('C2.molden',itype='molden',spin='beta')
grid.is_initialized = True
gridbarn["C2"], levels["C2"] = run("C0")

######## C3 ##############
qca = read.main_read('C3.molden',itype='molden',spin='alpha')
qcb = read.main_read('C3.molden',itype='molden',spin='beta')
grid.is_initialized = True
gridbarn["C3"], levels["C3"] = run("C3")

######## C4 ##############
qca = read.main_read('C4.molden',itype='molden',spin='alpha')
qcb = read.main_read('C4.molden',itype='molden',spin='beta')
grid.is_initialized = True
gridbarn["C4"], levels["C4"] = run("C4")

######## C5 ##############
qca = read.main_read('C5.molden',itype='molden',spin='alpha')
qcb = read.main_read('C5.molden',itype='molden',spin='beta')
grid.is_initialized = True
gridbarn["C5"], levels["C5"] = run("C5")

cmap = plt.get_cmap('rainbow')
# fig, ((ax1,ax2), (ax3,ax4),(ax5,ax6), (ax7,ax8), (ax9,ax10), (ax11,ax12), (ax13,ax14)) = plt.subplots(2,7)
fig, ((ax1,ax3,ax5,ax7,ax9,ax11,ax13), (ax2,ax4,ax6,ax8,ax10,ax12,ax14)) = plt.subplots(2,7, figsize=(17,8))

cf1 = plot(ax1,    gridbarn["Neutral"], levels["Neutral"])
cf2 = plotlog(ax2, gridbarn["Neutral"], levels["Neutral"])

cf3 = plot(ax3,    gridbarn["C0"], levels["C0"])
cf4 = plotlog(ax4, gridbarn["C0"], levels["C0"])

cf5 = plot(ax5,    gridbarn["C1"], levels["C1"])
cf6 = plotlog(ax6, gridbarn["C1"], levels["C1"])

cf7 = plot(ax7,    gridbarn["C2"], levels["C2"])
cf8 = plotlog(ax8, gridbarn["C2"], levels["C2"])

cf9 = plot(ax9,      gridbarn["C3"], levels["C3"])
cf10 = plotlog(ax10, gridbarn["C3"], levels["C3"])

cf11 = plot(ax11,    gridbarn["C4"], levels["C4"])
cf12 = plotlog(ax12, gridbarn["C4"], levels["C4"])

cf13 = plot(ax13,    gridbarn["C5"], levels["C5"])
cf14 = plotlog(ax14, gridbarn["C5"], levels["C5"])

# fig.colorbar(cf1, ax=ax1)
ax1.set_xlabel('$\Theta$')
ax1.set_ylabel('$\Phi$')
ax2.set_xlabel('$\Theta$')
ax2.set_ylabel('$\Phi$')
ax3.set_xlabel('$\Theta$')
ax4.set_xlabel('$\Theta$')
ax5.set_xlabel('$\Theta$')
ax6.set_xlabel('$\Theta$')
ax7.set_xlabel('$\Theta$')
ax8.set_xlabel('$\Theta$')
ax9.set_xlabel('$\Theta$')
ax10.set_xlabel('$\Theta$')
ax11.set_xlabel('$\Theta$')
ax12.set_xlabel('$\Theta$')
ax13.set_xlabel('$\Theta$')
ax14.set_xlabel('$\Theta$')
ax3.set_yticks([])
ax4.set_yticks([])
ax5.set_yticks([])
ax6.set_yticks([])
ax7.set_yticks([])
ax8.set_yticks([])
ax9.set_yticks([])
ax10.set_yticks([])
ax11.set_yticks([])
ax12.set_yticks([])
ax13.set_yticks([])
ax14.set_yticks([])

plt.savefig('crosssection_lowres.png',dpi=300)

griddiff   = {"C0":[], "C1":[], "C2":[], "C3":[], "C4":[], "C5":[]}
levelsdiff = {"C0":[], "C1":[], "C2":[], "C3":[], "C4":[], "C5":[]}
for name in griddiff:
    griddiff[name]   = gridbarn["Neutral"] - gridbarn[name]
    levelsdiff[name] = MaxNLocator(nbins=1000).tick_values(griddiff[name].min(), griddiff[name].max())
    
fig, ((ax1,ax2,ax3,ax4,ax5,ax6)) = plt.subplots(1,6, figsize=(17,8))

cf1 = plot(ax1, griddiff["C0"], levelsdiff["C0"])

cf2 = plot(ax2, griddiff["C1"], levelsdiff["C1"])

cf3 = plot(ax3, griddiff["C2"], levelsdiff["C2"])

cf4 = plot(ax4, griddiff["C3"], levelsdiff["C3"])

cf5 = plot(ax5, griddiff["C4"], levelsdiff["C4"])

cf6 = plot(ax6, griddiff["C5"], levelsdiff["C5"])

ax1.set_xlabel('$\Theta$')
ax1.set_ylabel('$\Phi$')
ax2.set_xlabel('$\Theta$')
ax3.set_xlabel('$\Theta$')
ax4.set_xlabel('$\Theta$')
ax5.set_xlabel('$\Theta$')
ax6.set_xlabel('$\Theta$')

ax3.set_yticks([])
ax4.set_yticks([])
ax5.set_yticks([])
ax6.set_yticks([])


plt.savefig('crosssection_difference_lowres.png',dpi=300)

# fig, ((ax1,ax3,ax5,ax7,ax9,ax11), (ax2,ax4,ax6,ax8,ax10,ax12)) = plt.subplots(2,6, figsize=(17,8))

# cf1 = plot(ax1,    griddiff["C0"], levelsdiff["C0"])
# cf2 = plotlog(ax2, griddiff["C0"], levelsdiff["C0"])

# cf3 = plot(ax3,    griddiff["C1"], levelsdiff["C1"])
# cf4 = plotlog(ax4, griddiff["C1"], levelsdiff["C1"])

# cf5 = plot(ax5,    griddiff["C2"], levelsdiff["C2"])
# cf6 = plotlog(ax6, griddiff["C2"], levelsdiff["C2"])

# cf7 = plot(ax7,    griddiff["C3"], levelsdiff["C3"])
# cf8 = plotlog(ax8, griddiff["C3"], levelsdiff["C3"])

# cf9 = plot(ax9,      griddiff["C4"], levelsdiff["C4"])
# cf10 = plotlog(ax10, griddiff["C4"], levelsdiff["C4"])

# cf11 = plot(ax11,    griddiff["C5"], levelsdiff["C5"])
# cf12 = plotlog(ax12, griddiff["C5"], levelsdiff["C5"])
# ax1.set_xlabel('$\Theta$')
# ax1.set_ylabel('$\Phi$')
# ax2.set_xlabel('$\Theta$')
# ax2.set_ylabel('$\Phi$')
# ax3.set_xlabel('$\Theta$')
# ax4.set_xlabel('$\Theta$')
# ax5.set_xlabel('$\Theta$')
# ax6.set_xlabel('$\Theta$')
# ax7.set_xlabel('$\Theta$')
# ax8.set_xlabel('$\Theta$')
# ax9.set_xlabel('$\Theta$')
# ax10.set_xlabel('$\Theta$')
# ax11.set_xlabel('$\Theta$')
# ax12.set_xlabel('$\Theta$')
# ax3.set_yticks([])
# ax4.set_yticks([])
# ax5.set_yticks([])
# ax6.set_yticks([])
# ax7.set_yticks([])
# ax8.set_yticks([])
# ax9.set_yticks([])
# ax10.set_yticks([])
# ax11.set_yticks([])
# ax12.set_yticks([])

# plt.savefig('crosssection_difference_lowres.png',dpi=300)

#write data
# a    = 1/137
# w   = 5600 / 27.2114
# kin = a * w
# array = np.asarray(paramlist)
# q0 = np.zeros((len(Theta) * len(Phi)))
# q1 = np.zeros((len(Theta) * len(Phi)))
# q2 = np.zeros((len(Theta) * len(Phi)))
# count = 0
# for i in range(len(Phi)):
#     for j in range(len(Theta)):
#         Q[count] = Qvector(kin, Theta[j], Phi[i])
#         count +=1
# data = np.column_stack((array[:,0],array[:,1],q0,q1,q2,DifScat))
# np.savetxt('neutral_lowres_data.txt',data)





