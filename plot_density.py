#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:05:42 2020

@author: afouda
"""

from orbkit import read,core,grid,options,extras,display,output

######### Ground HF ################

qca = read.main_read('Neutral.molden',itype='molden',all_mo=False, spin='alpha')
qcb = read.main_read('Neutral.molden',itype='molden',all_mo=False, spin='beta')

grid.delta_ = [0.1, 0.1, 0.1]
grid.grid_init()
print(grid.get_grid())
densa = core.rho_compute(qca, 
                            slice_length=1e4,
                            drv=None,
                            laplacian=False,
                            numproc=12)
    
densb = core.rho_compute(qcb, 
                            slice_length=1e4,
                            drv=None,
                            laplacian=False,
                            numproc=12)

denshfground = densa + densb

output.cube_creator(denshfground, "HF-Ground.cb", qca.geo_info, qca.geo_spec, comments='', labels=None)

######### Core HF ################

qca = read.main_read('C3.molden',itype='molden',all_mo=False, spin='alpha')
qcb = read.main_read('C3.molden',itype='molden',all_mo=False, spin='beta')

grid.delta_ = [0.1, 0.1, 0.1]
grid.grid_init()
print(grid.get_grid())
densa = core.rho_compute(qca, 
                            slice_length=1e4,
                            drv=None,
                            laplacian=False,
                            numproc=12)
    
densb = core.rho_compute(qcb, 
                            slice_length=1e4,
                            drv=None,
                            laplacian=False,
                            numproc=12)

denshfcore = densa + densb

output.cube_creator(denshfcore, "HF-core.cb", qca.geo_info, qca.geo_spec, comments='', labels=None)

######### Ground DFT ################


qca = read.main_read('Neutral_DFT.molden',itype='molden',all_mo=False, spin='alpha')
qcb = read.main_read('Neutral_DFT.molden',itype='molden',all_mo=False, spin='beta')

grid.delta_ = [0.1, 0.1, 0.1]
grid.grid_init()
print(grid.get_grid())
densa = core.rho_compute(qca, 
                            slice_length=1e4,
                            drv=None,
                            laplacian=False,
                            numproc=12)
    
densb = core.rho_compute(qcb, 
                            slice_length=1e4,
                            drv=None,
                            laplacian=False,
                            numproc=12)

densdftground = densa + densb

output.cube_creator(densdftground, "DFT-ground.cb", qca.geo_info, qca.geo_spec, comments='', labels=None)

######### Core DFT DFT ################

qca = read.main_read('DFT-core_hole.molden',itype='molden',all_mo=False, spin='alpha')
qcb = read.main_read('DFT-core_hole.molden',itype='molden',all_mo=False, spin='beta')

grid.delta_ = [0.1, 0.1, 0.1]
grid.grid_init()
print(grid.get_grid())
densa = core.rho_compute(qca, 
                            slice_length=1e4,
                            drv=None,
                            laplacian=False,
                            numproc=12)
    
densb = core.rho_compute(qcb, 
                            slice_length=1e4,
                            drv=None,
                            laplacian=False,
                            numproc=12)

densdftcore = densa + densb

output.cube_creator(densdftcore, "DFT-core.cb", qca.geo_info, qca.geo_spec, comments='', labels=None)

hfdiff = denshfground - denshfcore

dftdiff = densdftground - densdftcore

output.cube_creator(hfdiff, "HF-Diff.cb", qca.geo_info, qca.geo_spec, comments='', labels=None)
output.cube_creator(dftdiff, "DFT-Diff.cb", qca.geo_info, qca.geo_spec, comments='', labels=None)









