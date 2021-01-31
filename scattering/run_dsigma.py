#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:54:32 2020

@author: afouda

This code calculates coherent x-ray scattering differentail cross-section,
at various photon energies on various states cacluculate at different levels of theory

This script acts as an input file where the energies are selected. 
The states and methods selected here will correspond to molden files (that can be read by ORBKIT)
in a directory named molden_files named "state"_"method".molden 

X-RHO-SCATTER contains a PsiNumPy script taht can calculate ground and core-hole wavefunctions with either
HF or DFT. This code is adapted from PSIXAS, molden files could be generated by that code or any other orbkit 
compatible program.
"""

import numpy as np
import dsigma

e_dict      = {"5":5600, "9":9000,"24":24000}
state_list  = ["Neutral", "C3", "C4", "C5"] 
method_list = ["HF", "DFT", "DFTopt"]

#scattering angles 
Theta  = np.linspace(0, (80 * np.pi/180), 100)
Phi = np.linspace(0, (2 * np.pi), 100)

#values for numerical intergration of the density
nprocs    = 8  #parreliastion across intergration grid, ORBKIT and curbature run in serial
precision = 1e-3#intergation precision 
extent    =  9.00 #large enough grid so that the square of the density converges to the numebr of electrons

#see dsigma.py 
dsigma.crosssection(Theta, Phi, e_dict, state_list, method_list, nprocs, precision, extent).Coherent_Elastic()