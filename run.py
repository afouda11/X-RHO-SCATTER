#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:54:32 2020

@author: afouda
"""

import numpy as np
import cross_section as sigma

e_dict      = {"5":5600, "9":9000,"24":24000}
state_list  = ["Neutral", "C3", "C4", "C5"]
method_list = ["HF", "DFT", "DFTopt"]

Theta  = np.linspace(0, (80 * np.pi/180), 100)
Phi = np.linspace(0, (2 * np.pi), 100)

nprocs = 16
precision = 1e-4
extent =  9.00

sigma.crosssection(Theta, Phi, e_dict, state_list, method_list, nprocs, precision, extent).Coherent_Elastic()
