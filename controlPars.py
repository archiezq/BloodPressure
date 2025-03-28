#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:53:32 2020

@author: lex

class control():
    def _init_control(self,hr,tbv):
        self.hr=70
        self.tbv=5000
        

hr = control("hr","tbv")

print(hr)
"""
import numpy as np

class _init_control(object):
    def __init__(self):
        
        self.hr = 70;
        BW=80; # Kg body weight
        self.BW = BW; 
        self.tbv = 5250;
        self.pth = -2;
        self.tmin = 0.0
        self.tmax = 250 # Here on can set for how long you the simulation want to run in seconds
        self.T = 0.01; # Sample frequency = 1000hz
        self.N = round((self.tmax-self.tmin)/self.T)+1;
        self.RR = 12; # respiratory rate
        self.DF = 0.3; # Diffusion rate
        self.Pintra_t0 = -2; # Start negative intra-thoracic pressure
        
        self.Kn=0.5792;
        # Microcirculatory start parameters
        self.Kf=.4; # Filtration coefficient calculated as the reciprocal of membrane resistance, i.e. 1/Rint- > 1/12;
        self.Pcap=9.2; # 
        self.V_micro=11000; # volume of the 
        self.C_micro=BW*2.9; # Or 2.9 ml/mmHg*Kg
        self.sigma=0.9; # The reflection coefficient σ, considering the leakage of plasma protein, is set to be a constant 0.9 for typical systemic capillaries [J.R. Levick, Capillary filtration-absorption balance reconsidered in light of dynamic extravascular factors, Exp. Physiol. 76 (1991) 825–857.].
        self.n_int=5.2;
        self.n_cap=4.2;
        self.R_transcap=64; # The resistances to transcapillary fluid exchange and lymphatic drainage so determined were 64 and 20 peripheral resistance units, respectively.
        self.R_lymph=20;
        T_c=37;
        self.T_c=T_c;
        self.Temp=T_c+ 273.15;
        self.R_gas=62.3; # gasconstant
