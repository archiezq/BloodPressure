#%%
# CVSIM - IMPORT AND DEFINITIONS

import os
clear = lambda: os.system('clear')
clear()
#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
import sys
import cProfile
import pstats
import numpy as np #Numpy is a 'fundamental package for scientific computing with Python'
np.set_printoptions(threshold=sys.maxsize) # This makes sure that all elements are printed, without truncation.
import scipy as sp # SciPy is a free and open-source Python library used for scientific computing and technical computing
import matplotlib.pyplot as plt # Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
import scipy.signal as signal
from scipy.io import loadmat
import timerz
import math
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
import csv
import platform

# Load the right parameters settings
#from adultPars import _init_pars # Get the parameters for resistance, elastance and uvolume
#from controlPars import _init_control # Get the control parameters loaded.
#from reflexPars import _init_reflex # Get the control parameters loaded.

#import PySimpleGUI as sg


#%%
# Solve the model

def solve(scaling):
    
    Timer = timerz.Timer()
    Timer.start()

    n_variables = 107
    inputs = np.ones(n_variables)

    class _init_reflex(object):
        def __init__(self):

            """
                Create impulse responses
            """
            
            S_GRAN = 0.0625
            SUMZ = 0
            I_length=960
            S_INT = .25
            p=np.zeros(I_length)
            s=np.zeros(I_length)
            v=np.zeros(I_length)
            a=np.zeros(I_length)
            cpv=np.zeros(I_length)
            cpa=np.zeros(I_length)
            
            # Set up parasympathetic impulse response function
            start = int(np.round((scaling["baro_delay_para"] + scaling["baro_delay2"] + .59 - 3.0*S_GRAN) / S_GRAN)); #vec[25] // Delay of the parasymp. impulse response
            peak  = int(np.round((scaling["baro_delay_para"] + scaling["baro_delay2"] + .70 - 3.0*S_GRAN) / S_GRAN)); #vec[26] // Peak of the parasymp. impulse response
            end   = int(np.round((scaling["baro_delay_para"] + scaling["baro_delay2"] + 1 - 3.0*S_GRAN) / S_GRAN)); #vec[27] // End of the parasymp. impulse response
            SUMZ = 0
            for i in range(start,peak):
                p[i]=(i-start)/(peak-start)
                SUMZ=SUMZ+p[i]
            for i in range(peak,end):
                p[i]=(end-i)/(end-peak)
                SUMZ=SUMZ+p[i]
            for i in range(start,end):
                p[i]=p[i]/(SUMZ)
            
            # Do the same for the heldtbeta-sympathetic impulse response function.
            start = int(np.round((scaling["baro_delay_sympa"] + scaling["baro_delay2"] + 2.5 - 3.0*S_GRAN) / S_GRAN)); #vec[28] // Delay of the parasymp. impulse response
            peak  = int(np.round((scaling["baro_delay_sympa"] + scaling["baro_delay2"] + 3.5 - 3.0*S_GRAN) / S_GRAN)); #vec[29] // Peak of the parasymp. impulse response
            end   = int(np.round((scaling["baro_delay_sympa"] + scaling["baro_delay2"] + 15 - 3.0*S_GRAN) / S_GRAN)); #vec[30] // End of the parasymp. impulse response
            SUMZ = 0
            for i in range(start,peak):
                s[i]=(i-start)/(peak-start)
                SUMZ=SUMZ+s[i]
            for i in range(peak,end):
                s[i]=(end-i)/(end-peak)
                SUMZ=SUMZ+s[i]
            for i in range(start,end):
                s[i]=s[i]/(SUMZ)
                        
            # Do the same for the venous alpha-sympathetic impulse response function.
            start = int(np.round((scaling["baro_delay_BR_R"] + scaling["baro_delay2"] + 5 - 3.0*S_GRAN) / S_GRAN)); #vec[112] // Delay of the alpha-sympathetic impulse response
            peak  = int(np.round((scaling["baro_delay_BR_R"] + scaling["baro_delay2"] + 10 - 3.0*S_GRAN) / S_GRAN)); #vec[113] // Peak of the alpha-sympathetic impulse response
            end   = int(np.round((scaling["baro_delay_BR_R"] + scaling["baro_delay2"] + 42 - 3.0*S_GRAN) / S_GRAN)); #vec[114] // End of the alpha-sympathetic impulse response
            SUMZ=0
            for i in range(start,peak):
                v[i]=(i-start)/(peak-start)
                SUMZ=SUMZ+v[i]
            for i in range(peak,end):
                v[i]=(end-i)/(end-peak)
                SUMZ=SUMZ+v[i]
            for i in range(start,end):
                v[i]=v[i]/(SUMZ)
                                
            # Do the same for the arterial alpha-sympathetic impulse response function.
            start = int(np.round((scaling["baro_delay_BR_UV"] + scaling["baro_delay2"] + 2.5 - 3.0*S_GRAN) / S_GRAN)); #vec[115] // Delay  impulse response
            peak  = int(np.round((scaling["baro_delay_BR_UV"] + scaling["baro_delay2"] + 3.5 - 3.0*S_GRAN) / S_GRAN)); #vec[116] // Peak  impulse response
            end   = int(np.round((scaling["baro_delay_BR_UV"] + scaling["baro_delay2"] + 30 - 3.0*S_GRAN) / S_GRAN)); #vec[117] // End impulse response
            SUMZ=0
            for i in range(start,peak):
                a[i]=(i-start)/(peak-start)
                SUMZ=SUMZ+a[i]
            for i in range(peak,end):
                a[i]=(end-i)/(end-peak)
                SUMZ=SUMZ+a[i]
            for i in range(start,end):
                a[i]=a[i]/(SUMZ)
                                
            # Do the same for the cardiopulmonary to venous tone alpha-sympathetic impulse response function.
            start = int(np.round((scaling["baro_delay_CP_UV"] + scaling["baro_delay2"] + 5 - 3.0*S_GRAN) / S_GRAN)); #vec[147] // Delay to veins impulse response
            peak  = int(np.round((scaling["baro_delay_CP_UV"] + scaling["baro_delay2"] + 9 - 3.0*S_GRAN) / S_GRAN)); #vec[148] // Peak to veins impulse response
            end   = int(np.round((scaling["baro_delay_CP_UV"] + scaling["baro_delay2"] + 40 - 3.0*S_GRAN) / S_GRAN)); #vec[149] // End to veins impulse response
            SUMZ=0
            for i in range(start,peak):
                cpv[i]=(i-start)/(peak-start)
                SUMZ=SUMZ+cpv[i]
            for i in range(peak,end):
                cpv[i]=(end-i)/(end-peak)
                SUMZ=SUMZ+cpv[i]
            for i in range(start,end):
                cpv[i]=cpv[i]/(SUMZ)
                                
            # Do the same for the cardiopulmonary to arterial resistance alpha-sympathetic impulse response function.
            start = int(np.round((scaling["baro_delay_CP_R"] + scaling["baro_delay2"] + 2.5 - 3.0*S_GRAN) / S_GRAN)); #vec[150] // Delay to arteries impulse response
            peak  = int(np.round((scaling["baro_delay_CP_R"] + scaling["baro_delay2"] + 5.5 - 3.0*S_GRAN) / S_GRAN)); #vec[151] // Peak to arteries impulse response
            end   = int(np.round((scaling["baro_delay_CP_R"] + scaling["baro_delay2"] + 35 - 3.0*S_GRAN) / S_GRAN)); #vec[152] // End to arteries impulse response
            SUMZ=0
            for i in range(start,peak):
                cpa[i]=(i-start)/(peak-start)
                SUMZ=SUMZ+cpa[i]
            for i in range(peak,end):
                cpa[i]=(end-i)/(end-peak)
                SUMZ=SUMZ+cpa[i]
            for i in range(start,end):
                cpa[i]=cpa[i]/(SUMZ)
            """
            Assign the parameters the the self function so then can be opened in the main function when called upon
            """
            # Inpulse responses
            self.p = p
            self.s = s
            self.a = a
            self.v =v
            self.cpv = cpv
            self.cpa = cpa
            # Setpoints
            self.ABP_setp = 95
            self.PP_setp = 35
            self.RAP_setp = 3 # vec[15]
            self.ABRsc = 18 # vec[1]
            self.RAPsc = 5 # vec[16]
            # Gains
            #self.RRsgain = 0.012 # vec[2]
            #self.RRpgain = 0.009 # vec[3]
            self.RRsgain = 0.012 # vec[2]
            self.RRpgain = 0.005 # vec[3]
            self.rr_sym=0.015 #vec[2] // beta
            self.rr_para=0.09 #vec[3] // para
            self.beta=1
            self.alpha=1
            # Timing
            self.S_GRAN = S_GRAN
            self.SUMZ = SUMZ
            self.I_length=I_length
            self.S_INT=S_INT
    
    #======= define the cerebral parameters =======#
    if cerebralModelOn==1:
        class cerebralPars:

            # basal values of quantities related to the intracranial circuit (supine condition) Table 1
            # LvL, i am going to need to clarification on what these parameters are/represent
            C_pan = 0.205 # ml/mmHg, basal capacity of the pial arteries
            DeltaC_pa1 = 2.87 # ml/mmHg
            DeltaC_pa2 = 0.164 # ml/mmHg
            G_aut = 3.0*scaling["G_aut"] # Gain of the autoregulation mechanism related to CBF variations
            #G_aut = 1.0 # Gain of the autoregulation mechanism related to CBF variations
            k_E = 0.077 # ml^-1, Intracranial elastance coefficient
            k_R = 13.1*10**3 # mmHg^3 s ml^-1
            k_ven = 0.155 # ml^-1
            #-------
            #P_a = 100 # mmHg , Lvl, in the original paper fixed at 100 mmHg. We want to make this pulsatile by linking it to the 21-comp model
            P_car = 100 # mmHg, initial carotid pressure
            R_car = 0.008 # mmHg s ml^-1, carotid resistance
            """ R = 8*rho*L/(pi*r^4) = 8*3.5e-3*21.5e-2/(pi*(5.5e-3)^4) = 0.016 mmHg s ml^-1
            2 resistors in parallel: 1/Rt = 1/R1 + 1/R2 = 1/0.008 mmHg s ml^-1
            """
            #-------
            P_ic = 9.5 # mmHg
            P_pa_initial = 58.9 # mmHg Lvl, is this a fixed value?
            P_pa = 58.9 
            P_v = 14.1 # mmHg LvL, is this a fixed value? difference with P_v1? Does not seem to be fixed see line 520
            P_v1 = -2.5 # mmHg #LvL, Fixed?
            P_vs = 6.0 # mmHg #LvL, dynamic value see line 587?
            Q_n = 12.5 # ml/s, redifined in the code?
            #Q_n = 12.5 # ml/s, redifined in the code?
            R_0 = 526.3 # mmHg s ml^-1, Resistance to the cerebrospinal fluid outflow
            R_f = 2.38*10**3 # mmHg s ml^-1
            #R_la = 0.6 # mmHg s ml^-1, # ORIGINAL VALUE
            R_la = 0.6 # mmHg s ml^-1

            R_pv = 0.880 # mmHg s ml^-1
            R_vs1 = 0.366 # mmHg s ml^-1
            #tau_aut = 20 # s # ORIGINAL VALUE
            tau_aut = 20 # s
            x_aut = 2.16*10**(-4)
            hbf = 0.0

            # MCA velocity parameters (Hayashi et al. 1979) ORIGINAL VALUES
            k_v = 1*scaling["k_v"] # Doppler angle correction parameter:
            """ Scaling factor that accounts for the slope of the Doppler probe relative to the vessel under examination
            and for the proportionality between average and maximum velocity in the section. """
            r_mca_n = 0.14 # cm. MCA inner radius in basal condition
            """ Sandhya Arvind Gunnal - 2019 - Diameter of the MCA at its middle of the main stem was measured. It was ranging from 2 to 5 mm with the mean of 3 mm. """
            k_mca = 12 # parameter representing the rigidity of the artery
            P_a_n = 100 # mmHg. Normal arterial pressure -> revise?
            P_ic_n = 9.5 # mmHg. Normal intracranial pressure -> revise?

            #k_v = 2/3 # Doppler angle correction parameter:
            #r_mca_n = 0.14 # cm. MCA inner radius in basal condition
            #k_mca = 12 # parameter representing the rigidity of the artery
            #P_a_n = 100 # mmHg. Normal arterial pressure -> revise?
            #P_ic_n = 9.5 # mmHg. Normal intracranial pressure -> revise?

            "-------------------------------------------------------------------------------------------------------------"

            # Table 2. Basal values of pressures related to the jugular-vertebral circuit (supine condition)

            P_c3 = 6.00 # mmHg
            P_c2 = 5.85 # mmHg
            P_vv = 5.80 # mmHg
            P_jr3 = 5.85# mmHg
            P_jl3 = 5.85 # mmHg
            P_jr2 = 5.70  # mmHg
            P_jl2 = 5.70 # mmHg
            P_azy = 5.50 # mmHg
            P_svc1 = 5.40 # mmHg
            P_svc = 5.20 # mmHg
            #------
            P_cv_initial = 5.0 # mmHg #Lvl, fixed CVP value that needs to be linked to the 21-comp model 
            #------
            P_j3ext = 0.0
            P_j2ext = 0.0
            P_j1ext = -6.5 # mmHg, avg thoracic pressure taken from source (18) as suggested by authors
            P_rv = 5.30 # mmHg


            "---------------------------- initial flows ----------------------------------------"

            # Table 3. Basal values of flows related to the jugular-vertebral circuit (supine condition)

            Q_car = 12.5 # ml/s
            Q = 12.5 # ml/s
            Q_la = 12.5 # ml/s ?
            #Q_pa = 7.5 # ml/s ?
            Q_pa = 12.5 # ml/s ?
            Q_auto = 11.7
            Q_jr3 = 5.85 # ml/s
            Q_jl3  = 5.85 # ml/s
            Q_vvr = 0.40 # ml/s
            Q_vvl = 0.40 # ml/s
            Q_ex = 5.00 # ml/s
            Q_cjr3 = 1.00 # ml/s
            Q_cjl3 = 1.00 # ml/s
            Q_cjr2 = 1.00 # ml/s
            Q_cjl2 = 1.00 # ml/s
            Q_c2 = 3.00 # ml/s
            Q_azy1 = 0.40 # ml/s
            Q_lv = 0.13 # ml/s
            Q_rv = 0.27 # ml/s

            Q_svc = Q_jr3 + Q_jl3 + Q_vvr + Q_vvl # ml/s

            "----------------------------------------- capacities ----------------------------------------------------"

            # capacities related to the jugular-vertrebral circuit (supine condition) Table 4

            C_vs = 0.5 # ml/mmHg
            C_jr3 =  1.0 # ml/mmHg
            C_jl3 =  1.0 # ml/mmHg
            C_jr2 = 2.5 # ml/mmHg
            C_jl2 = 2.5 # ml/mmHg
            C_c3 = 0.7 # ml/mmHg
            C_c2 = 1.4 # ml/mmHg
            C_svc = 20.0 # ml/mmHg
            C_azy = 0.5 # ml/mmHg
            C_vv = 0.5 # ml/mmHg
            C_svc1 = 18.0 #attempts to match the original model's predictions
            #C_car = 1.63 # ml/mmHg, Duprez et al. 2000
            #C_car = 0.2*(1/(1.07158593*1.6)) # ml/mmHg
            C_car = 0.2 # ml/mmHg
            

            "-------------------------------------------------------------------------------------------------------------"

            # values associated to posteriori information mentioned in text (last paragraph of assignment of basal parameters)
            A = 0.8
            k_jr3 = 11.0
            k_jr2 = 13.0
            k_jr1 = 6.9
            k_jl3 = 11.0
            k_jl2 = 13.0
            k_jl1 = 6.9

            "-------------------------------------------------------------------------------------------------------------"

            P_paP_ic = P_pa_initial - P_ic # difference in pressure between pial arterioles and intracranial pressure (used to gain an initial value)

            P_vP_ic = P_v - P_ic # difference in pressure between venous pressure and intracranial pressure 

            # lengths of segments

            L3 = 0.116 # m
            L2  = 0.105 # m
            L1 = 0.03 # m

            "------------------------------------conductance calculations------------------------------------------"

            # vessel conductances (calculated using pressure-flow relationships)

            G_cjr3 = Q_cjr3/(P_c3 - P_jr3)
            G_cjr2 = Q_cjr2/(P_c2 - P_jr2)
            G_cjl3 = Q_cjl3/(P_c3 - P_jl3)
            G_cjl2 = Q_cjl2/(P_c2 - P_jl2)

            G_vv2 = (Q_vvr+Q_vvl)/(P_vv - P_rv)

            G_c3 = 0 
            G_c2 = Q_c2/(P_c3 - P_c2)
            G_c1 = (Q_c2 - Q_cjr2 - Q_cjl2)/(P_c2 - P_cv_initial)

            G_ex = Q_ex/(P_pa_initial - P_vs)

            G_svc1 = (Q_jr3 + Q_cjr3 + Q_cjr2)/(P_jr2 - P_svc1) + (Q_jl3 + Q_cjl3 + Q_cjl2)/(P_jl2 - P_svc1)

            G_svc2 = (Q_jr3 + Q_cjr3 + Q_cjr2 + Q_jl3 + Q_cjl3 + Q_cjl2 + Q_vvr + Q_vvl)/(P_svc1 - P_svc)

            G_lv = Q_lv/(P_vv - P_azy)
            G_rv = Q_rv/(P_vv - P_svc) # should this be P[4]?

            G_vvr = Q_vvr/(P_vs - P_vv)
            G_vvl = Q_vvl/(P_vs - P_vv)

            G_0 = 1.0/R_0
            G_azy1 = Q_azy1/(P_vv - P_azy)
            G_azy2 = (Q_lv + Q_azy1)/(P_azy - P_svc1)

            G_svc = Q_svc/(P_vs - P_svc)

            "--------------------------------------loop timing------------------------------------------------------"

            #dt = controlPars.T # time step
            t = 0.0
            run_once = 0


            "-----------------------------empty arrays to store data---------------------------------------------"

            #G_c2 = []
            #G_c1 = []

            alphalist = []
            P_clist = []
            P_palist = []
            P_iclist = []
            P_vlist = []
            C_vilist = []
            x_autlist = []
            C_palist = []
            R_palist = []
            Q_flist = []
            Q_0list = []
            R_vslist = []
            R_palist = []
            C_iclist = []
            G_jr3list = []
            G_jr2list = []
            G_jr1list =[]
            G_jl3list = []
            G_jl2list = []
            G_jl1list =[]
            G_vvrlist = []
            G_vvllist = []
            G_c1list = []
            G_cjr3list = []
            G_cjr2list = []
            G_cjl3list = []
            G_cjl2list = []
            P_paP_iclist = []
            C_iclist = []
            C_vilist = []
            C_palist = []
            R_palist = []
            x_autlist = []
            Q_flist = []
            Q_0list = []
            Q_c3list =[]
            Q_c2list = []
            Q_c1list = []
            Q_jr3list = []
            Q_jr2list = []
            Q_jr1list = []
            Q_jl3list = []
            Q_jl2list = []
            Q_jl1list = []
            Q_svc1list = []
            Q_nlist = []
            Qlist = []
            tlist = []
            P_vslist = []
            tau_autlist = []
            P_paP_icreducedlist = []
            P_vP_iclist = []
            P_jr3list = []
            P_jr2list = []
            P_jr1list = []
            P_jl3list = []
            P_jl2list = []
            P_jl1list = []
            P_svclist= []
            P_svc1list = []
            P_c1list = []
            P_c2list = []
            P_c3list = []
            Q_vvrlist = []
            Q_vvllist = []
            Q_vvlist = []
            P_vvlist = []
            dP_vvdtlist = []
            Q_cjr3list = []
            Q_cjl3list = []
            dP_jr3dtlist = []
            dP_vP_icdtlist = []
            Q_rvlist = []
            Q_lvlist = []
            P_azylist = []
            P_vvlist = []
            Q_vslist = []
            Q_exlist = []
            G_c3list = []
            Q_list = []
            Q_totlist = []
            Q_svc2list = []
            Q_rvlist = []
            Q_azy2list = []
            Q_svc3list = []
            Qbrain_buffer = []
            Q_autolist=[]
            P_svc_reqlist = []
            Q_out_netlist = []
            #Qbrain_buffer=np.full(int(reflexPars.S_INT/controlPars.T),Q_n);
            # thrombotic events
            i_condition =0
            clot_formation =  0 # set this to 1 to enable thrombotic events to occur
            #i = 0 # i is defining different conditions for plot generation (simulating 0 conductances in vessels)
                # LvL, please give some more info on what eacht i means.

            dx_autdt = (1.0/tau_aut) * (-x_aut + G_aut * (Q_auto - Q_n) / Q_n) # initialize dx_autdt

    # Get all the pars from the different parameters values
    global subjectPars,controlPars,reflexPars, filename, sts_n
    subjectPars = _init_pars(); # Here the compartments parameters are assigned
    controlPars=_init_control(); # Get all the control parameters stored to the list 'control'.You can access the heart rate for instance by contorl.hr.
    reflexPars=_init_reflex(); # Get all the reflex parameters stored to the list 'reflexPars'.
    if cerebralModelOn==1:
        crb = cerebralPars()

    sit_index = find_index_of_time(data_time, sts0)
    stand_index = find_index_of_time(data_time, sts1)
    reflexPars.ABP_setp = sum(data_BP[sit_index:stand_index])/(stand_index-sit_index)*scaling["ABP_setp"]
    #print(reflexPars.ABP_setp)
    if cerebralModelOn==1:
        crb.P_a_n = reflexPars.ABP_setp
    
    # =============================================================================
    # Interventions
    # =============================================================================
    sit_index_10Hz = find_index_of_time(data_time_10Hz, round(sts0, 1))
    stand_index_10Hz = find_index_of_time(data_time_10Hz, round(sts1, 1))
    reflexPars.PP_setp = sum(data_SBP[sit_index_10Hz:stand_index_10Hz]-data_DBP[sit_index_10Hz:stand_index_10Hz])/(stand_index_10Hz-sit_index_10Hz)

    global delta_t, HP, tmax, T, resting_time, time_after_sts, limit, HR, abp_hist,pp_hist, rap_hist,para_resp,beta_resp,alpha_resp,alpha_respv,alphav_resp,alphav_respv
    global s_abp, abp_ma,rap_ma,s_rap,abp_pp,pp_abp, ErvMAX, ElvMAX, V_micro, t_eval, store_P, store_HR, sts_n_end
    
    startTimeStS=resting_time; # IF from zero G to 1 G (standing test) and when this should start (in sec).
    strainTime = -1; # Time when the strain should start (in sec).
    supine_or_sit = 0 # supine = 0, sit = 1
    if supine_or_sit == 0:
        alpha_0 = 0
        if math.isnan(sts_n_end):
            sts_n_end = sts_n + 7
        delta_t = sts_n_end-sts_n
    if supine_or_sit == 1:
        alpha_0 = 60
        if math.isnan(sts_n_end):
            sts_n_end = sts_n + 3
        delta_t = sts_n_end-sts_n
        
    TimeToStand = delta_t # [s]

    # =============================================================================
    # Initial settings; Heart rate, Intrathoracic pressure, Total blood volume and Unstressed blood volume.
    # =============================================================================
    HR = controlPars.hr*scaling["HR_t0"]
    HP=60/HR
    
    RR=controlPars.RR
    
    #P_intra_t0=controlPars.P_intra_t0;
    P_intra_t0=-4
    rrp=60/controlPars.RR; # Respiratory rate period
    #rrp=0
    P_sts = 15 * scaling["STS_pressure"] # mmHg
    #P_sts = 43 * scaling["STS_pressure"] # mmHg Source?
    """ Young adult:
    Standing up from the supine position is invariably accompanied by
    (involuntary) contraction of leg and abdominal muscles with a precip-
    Itous transient increase of intra-abdominal pressure (43 mmHg +- 22% on
    average in young adults) (Tanaka et al., 1996) and abrupt rise (about 10
    mmHg) of right atrial pressure resulting in an increase in right ventricular
    filling and therefore CO (Sprangers et al., 1991b)."""

    #P_sts_legs = 30 * scaling["STS_pressure"] # mmHg Arnoldi -> young person
    #P_sts_abdom = 43 * scaling["STS_pressure"] # mmHg Arnoldi -> young person
    P_sts_legs = 20 * scaling["STS_pressure"] # mmHg 
    P_sts_abdom = 10 * scaling["STS_pressure"] # mmHg 
    P_sts_thorax = 5 * scaling["STS_pressure"] # mmHg


    P_stand = 5 * scaling["Stand_muscleP"] # mmHg
    hydro1 = 3 # originally 3
    hydro2 = 2 # originally 2
    limit = 100 # What is an appropriate limit?  -> scales 1/limit^4 with resistance
    
    #controlPars.BW = controlPars.BW
    #controlPars.BW = 77
    controlPars.BW = weight
    BW = controlPars.BW
    controlPars.C_micro=BW*2.9; # ref. ?
    #controlPars.H = controlPars.H
    #controlPars.H = 1.76
    controlPars.H = length/100
    H = controlPars.H
    controlPars.tbv = 70/(np.sqrt((BW/(22*H**2)))) * BW # Lemmens-Bernstein-Brodsky Equation
    TBV=controlPars.tbv*scaling["v_ratio"]
    #TBV=controlPars.tbv
    
    # Nadler eq.:
    """
    if sex == 0:
        TBV2 = (0.3669 * H**3 + 0.03219 * BW + 0.6041) * 1000
    if sex == 1:
        TBV2 = (0.3561 * H**3 + 0.03308 * BW + 0.1833) * 1000
    #print("TBV1 and TBV2: ", TBV, TBV2)
    """
    
    v_ratio = TBV/5250
    h_ratio = H/1.7
    subjectPars.vessel_length=np.array(subjectPars.vessel_length)*h_ratio
    
    # Microcirculation start parameters
    #controlPars.V_micro = controlPars.V_micro
    controlPars.V_micro = controlPars.V_micro*v_ratio
    V_micro=controlPars.V_micro
    #V_micro2 = 11400
    V_micro2 = 11400*v_ratio

    # these 3 lines below ared to create a non-linear compliance in the venous compartment (10,12,13).
    Vmax10=1500
    Vmax12=1000
    Vmax13=150
       
    ### Individual scaling ###
    ### Resistance ###

    subjectPars.resistance[:, 0]=[0.007, np.nan, np.nan, 0.003, 0.011, np.nan]       # Ascending Aorta - ADJUSTED 150621 5,0 from 0.03 to 0.011
    subjectPars.resistance[:, 1]=[0.003, np.nan, np.nan, 0.014, np.nan, np.nan]        # Upper thoracic artery
    #subjectPars.resistance[:, 1]=[0.003, np.nan, np.nan, 0.014, R_car, np.nan]        # Upper thoracic artery
    subjectPars.resistance[:, 2]=[0.014, np.nan, np.nan, 4.9*scaling["Rp"], np.nan, np.nan]         # Upper body arteries
    subjectPars.resistance[:, 3]=[4.9*scaling["Rp"], np.nan, np.nan, 0.11, np.nan, np.nan]          # Upper body veins
    subjectPars.resistance[:, 4]=[0.11, np.nan, np.nan, 0.028, np.nan, np.nan]         # Super vena cava - ADJUSTED 150621 3, 4 from 0.06 to 0.028
    subjectPars.resistance[:, 5]=[0.011, np.nan, np.nan, 0.01, np.nan, np.nan]        # Lower thoracic artery  - ADJUSTED 150621 0, 5 from 0.03 to 0.011
    subjectPars.resistance[:, 6]=[0.01, np.nan, np.nan, 0.1, 0.03, 0.09]            # Abdominal aorta - ADJUSTED 150621 0, 5 from 0.03 to 0.01
    subjectPars.resistance[:, 7]=[0.1, np.nan, np.nan, 4.1*scaling["Rp"], np.nan, np.nan]         # Renal arteries
    subjectPars.resistance[:, 8]=[4.1*scaling["Rp"], np.nan, np.nan, 0.11, np.nan, np.nan]         # Renal veins
    subjectPars.resistance[:, 9]=[0.03, np.nan, np.nan, 3*scaling["Rp"], np.nan, np.nan]          # Splanchnic arteries
    subjectPars.resistance[:, 10]=[3*scaling["Rp"], np.nan, np.nan, 0.07, np.nan, np.nan]          # Splanchnic veins
    subjectPars.resistance[:, 11]=[0.09, np.nan, np.nan, 4.5*scaling["Rp"], np.nan, np.nan]         # Lower body arteries
    subjectPars.resistance[:, 12]=[4.5*scaling["Rp"], np.nan, np.nan, 0.1, np.nan, np.nan]          # Lower body veins
    subjectPars.resistance[:, 13]=[0.11, 0.07, 0.1, 0.019, np.nan, np.nan]             # Abdominal veins
    subjectPars.resistance[:, 14]=[0.019, np.nan, np.nan, 0.008, np.nan, np.nan]     # Inferior vena cava
    subjectPars.resistance[:, 15]=[0.008, 0.028, np.nan, 0.006, np.nan, np.nan]       # Right atrium
    subjectPars.resistance[:, 16]=[0.006, np.nan, np.nan, 0.003, np.nan, np.nan]     # Right ventricle
    subjectPars.resistance[:, 17]=[0.003, np.nan, np.nan, 0.07*scaling["Rp"], np.nan, np.nan]      # Pulmonary artery
    subjectPars.resistance[:, 18]=[0.07*scaling["Rp"], np.nan, np.nan, 0.006, np.nan, np.nan]       # Pulmonary veins
    subjectPars.resistance[:, 19]=[0.006, np.nan, np.nan, 0.01, np.nan, np.nan]     # Left atrium
    subjectPars.resistance[:, 20]=[0.01, np.nan, np.nan, 0.007, np.nan, np.nan]    # Left ventricle

    ### Elastance ###
    inputs[0:3] = inputs[5:8] = inputs[9] = inputs[11] = inputs[17] = scaling["E_arteries"]
    inputs[3:5] = inputs[8] = inputs[10] = inputs[12:15] = inputs[18] = scaling["E_veins"]
    subjectPars.elastance=np.array(subjectPars.elastance)*inputs[0:22] # heart elastanceses are included,

    ### Unstressed volume ###
    #subjectPars.uvolume=np.array(subjectPars.uvolume)*scaling["Global_UV"]
    subjectPars.uvolume=np.array(subjectPars.uvolume) * scaling["Global_UV"]

    ### Assign the Heart elastances. ###
    subjectPars.elastance[0,16] = subjectPars.elastance[0,16] * scaling["max_RV_E"]
    subjectPars.elastance[1,16] = subjectPars.elastance[1,16] * scaling["min_RV_E"]
    subjectPars.elastance[0,20] = subjectPars.elastance[0,20] * scaling["max_LV_E"]
    subjectPars.elastance[1,20] = subjectPars.elastance[1,20] * scaling["min_LV_E"]

    # Make sure Emin does not become larger than Emax
    if subjectPars.elastance[0,15] < 1.05*subjectPars.elastance[1,15]:
        subjectPars.elastance[0,15] = 1.05*subjectPars.elastance[1,15]
        print("LIMIT")
    if subjectPars.elastance[0,16] < 1.05*subjectPars.elastance[1,16]:
        subjectPars.elastance[0,16] = 1.05*subjectPars.elastance[1,16]
        print("LIMIT")
    if subjectPars.elastance[0,19] < 1.05*subjectPars.elastance[1,19]:
        print("Left Atrium LIMIT! EMin = ", subjectPars.elastance[1,19], "EMax = ", subjectPars.elastance[0,19])
        subjectPars.elastance[0,19] = 1.05*subjectPars.elastance[1,19]
    if subjectPars.elastance[0,20] < 1.05*subjectPars.elastance[1,20]:
        subjectPars.elastance[0,20] = 1.05*subjectPars.elastance[1,20]
        print("LIMIT")
        
    ### More scaling ###
    subjectPars.resistance=np.array(subjectPars.resistance)*scaling["Global_R"]
    #subjectPars.vessel_length=np.array(subjectPars.vessel_length)
    
    TBUV=np.sum(subjectPars.uvolume)
    
    global HP_min, HP_max, R3, R8, R10, R12, UV3, UV8, UV10, UV12, ErvNorm, ElvNorm, ElaMIN, ElaMAX, ElvMIN, ErvMAX, ElvMAX, EraMIN, EraMAX, ErvMIN
    R3 = subjectPars.resistance[0,3].copy()
    R8 = subjectPars.resistance[0,8].copy()
    R10 = subjectPars.resistance[0,10].copy()
    R12 = subjectPars.resistance[0,12].copy()
    
    UV3 = subjectPars.uvolume[0,3].copy()
    UV8 = subjectPars.uvolume[0,8].copy()
    UV10 = subjectPars.uvolume[0,10].copy()
    UV12 = subjectPars.uvolume[0,12].copy()
    
    ElaMIN=subjectPars.elastance[1,19].copy()
    ElaMAX=subjectPars.elastance[0,19].copy()
    ElvMIN=subjectPars.elastance[1,20].copy()
    ElvMAX=subjectPars.elastance[0,20].copy()
    
    EraMIN=subjectPars.elastance[1,15].copy()
    EraMAX=subjectPars.elastance[0,15].copy()
    ErvMIN=subjectPars.elastance[1,16].copy()
    ErvMAX=subjectPars.elastance[0,16].copy()

    HP_min = 1.5 # minimal heart period
    HP_max = 0.2 # maximal heart period

    ### IMPULSE FACTORS ###
    # More setpoints
    reflexPars.RAP_setp = 3 * scaling["RAP_setp"] # vec[15] -> some influence
    reflexPars.ABRsc = 18 # vec[1] -> little influence
    reflexPars.RAPsc = 5  # vec[16] -> very little influence

    # Impulse responses
    reflexPars.p = reflexPars.p
    reflexPars.s = reflexPars.s
    reflexPars.a = reflexPars.a
    reflexPars.v = reflexPars.v
    reflexPars.cpv = reflexPars.cpv
    reflexPars.cpa = reflexPars.cpa
    
    # Gains
    reflexPars.RRsgain = reflexPars.RRsgain * scaling["RRsgain"] # vec[2]
    reflexPars.RRpgain = reflexPars.RRpgain * scaling["RRpgain"] # vec[3]
    #reflexPars.rr_sym = reflexPars.rr_sym #vec[2] // beta -> not used
    #reflexPars.rr_para = reflexPars.rr_para #vec[3] // para -> not used
    reflexPars.beta = reflexPars.beta * scaling["beta"]
    reflexPars.alpha = reflexPars.alpha * scaling["alpha"]
    gain1 = 0.021 * scaling["beta_resp_Erv"]
    gain2 = 0.014 * scaling["beta_resp_Elv"]
    gain3 = -.13 * scaling["alpha_resp"]
    gain4 = -.3 * scaling["alphav_resp"]
    
    """
    Assign the parameters
    """
    #CARDIAC CYCLE TIMER AND COUTER INITIALIZATIONS
    global t_cc_onset, cc_switch, t_rf_onset, idx_check, t_resp_onset
    t_cc_onset = np.array([0.0])
    cc_switch = 0
    t_rf_onset = 0
    idx_check = -1
    t_resp_onset = 0.0
    
    """
     STATE VARIABLE INITIALIZATION
    """
    global V, P, F, Pgrav, E, R, UV, vl
    # The value below are not stored every moment, only the most recent one.
    V=np.zeros((22)) # number of compartments+1, this 21+1=22 (+ V_micro);
    P=np.zeros((22))
    F=np.zeros((24))
    #E=np.zeros((21))
    Pgrav=np.zeros((22))
    E=np.array(subjectPars.elastance[0,:])
    #E=np.array(subjectPars.elastance[0,:])*C_age
    R=np.array(subjectPars.resistance)
    UV=np.array(subjectPars.uvolume[0,:])
    vl=np.array(subjectPars.vessel_length[0,:])
    
    global abp_temp, cvp_temp, co_temp, hr_temp, finap_temp, impulse, Out_av, Out_wave, store_t, store_P_muscle, store_P_intra, HR_list
    global store_E, store_crb, store_BP_min, store_BP_max, store_UV,store_P_muscle2
    global store_crb_Q_ic, store_crb_Q_j, store_crb_Q_v, store_crb_Q_out, store_crb_P, store_crb_C, store_crb_R, store_crb_G, store_crb_x, store_crb_mca
    # Initialize arrays to store the values
    store_t=[0.0]
    store_BP_min=[]
    store_BP_max=[]
    store_finap=[]
    store_P = np.zeros((22, len(t_eval)))
    store_HR = np.zeros(len(t_eval))
    store_P_muscle = np.zeros(len(t_eval))
    store_P_intra = np.zeros(len(t_eval))
    store_P_muscle2 = np.zeros((4, len(t_eval)))
    store_E = np.zeros((4, len(t_eval)))
    store_UV = np.zeros((22, len(t_eval)))
    Out_av=[]
    Out_wave=[]
    HR_list=[1/HP*60]
    
    store_crb = np.zeros((len(t_eval)))
    store_crb_Q_ic = np.zeros((7, len(t_eval))) # store intracranial flows
    store_crb_Q_j = np.zeros((9, len(t_eval))) # store jugular flows
    store_crb_Q_v = np.zeros((3, len(t_eval))) # store vertebral flows
    store_crb_Q_out = np.zeros((7, len(t_eval))) # store out-flows: azygos & svc flows
    store_crb_P = np.zeros((2, len(t_eval))) # Store pressure in the cerebral veins and pial arterioles
    store_crb_C = np.zeros((3, len(t_eval))) # store intracranial capacities
    store_crb_R = np.zeros((2, len(t_eval))) # store intracranial resistances
    store_crb_G = np.zeros((12, len(t_eval))) # store jugular conductances
    store_crb_x = np.zeros(len(t_eval)) # State variable of the autoregulation mechanism related to cerebral flow variations
    store_crb_mca = np.zeros((2, len(t_eval))) # store velocity and radius of middle cerebral artery
    if cerebralModelOn==1:
        crb.Qbrain_buffer = np.full(int(crb_buffer_size/T), crb.Q_n)

    global abp_buffer, rap_buffer, abp_ma, abp_pp, rap_ma, abp_hist, rap_hist, abp_hist, rap_hist, pp_hist, baro_buffer
    #initilize the arrays needed for the relfexes
    abp_buffer=np.full(int(reflexPars.S_GRAN/T), reflexPars.ABP_setp) # NEW: I changed this back to S_GRAN. OLD: I artifically increased the buffer time, to get at least one heart beat. Becuase baroreceptor sense MAP not SYS.
    rap_buffer=np.full(int(reflexPars.S_GRAN/T), reflexPars.RAP_setp)
    abp_ma=np.full(baro_buffer, reflexPars.ABP_setp) # Heldt uses 3*S_GRAN: 2*S_GRAN convolution and 1*S_GRAN lin_interpolation
    abp_pp=np.full(baro_buffer, 30) # Heldt uses 3*S_GRAN: 2*S_GRAN convolution and 1*S_GRAN lin_interpolation
    rap_ma=np.full(baro_buffer, reflexPars.RAP_setp)  # Heldt uses 3*S_GRAN: 2*S_GRAN convolution and 1*S_GRAN lin_interpolation
    abp_hist=np.full(int(60/reflexPars.S_GRAN), 0) 
    rap_hist=np.full(int(60/reflexPars.S_GRAN), 0)
    pp_hist=np.full(int(60/reflexPars.S_GRAN), 0)
    beta_resp, para_resp, alpha_resp, alpha_respv, alphav_resp, alphav_respv = 0,0,0,0,0,0
    
    def impulseFunction(abp_buffer,rap_buffer):
        global abp_hist,pp_hist, rap_hist, para_resp,beta_resp,alpha_resp,alpha_respv,alphav_resp,alphav_respv, s_abp, abp_ma,rap_ma,s_rap,abp_pp,pp_abp
        global para_resp_old, beta_resp_old, alpha_resp_old, alpha_respv_old, alphav_resp_old, alphav_respv_old
        global para_resp_new, beta_resp_new, alpha_resp_new, alpha_respv_new, alphav_resp_new, alphav_respv_new
        para_resp_old = para_resp
        beta_resp_old = beta_resp
        alpha_resp_old = alpha_resp
        alpha_respv_old = alpha_respv
        alphav_resp_old = alphav_resp
        alphav_respv_old = alphav_respv
        # Step II -> Get a moving average filter over the buffer with a width of 0.25 sec. abp_ma of size 4 (* 0.0625 s)
        abp_ma = np.append(abp_ma[1:], np.mean(abp_buffer))
        #abp_pp = np.append(abp_pp[1:],np.max(abp_buffer)-np.min(abp_buffer)) # THIS IS WRONG, the PP cannot be determined in 0.25 seconds
        rap_ma = np.append(rap_ma[1:], np.mean(rap_buffer))
        # Step III -> Create a saturation curve of the bloodpressure, this curves covers 0.25 seconds.
        s_abp = np.arctan((np.mean(abp_ma) - reflexPars.ABP_setp) / reflexPars.ABRsc) * reflexPars.ABRsc #* control.ABRsc; 
        #pp_abp = np.arctan((np.mean(abp_pp) - reflexPars.PP_setp) / reflexPars.ABRsc) * reflexPars.ABRsc #* control.ABRsc; 
        s_rap = np.arctan((np.mean(rap_ma) - reflexPars.RAP_setp) / reflexPars.RAPsc) * reflexPars.RAPsc
        # Step IV -> stack these curves into a new vector with the length of 60 seconds/0.0625 = 960 elements
        abp_hist = np.append(abp_hist[1:], s_abp)
        #pp_hist = np.append(pp_hist[1:],np.mean(s_abp+pp_abp))
        rap_hist = np.append(rap_hist[1:], s_rap)
        # Step IV -> get the convolution of the impulse response and the blood pressure saturation curve
        para_resp_new=np.sum(np.flip(abp_hist,0)* reflexPars.p) # pp_hist changed to abp_hist
        beta_resp_new=np.sum(np.flip(abp_hist,0)* reflexPars.s) # pp_hist changed to abp_hist
        alpha_resp_new=np.sum(np.flip(abp_hist,0)* reflexPars.a) # pp_hist changed to abp_hist
        alpha_respv_new=np.sum(np.flip(abp_hist,0)* reflexPars.v) # pp_hist changed to abp_hist
        alphav_resp_new=np.sum(np.flip(rap_hist,0)* reflexPars.cpa)
        alphav_respv_new=np.sum(np.flip(rap_hist,0)* reflexPars.cpv)
    
    def ABRreflexDef():
        global HP, HR, ErvMAX, ElvMAX
        if HR==0:
            HP=0
        else:
            HP=(60/HR)+beta_resp*reflexPars.RRsgain+para_resp*reflexPars.RRpgain
            # Check if HP is NaN and set to HP_max if it is
            if math.isnan(HP):
                HP = HP_max
            # Set some limits
            if HP < HP_max: # the heart rate can't be higher then 300
                HP=HP_max
            if HP > HP_min: # the heart rate cannot be lower then 40
                HP=HP_min
        # Contractility feedback. Limit contractility feedback so end-systolic ventricular elastances do not become too large during severe stress.
        ErvMAX=1/(1/subjectPars.elastance[0,16]+beta_resp*gain1)
        if 1/ErvMAX < .01:
            ErvMAX=1/.01
        ElvMAX=1/(1/subjectPars.elastance[0,20]+beta_resp*gain2)
        if 1/ElvMAX<.05:
            ElvMAX=1/.05
        return ErvMAX, ElvMAX, HP
    
    def CPRreflexDef():
        # Upper body compartment
        R[0,3] = subjectPars.resistance[0,3] + gain3*alpha_resp + gain4*alphav_resp #  vec[4]=-.13  vec[17]=-0.3
        # R kidney compartment
        R[0,8] = subjectPars.resistance[0,8] + gain3*alpha_resp + gain4*alphav_resp #  vec[5]=-.13  vec[18]=-.3
        # R splanchnic compartment
        R[0,10] = subjectPars.resistance[0,10] + gain3*alpha_resp + gain4*alphav_resp #  vec[6]=-.13  vec[19]=-.3
        # R lower body compartment
        R[0,12] = subjectPars.resistance[0,12] + gain3*alpha_resp + gain4*alphav_resp #  vec[7]=-0.13  vec[20]=-.3
        
        # Limit R, what is an appropriate limit?
        R[0,3] = R_lim(R[0,3], R3)
        R[0,8] = R_lim(R[0,8], R8)
        R[0,10] = R_lim(R[0,10], R10)
        R[0,12] = R_lim(R[0,12], R12)
        
        # Venous tone feedback implementation.
        # Upper body veins compartment 
        UV[3] = subjectPars.uvolume[0,3] + alpha_respv*5.3*reflexPars.alpha + alphav_respv*13.5*reflexPars.beta #vec[8]=5.3, vec[21]=13.5
        # Renal veins
        UV[8] = subjectPars.uvolume[0,8] + alpha_respv*1.3*reflexPars.alpha + alphav_respv*2.7*reflexPars.beta #vec[9]=1.3, vec[22]=2.7
        # Splanchnic veins
        UV[10] = subjectPars.uvolume[0,10] + alpha_respv*13.3*reflexPars.alpha + alphav_respv*64*reflexPars.beta #vec[10]=13.3, vec[23]=64
        # Lower body veins
        UV[12] = subjectPars.uvolume[0,12] + alpha_respv*6.7*reflexPars.alpha + alphav_respv*30*reflexPars.beta #vec[11]=6.7, vec[24]=30
        """
        UV[3] = UV_lim(UV[3], UV3)
        UV[8] = UV_lim(UV[8], UV8)
        UV[10] = UV_lim(UV[10], UV10)
        UV[12] = UV_lim(UV[12], UV12)
        """

    """
         RUN TIME EQUATIONS
    """
    global alpha_tilt, P_muscle_thorax, hydroP, P_muscle_abdom, P_muscle_legs
    alpha_tilt = alpha_0
    #P_muscle = 0 # 
    P_muscle_abdom = 0 #
    P_muscle_legs = 0 #
    P_muscle_thorax = 0 #
    EarthG = 9.81
    MarsG = 3.721
    rho = 1060; # Density of blood is 1060 g/ml thus 1060 kg/m^3
    ConvertNtommHg = 0.0075
    if planet==2:
        Gforce = MarsG
    else:
        Gforce = EarthG
    # hydrostatic gravity = rho * g * h = density of the blood (g/L) * gravity (m/s^2) * height (m) = N/m2
    #How to convert Newton Per Square Meter to Millimeter Of Mercury (N/m2 to mmHg)? 1 N/m2 = 0.0075006375541921 mmHg.
    hydroP = Gforce*rho*np.sin(alpha_0/360*2*np.pi)*ConvertNtommHg; 
    for x in range(0,15): # zero up to and including 14
        if x in [6,11,12,13]: # the legs, IVC, and adb aorto have an even stronger gravity effect
            Pgrav[x]=(vl[x]/100)/hydro1*hydroP
        else:
            Pgrav[x]=(vl[x]/100)/hydro2*hydroP
    Pgrav[21]=(vl[21]/100)/hydro2*hydroP # Carotid artery
            
    """ Initialize cardiac parameters """  
    global Tav, Tsa, Tsv, K_av, K_sa, K_sv
    K_av = 0.12 * scaling["SD_ratio"]
    K_sa = 0.2 * scaling["SD_ratio"]
    K_sv = 0.3 * scaling["SD_ratio"]
        
    Tav=K_av*np.sqrt(HP)
    Tsa=K_sa*np.sqrt(HP)
    Tsv=K_sv*np.sqrt(HP)

    """ Ventilation parameters """
    IEratio=0.6;#% [MODEL]  6
    RRP =60 /RR
    TI= RRP/(1+1/IEratio)
    TE = RRP-TI
    #Pmusmin=-2;#% cmH2O [MODEL]
    Pmusmin=-2;#% cmH2O [MODEL]
    tau = TE/5 ;#%[MODEL]
    
    """ Fluid loading """
    if fluidLoading==1:
        TBV=TBV*1.05
    
    """ Initialize volumes """
    # this is used to calculate the inital volume of the compartments, based on their relative unstressed volumes
    for x in range(0,len(V)-1): # this loops from 0 to 21, excluding V_micro
        V[x]=TBV*(UV[x]/TBUV)
        #V[x]=UV[x]
    V[21] = V_micro
    
    """!!!
        Start the simulation
    """
    def esse_cerebral_NEW(t, x_esse):
        global nrr, ncc, nrf, n, n4, HP, V_micro, ElvMIN, ErvMIN, ErvMAX, ElvMAX
        global abp_buffer, rap_buffer, abp_ma, abp_pp, rap_ma, abp_hist, rap_hist, abp_hist, rap_hist, pp_hist
        global V, P, F, Pgrav, E, R, UV, vl, t_temp, cvp_temp, co_temp, hr_temp, store_finap_temp
        global hydroP, alpha_tilt, Tav, Tsa, Tsv, P_muscle_thorax, P_muscle_abdom, P_muscle_legs, P_intra, F_micro_lower, F_micro_upper, F_lymphatic
        global cc_switch, t_rf, t_rf_onset, t_cc, t_cc_onset, t_resp_onset, impulse
        global para_resp,beta_resp,alpha_resp,alpha_respv,alphav_resp,alphav_respv, idx_check
        
        """ Update values """
        V = x_esse[:22]  # Volumes of cardiovascular model
        V_micro = V[-1] # V_micro
        if cerebralModelOn==1:
            crb.x_aut = x_esse[22] # autoregulation
            crb.P_vP_ic = x_esse[23] # pressure of cerebal veins vi from equation (5)
            crb.P_paP_ic = x_esse[24] # pressure of pial arterioles from equation (3)
            crb.P_ic = x_esse[25] # intracranial pressure from equation (15)
            crb.P_vs = x_esse[26] # cerebral sinus veins pressure
            if carotidOn==1:
                crb.P_car = x_esse[27] # carotid artery pressure
            if cerebralVeinsOn==1:
                crb.P_jr3 = x_esse[27+carotidOn] # 3rd segment of the right jugular vein pressure
                crb.P_jl3 = x_esse[28+carotidOn] # 3rd segment of the left jugular vein pressure
                crb.P_jr2 = x_esse[29+carotidOn] # 2nd segment of the right jugular vein pressure
                crb.P_jl2 = x_esse[30+carotidOn] # 2nd segment of the left jugular vein pressure
                crb.P_c3 = x_esse[31+carotidOn] # pressure in upper segment of the collateral network
                crb.P_c2 = x_esse[32+carotidOn] # pressure in middle segment of the collateral network
                crb.P_svc1 = x_esse[33+carotidOn] # pressure in the superior segment of the superior vena cava
                crb.P_azy = x_esse[34+carotidOn] # pressure in the azygos system
                crb.P_svc = x_esse[35+carotidOn] # testing?
                crb.P_vv = x_esse[36+carotidOn] # pressure in the vertebral vein
            crb.P_v = crb.P_vP_ic + crb.P_ic # venous pressure ?
            crb.P_pa = crb.P_paP_ic + crb.P_ic # pial arterioles pressure

        """ Make sure there is no volume added or removed as a result of the integration """
        sum_v = np.sum(V[:-1]) 
        
        # Check if the value is NaN and terminate if it is
        if math.isnan(sum_v):
            #print("The value is NaN. Terminating the script.")
            sys.exit()
        
        if t>1: # after 1 second; check the volumes
            V[13]=V[13]+TBV-sum_v+controlPars.V_micro-V_micro
        
        """ The baroreflex and CPReflex """
        t_rf = t - t_rf_onset # time in refelx function
        if t_rf>=reflexPars.S_GRAN and (ABRreflexOn==1 or CPRreflexOn==1): # Call every 0.0625 s
            t_rf_onset = t # New onset time reflex function
            t_rf = 0 # New time in new reflex function
            rf_switch = 1 # Reflex function switch ON
            # create impulse response
            impulseFunction(abp_buffer,rap_buffer)

        if t>=4*reflexPars.S_GRAN: # Linear interpolation between impulse activations
            if ABRreflexOn==1:
                para_resp = (para_resp_new-para_resp_old)*t_rf/reflexPars.S_GRAN+para_resp_old
                beta_resp = (beta_resp_new-beta_resp_old)*t_rf/reflexPars.S_GRAN+beta_resp_old
                alpha_resp = (alpha_resp_new-alpha_resp_old)*t_rf/reflexPars.S_GRAN+alpha_resp_old
                alpha_respv = (alpha_respv_new-alpha_respv_old)*t_rf/reflexPars.S_GRAN+alpha_respv_old
                ErvMAX, ElvMAX, HP = ABRreflexDef()
            if CPRreflexOn==1:
                alphav_resp = (alphav_resp_new-alphav_resp_old)*t_rf/reflexPars.S_GRAN+alphav_resp_old
                alphav_respv = (alphav_respv_new-alphav_respv_old)*t_rf/reflexPars.S_GRAN+alphav_respv_old
                CPRreflexDef()
                
        """ Cardiac Cycle (CC) """
        t_cc = t - t_cc_onset[-1] # Time in cardiac cycle
        if t_cc >= HP: # If 1 CC has past:
            HR_list.append(1/HP*60)
            t_cc_onset = np.concatenate((t_cc_onset, [t_cc_onset[-1] + HP]), axis=None) # New onset time cardiac cycle
            t_cc = t - t_cc_onset[-1] # New time in new cardiac cycle
            cc_switch = 1 # Cardiac cycle switch ON
            Tav = K_av*np.sqrt(HP)
            Tsa = K_sa*np.sqrt(HP)
            Tsv = K_sv*np.sqrt(HP)
        # Atria
        if t_cc <= Tsa and t_cc >=0:
            E[19] = 0.5*(ElaMAX-ElaMIN)*(1-np.cos(np.pi*t_cc/Tsa))+ElaMIN
            E[15] = 0.5*(EraMAX-EraMIN)*(1-np.cos(np.pi*t_cc/Tsa))+EraMIN
        elif t_cc>Tsa and t_cc<=(Tsa*1.5):
            E[19] = 0.5*(ElaMAX-ElaMIN)*(1+np.cos(2*np.pi*t_cc/Tsa))+ElaMIN
            E[15] = 0.5*(EraMAX-EraMIN)*(1+np.cos(2*np.pi*t_cc/Tsa))+EraMIN
        else:
            E[19] =ElaMIN
            E[15] =EraMIN
        # Ventricle
        if t_cc < Tav:
            E[20] =ElvMIN
            E[16] =ErvMIN
        elif t_cc <= (Tav+Tsv):
            E[20] = 0.5*(ElvMAX-ElvMIN)*(1-np.cos(np.pi*(t_cc-Tav)/Tsv))+ElvMIN
            E[16] = 0.5*(ErvMAX-ErvMIN)*(1-np.cos(np.pi*(t_cc-Tav)/Tsv))+ErvMIN
        elif t_cc > (Tsv+Tav) and t_cc <= ((Tsv+Tav)*3/2):
            E[20] = 0.5*(ElvMAX-ElvMIN)*(1+np.cos(2*np.pi*(t_cc-(Tav+Tsv))/(Tav+Tsv)))+ElvMIN
            E[16] = 0.5*(ErvMAX-ErvMIN)*(1+np.cos(2*np.pi*(t_cc-(Tav+Tsv))/(Tav+Tsv)))+ErvMIN
        else:
            E[20] =ElvMIN
            E[16] =ErvMIN

    # =============================================================================
    #    Supine to standing test
    # =============================================================================

        if StStest==1:

            tau_STS=TimeToStand; 
            startTimeStrain = startTimeStS+strainTime
            
            if t >= startTimeStS and t<(startTimeStS+tau_STS):
                #print("STA!")
                #alpha_tilt = 90*np.sin(.5*np.pi*((n-startTimeStS/T))/(tau_STS/T))* np.pi / 180.0; # Smooth transition from standing to lying down.  Where tau_STS is a constant of the total transition time.
                tilt_time=(t-startTimeStS)
                
                alpha_rad_half = (1.0 - np.cos(np.pi * tilt_time / tau_STS)) / 2
                #alpha_rad_full = (1.0 - np.cos(2 * np.pi * tilt_time / tau_STS)) / 2
                alpha_tilt = alpha_0 + (90 - alpha_0) * alpha_rad_half

                hydroP = Gforce*rho*np.sin(alpha_tilt/360*2*np.pi)*ConvertNtommHg

                for x in range(0,15): # up to and including 14
                    if x in [6,11,12,13]: 
                        Pgrav[x]=(vl[x]/100)/hydro1*hydroP # The external pressure from muscles during standing up was simulated by deviding the hydro by 3 for the leg and abdominal compartments.
                    else:
                        Pgrav[x]=(vl[x]/100)/hydro2*hydroP
                Pgrav[21]=(vl[21]/100)/hydro2*hydroP
                        
                #alpha_tilt = 90*np.sin(.5*np.pi*((n-startTimeStS/T))/(tau_STS/T))* np.pi / 180.0; # Smooth transition from standing to lying down.  Where tau_STS is a constant of the total transition time.

            if t >= startTimeStrain and t<(startTimeStrain+tau_STS): # -1, because people strain before standing up
                tilt_time2=(t-startTimeStrain)
                power = 1
                alpha_rad_half2 = (1.0 - np.cos(np.pi * tilt_time2 / tau_STS)) / 2
                alpha_rad_full = alpha_rad_full_func(tilt_time2, tau_STS, p=power)
                #P_muscle = P_sts * 0.5 * ( 1 - np.cos(2 * np.pi * tilt_time2 / tau_STS)) + 0.5 * P_stand * (1 - np.cos( np.pi * tilt_time2 / tau_STS))
                P_muscle_legs = P_sts_legs * alpha_rad_full + P_stand * alpha_rad_half2
                P_muscle_abdom = P_sts_abdom * alpha_rad_full
                #P_muscle_thorax = P_sts * 0.5 * ( 1 - np.cos(2 * np.pi * tilt_time2/tau_STS))
                P_muscle_thorax = P_sts_thorax * alpha_rad_full


        """ Respiratory pattern + influence of gravity on this pressure """
        t_resp = t - t_resp_onset # Time in respiratory cycle
        if t_resp > RRP: # If 1 RRP has past:
            t_resp_onset = t
            t_resp = t - t_resp_onset
        if t_resp <= TI: # inspiration
            P_intra = P_muscle_thorax + P_intra_t0 + ((-Pmusmin/(TI*TE))*(t_resp)**2+((Pmusmin*rrp)/(TI*TE))*(t_resp))-(np.sin(alpha_tilt/(2*np.pi))); 
        else: # expiration
            P_intra = P_muscle_thorax + P_intra_t0 + ((Pmusmin/(1-np.exp(-TE/tau)))*(np.exp(-((t_resp)-TI)/tau)-np.exp(-TE/tau)))-(np.sin(alpha_tilt/(2*np.pi))); 
        
        """ Calculate the pressures """
        # This equations calculated the new pressure based on the compartment's Elastance, volume and unstressed volume.
        # for the intra-thoracic compartments: add the intra-thoracic pressure
        P[0]=(E[0]*(V[0]-UV[0])+P_intra)
        P[1]=(E[1]*(V[1]-UV[1])+P_intra)
        P[4]=(E[4]*(V[4]-UV[4])+P_intra)
        P[5]=(E[5]*(V[5]-UV[5])+P_intra)
        P[14]=(E[14]*(V[14]-UV[14])+P_intra)
        P[15]=(E[15]*(V[15]-UV[15])+P_intra)
        P[16]=(E[16]*(V[16]-UV[16])+P_intra)
        P[17]=(E[17]*(V[17]-UV[17])+P_intra)
        P[18]=(E[18]*(V[18]-UV[18])+P_intra)
        P[19]=(E[19]*(V[19]-UV[19])+P_intra)
        P[20]=(E[20]*(V[20]-UV[20])+P_intra)
        P[10]=(np.tan((V[10]-UV[10])/(2*Vmax10/np.pi)))/(np.pi*(1/E[10])/(2*Vmax10))+P_muscle_abdom
        P[12]=(np.tan((V[12]-UV[12])/(2*Vmax12/np.pi)))/(np.pi*(1/E[12])/(2*Vmax12))+P_muscle_legs
        P[13]=(np.tan((V[13]-UV[13])/(2*Vmax13/np.pi)))/(np.pi*(1/E[13])/(2*Vmax13))+P_muscle_abdom
        # calculate the pressures of the extrathoracic compartments
        P[2]=(E[2]*(V[2]-UV[2]))
        P[3]=(E[3]*(V[3]-UV[3]))
        P[6]=(E[6]*(V[6]-UV[6]))
        P[7]=(E[7]*(V[7]-UV[7]))
        P[8]=(E[8]*(V[8]-UV[8]))+P_muscle_abdom
        P[9]=(E[9]*(V[9]-UV[9]))
        P[11]=(E[11]*(V[11]-UV[11]))

        if cerebralModelOn==1:
            "----------------------------------blood clot modelling------------------------------------------------"

            #clot_start = 10 # choose time for clot to start

            #if crb.crb.clot_formation > 0:
            #    if t>crb.clot_start:
            #        crb.G_jr1 = 0    # choose clot location by setting the conductance of a compartment to 0
            #        crb.G_jl1 = 0
            #        # G_azy2 = 0 

            
            "-------------------------------intracranial equations-----------------------------------------------"
            """
            #if len(crb.Qbrain_buffer) > int(2000): # Cerebral
            #    crb.Q_auto = np.mean(crb.Qbrain_buffer[-2000:])
                #print("AAN")
            # differential equation describing autoregulation from equation (7)
            # Cerebral autoregulation is a function of flow in this model. Something we should discus in the manuscript.
            #if len(crb.Qbrain_buffer) > int(int(reflexPars.S_INT/controlPars.T)):
            #    crb.Q_auto = np.mean(crb.Qbrain_buffer[-(int(reflexPars.S_INT/controlPars.T)):])
            #if len(crb.Qbrain_buffer) > int(2000):
            #    crb.Q_auto = np.mean(crb.Qbrain_buffer[-2000:])
            crb.Q_auto = np.mean(crb.Qbrain_buffer)
            crb.dx_autdt = (1.0/crb.tau_aut) * (-crb.x_aut + crb.G_aut * (crb.Q_auto - crb.Q_n) / crb.Q_n) 

            #crb.dx_autdt = (1.0/crb.tau_aut)*(-crb.x_aut + crb.G_aut*(crb.Q_la-crb.Q_n)/crb.Q_n) 
            # when I calculate Q, Q_n = 12.5 causes the model to fail during the supine to standing test. 
            """

            # calculate autoregulation
            #crb.x_aut += crb.dx_autdt*crb.dt 
            # conditions for vasoconstriction/vasodilation
            if crb.x_aut <= 0.0: # vasodilation, added OR EQUALS
                crb.DeltaC_pa = crb.DeltaC_pa1
                crb.kC_pa = crb.DeltaC_pa1/4.0
            if crb.x_aut > 0.0: #vasocontriction
                crb.DeltaC_pa = crb.DeltaC_pa2
                crb.kC_pa = crb.DeltaC_pa2/4.0
                
            # calculate intranical capacity
            crb.C_ic = 1.0/(crb.k_E*crb.P_ic)
            
            # calculate cerebral veins vi capacity from equation (6)
            crb.C_vi = 1.0/(crb.k_ven*(crb.P_v - crb.P_ic - crb.P_v1))
            
            # calculate pial arteries capacity from equation (8)
            crb.C_pa = (crb.C_pan-crb.DeltaC_pa/2.0 + (crb.C_pan + crb.DeltaC_pa/2.0)*np.exp(-crb.x_aut/crb.kC_pa))/(1.0+np.exp(-crb.x_aut/crb.kC_pa))

            # Formula uses dx_autdt from the previous timestep, but x_aut from the current timestep, is this correct?
            crb.dC_padt = (-crb.DeltaC_pa*crb.dx_autdt)/(crb.kC_pa*(1+np.exp(-crb.x_aut/crb.kC_pa)))
            
            # calculate pial arterial resistance from equation (11)
            crb.R_pa = crb.k_R*crb.C_pan**2/(((crb.P_pa-crb.P_ic)*crb.C_pa)**2)
            
            # calculate capillary pressure from equation (4)
            crb.P_c = (crb.P_v/crb.R_pv + crb.P_pa/(crb.R_pa/2.0) + 
                    crb.P_ic/crb.R_f)/(1.0/crb.R_pv + 1.0/(crb.R_pa/2.0) + 1.0/crb.R_f)

            
            "------------------------------------intracranial flows and further pressures---------------------------------"
            
            # calculate CSF formation rate from equation (12)
            if crb.P_c>crb.P_ic:
                crb.Q_f = (crb.P_c - crb.P_ic)/crb.R_f 
            else: 
                crb.Q_f = 0.0
                
            # calculate CSF outflow rate from equation (13)
            if crb.P_ic>crb.P_vs:
                crb.Q_0 = (crb.P_ic - crb.P_vs)/crb.R_0
            else: 
                crb.Q_0 = 0.0
            
            # calculate resistance of the terminal intracranial veins R_vs
            if crb.P_v>crb.P_vs:
                crb.R_vs = (crb.P_v - crb.P_vs)/(crb.P_v - crb.P_ic)*crb.R_vs1
            else: 
                crb.R_vs = crb.R_vs1
                
            # calculating the conductance of the terminal intracranial veins
            crb.G_vs = 1.0/crb.R_vs

            # differential equation describing pressure of cerebal veins vi from equation (5)
            crb.dP_vP_icdt = 1.0/crb.C_vi*((crb.P_c - crb.P_v)/crb.R_pv - (crb.P_v - crb.P_vs)/crb.R_vs)

            # NEW
            if carotidOn==1:
                # differential equation pressure of pial arterioles from equation (3)   
                crb.dP_cardt = 1/crb.C_car*( (P[1] - crb.P_car)/crb.R_car - (crb.P_car - crb.P_pa)/(crb.R_la + (crb.R_pa/2)))
                crb.dP_paP_icreduceddt = 1/crb.C_pa*((crb.P_car - crb.P_pa)/(crb.R_la + (crb.R_pa/2)) - (crb.P_pa - crb.P_c)/(crb.R_pa/2) - crb.dC_padt*(crb.P_pa - crb.P_ic))

                crb.Q_car = (P[1] - crb.P_car - Pgrav[21])/(crb.R_car + crb.R_la)
                if crb.Q_car < 0:
                    crb.Q_car = 0

                crb.Q_la = (crb.P_car - crb.P_pa)/(crb.R_la + crb.R_pa/2)
                # calculating flow velocity of MCA
                detla_P_car_ic = crb.P_car - crb.P_ic
                if detla_P_car_ic < 0:
                    detla_P_car_ic = 0.0001
                crb.r_mca = crb.r_mca_n * (np.log((detla_P_car_ic) / (crb.P_a_n - crb.P_ic_n)) / crb.k_mca + 1) # new

            if carotidOn==0:
                # differential equation pressure of pial arterioles from equation (3)   
                crb.dP_paP_icreduceddt = 1/crb.C_pa*((P[1] - crb.P_pa)/(crb.R_la + (crb.R_pa/2)) - (crb.P_pa - crb.P_c)/(crb.R_pa/2)- crb.dC_padt*(crb.P_pa-crb.P_ic))
                crb.Q_la = (P[1] - crb.P_pa - Pgrav[21])/(crb.R_la + crb.R_pa/2)

                # calculating flow velocity of MCA
                detla_P_car_ic = P[1] - crb.P_ic
                if detla_P_car_ic < 0:
                    detla_P_car_ic = 0.0001
                crb.r_mca = crb.r_mca_n * (np.log((detla_P_car_ic) / (crb.P_a_n - crb.P_ic_n)) / crb.k_mca + 1) # new

            if crb.Q_la <0:
                crb.Q_la =0

            crb.Q_pa = (crb.P_pa - crb.P_c)/(crb.R_pa/2)

            # differential equation describing intracranial pressure from equation (15)
            # dP_icreduceddt = dP_icdt - 1/C_ic*(P_paP_ic*C_pa + (P_vP_ic)*C_vi + C_pa*(P_a-P_ic))
            crb.dP_icreduceddt = 1.0/crb.C_ic*(crb.dP_paP_icreduceddt*crb.C_pa + crb.dP_vP_icdt*crb.C_vi + crb.dC_padt*(crb.P_pa-crb.P_ic) + crb.Q_f - crb.Q_0 + crb.hbf) 

            #crb.Q_n = 12.5
            # I've set Q = Q_n here. When Q_n = 12.5, issues are caused in the autoregulation 
            #when going from supine to standing, I suppose it's due to an unrecoverable flow.
            
            # calculating flow velocity of MCA
            crb.v_mca = crb.k_v * crb.Q_la / (3 * np.pi * crb.r_mca**2) # new

            # calculating flow in the venous sinus
            crb.Q_vs = crb.G_vs*(crb.P_v - crb.P_vs)

            if cerebralVeinsOn==0:

                crb.Q_svc = crb.G_svc * (crb.P_vs - P[4]) # new

                # differential equation for venous sinus pressures
                crb.dP_vsdt = 1.0/crb.C_vs*((crb.P_v - crb.P_vs)*crb.G_vs - (crb.P_vs - crb.P_ic)*crb.G_0 - (crb.P_vs - P[4]) * crb.G_svc)


            if cerebralVeinsOn==1:

                # calculating flow in the azygos veins
                crb.Q_azy2 = (crb.P_azy - P[4])*crb.G_azy2 # new, seems to be okay
                
                # calculating flow in the external carotid
                if carotidOn==1:
                    crb.Q_ex = crb.G_ex*(crb.P_car - crb.P_c3) # adjusted to P_c3 rather than P_vs
                else:
                    crb.Q_ex = crb.G_ex*(P[1] - crb.P_c3 - Pgrav[21]) # adjusted to P_c3 rather than P_vs 

                # calculating total flow into intracranial model 
                crb.Q_tot = crb.Q_la + crb.Q_ex # new

                if crb.i_condition == 0:
                    crb.G_jr3 = crb.k_jr3*(1.0+(2.0/np.pi)*np.arctan((crb.P_vs - hydroP*crb.L3 - crb.P_j3ext)/crb.A))**2
                    crb.G_jl3 = crb.k_jl3*(1.0+(2.0/np.pi)*np.arctan((crb.P_vs - hydroP*crb.L3 - crb.P_j3ext)/crb.A))**2
                    crb.G_jr2 = crb.k_jr2*(1.0+(2.0/np.pi)*np.arctan((crb.P_jr3 - hydroP*crb.L2 - crb.P_j2ext)/crb.A))**2
                    crb.G_jl2 = crb.k_jl2*(1.0+(2.0/np.pi)*np.arctan((crb.P_jl3 - hydroP*crb.L2 - crb.P_j2ext)/crb.A))**2
                    crb.G_jr1 = crb.k_jr1*(1.0+(2.0/np.pi)*np.arctan((crb.P_jr2 - hydroP*crb.L1 - crb.P_j1ext)/crb.A))**2
                    crb.G_jl1 = crb.k_jl1*(1.0+(2.0/np.pi)*np.arctan((crb.P_jl2 - hydroP*crb.L1 - crb.P_j1ext)/crb.A))**2
                    
                if crb.i_condition == 1:
                    crb.G_jr3 = 0

                if crb.i_condition>1 and crb.i_condition<3:
                    crb.G_jr2 = 0

                if crb.i_condition>2 and crb.i_condition<4:
                    crb.G_jr1 = 0

                "-------------------------------jugular-vertebral flows---------------------------------------------------------"

                # calculating flow in the collateral segments 
                crb.Q_c3 = crb.G_c3*(crb.P_vs - crb.P_c3)
                crb.Q_c2 = crb.G_c2*(crb.P_c3 - crb.P_c2)
                crb.Q_c1 = crb.G_c1*(crb.P_c2 - P[4]) # Qc1 Flow in the lower segment of the collateral network, does this flow into C[4]?

                # calculating flow in the anastomotic connections
                crb.Q_cjr3 = crb.G_cjr3*(crb.P_c3 - crb.P_jr3)
                crb.Q_cjr2 = crb.G_cjr2*(crb.P_c2 - crb.P_jr2)
                crb.Q_cjl3 = crb.G_cjl3*(crb.P_c3 - crb.P_jl3)
                crb.Q_cjl2 = crb.G_cjl2*(crb.P_c2 - crb.P_jl2)

                # calculating flow in the jugular segments
                crb.Q_jr3 = crb.G_jr3*(crb.P_vs - crb.P_jr3) 
                crb.Q_jl3 = crb.G_jl3*(crb.P_vs - crb.P_jl3) 
                crb.Q_jr2 = crb.G_jr2*(crb.P_jr3 - crb.P_jr2)
                crb.Q_jl2 = crb.G_jl2*(crb.P_jl3 - crb.P_jl2)
                crb.Q_jr1 = crb.G_jr1*(crb.P_jr2 - crb.P_svc1)
                crb.Q_jl1 = crb.G_jl1*(crb.P_jl2 - crb.P_svc1)

                # calculating flow in the vertebral veins
                crb.Q_vv = crb.G_vvr*(crb.P_vs - crb.P_vv) + crb.G_vvl*(crb.P_vs - crb.P_vv)
                crb.Q_vvr = crb.G_vvr*(crb.P_vs - crb.P_vv)
                crb.Q_vvl = crb.G_vvl*(crb.P_vs - crb.P_vv)

                # Calculating flow in the superior vena cava (superior tract)
                #crb.Q_svc1 = crb.G_jr1*(crb.P_jr2 - crb.P_svc1) + crb.G_jl1*(crb.P_jl2 - crb.P_svc1) # why?
                crb.Q_svc1 = crb.Q_jr1 + crb.Q_jl1 # doesn't use P[4]

                # Calculating flow in the renal vein
                crb.Q_rv = crb.G_rv*(crb.P_vv - P[14]) # new, P[4] -> P[14]

                # Calculating outflow in the model
                #crb.Q_svc2 = (crb.P_svc1-P[4])*crb.G_svc2

                "-------------------------------jugular-vertebral circuit equations-----------------------------------------------"
                
                # differential equation for venous sinus pressures
                crb.dP_vsdt = 1.0/crb.C_vs*((crb.P_v - crb.P_vs)*crb.G_vs - (crb.P_vs - crb.P_ic)*crb.G_0 - (crb.P_vs-crb.P_jr3)*crb.G_jr3 -
                                            (crb.P_vs - crb.P_jl3)*crb.G_jl3 - (crb.P_vs - crb.P_c3)*crb.G_c3 - (crb.P_vs - crb.P_vv)*crb.G_vvr - (crb.P_vs - crb.P_vv)*crb.G_vvl)
                    
                # differential equation describing pressure in the 3rd segment of the right jugular vein
                crb.dP_jr3dt = 1.0/crb.C_jr3*((crb.P_vs-crb.P_jr3)*crb.G_jr3 - (crb.P_jr3 - crb.P_c3)*crb.G_cjr3 - (crb.P_jr3-crb.P_jr2)*crb.G_jr2)
                            
                # differential equation describing pressure in the 3rd segment of the left jugular vein
                crb.dP_jl3dt = 1.0/crb.C_jl3*((crb.P_vs-crb.P_jl3)*crb.G_jl3 - (crb.P_jl3 - crb.P_c3)*crb.G_cjl3 - (crb.P_jl3-crb.P_jl2)*crb.G_jl2)    
                            
                # differential equation describing pressure in the 2nd segment of the right jugular vein
                crb.dP_jr2dt = 1.0/crb.C_jr2*((crb.P_jr3 - crb.P_jr2)*crb.G_jr2 - (crb.P_jr2 - crb.P_c2)*crb.G_cjr2 - (crb.P_jr2 - crb.P_svc1)*crb.G_jr1)
                                                
                # differential equation describing pressure in the 2nd segment of the left jugular vein
                crb.dP_jl2dt = 1.0/crb.C_jl2*((crb.P_jl3 - crb.P_jl2)*crb.G_jl2 - (crb.P_jl2 - crb.P_c2)*crb.G_cjl2 - (crb.P_jl2 - crb.P_svc1)*crb.G_jl1)
                            
                # differential equation describing pressure in upper segment of the collateral network
                crb.dP_c3dt = 1.0/crb.C_c3*((crb.P_vs - crb.P_c3)*crb.G_c3 + (crb.P_jr3 - crb.P_c3)*crb.G_cjr3 + 
                                            (crb.P_jl3 - crb.P_c3)*crb.G_cjl3 + (P[1] - crb.P_c3)*crb.G_ex - 
                                            (crb.P_c3-crb.P_c2)*crb.G_c2)
                            
                # differential equation describing pressure in middle segment of the collateral network
                crb.dP_c2dt = 1.0/crb.C_c2*((crb.P_c3 - crb.P_c2)*crb.G_c2 + (crb.P_jr2 - crb.P_c2)*crb.G_cjr2 + (crb.P_jl2 - crb.P_c2)*crb.G_cjl2 - (crb.P_c2-P[4])*crb.G_c1)
                
                # define a new, intermediary pressure compartment to ensure inflow into cerebral model equals out flow
                # we must define a new net outflow that is the difference between the inflow and outflow of all other exiting compartments
                crb.Q_out_net = crb.Q_tot - (crb.Q_c1 + crb.Q_rv)
                        
                # differential equation describing pressure in the superior segment of the superior vena cava
                crb.dP_svc1dt = 1.0/crb.C_svc1*(crb.G_jr1*(crb.P_jr2 - crb.P_svc1) + 
                                                crb.G_jl1*(crb.P_jl2 - crb.P_svc1) 
                                                - (crb.P_svc1-crb.P_svc)*crb.G_svc1) # note I've estimated C_svc1
                            
                # diffentrial equation descrbing pressure in the azygos system
                crb.dP_azydt = 1/crb.C_azy*((crb.P_vv - crb.P_azy)*crb.G_azy1 + (crb.P_rv - crb.P_azy)*crb.G_lv - 
                                            (crb.P_azy - crb.P_svc)*crb.G_azy2)
            
                # testing ?
                crb.dP_svcdt = 1.0/crb.C_svc*(crb.G_svc1*(crb.P_svc1-crb.P_svc) + crb.G_azy2*(crb.P_azy-crb.P_svc) 
                                            - crb.Q_out_net)
                #                                      - crb.G_svc2*(crb.P_svc - P[4])) # ?
            
                crb.Q_svc2 = crb.Q_out_net # Check this!
                
                # differential equation describing pressure in the vertebral vein
                crb.dP_vvdt = 1.0/crb.C_vv*(crb.G_vvr*(crb.P_vs - crb.P_vv) + crb.G_vvl*(crb.P_vs - crb.P_vv)
                                            - crb.G_vv2*(crb.P_vv - crb.P_rv) - crb.G_azy1*(crb.P_vv - crb.P_azy))
            
            """
            # for plotting flows
            if n < startTimeStS/2:
                flows_supine = [crb.Q_tot, crb.Q_ex, crb.Q_jr3 + crb.Q_jl3, crb.Q_jr2 + crb.Q_jl2, crb.Q_jr1 + crb.Q_jl1, crb.Q_vv, crb.Q_c3, crb.Q_svc1, crb.Q_svc2 + crb.Q_c1 + crb.Q_rv]
            
            if n > (startTimeStS+tau_STS):
                flows_upright = [crb.Q_tot, crb.Q_ex, crb.Q_jr3 + crb.Q_jl3, crb.Q_jr2 + crb.Q_jl2, crb.Q_jr1 + crb.Q_jl1, crb.Q_vv, crb.Q_c3, crb.Q_svc1, crb.Q_svc2 + crb.Q_c1 + crb.Q_rv]
            
            #cerebral_flow = [crb.Q_tot, crb.Q_ex, crb.Q_jr3 + crb.Q_jl3, crb.Q_jr2 + crb.Q_jl2, crb.Q_jr1 + crb.Q_jl1, crb.Q_vv, crb.Q_c3, crb.Q_svc1, crb.Q_svc2 + crb.Q_c1 + crb.Q_rv]
            """
            #if t < start_time:
            #    flows_supine = [Q + Q_ex, Q_ex, Q, Q_jr3 + Q_jl3, Q_jr2 + Q_jl2, Q_jr1 + Q_jl1, Q_vv, Q_c3, Q_svc1]
                
            #if t>start_time + tau_1:
            #    flows_upright = [Q+Q_ex, Q_ex, Q, Q_jr3 + Q_jl3, Q_jr2 + Q_jl2, Q_jr1 + Q_jl1, Q_vv, Q_c3, Q_svc1]
                
            # =============================================================================
        

        """ Microcirculation """
        if micro_switch == 0:
            F_micro_lower = 0
            F_micro_upper = 0
            F_lymphatic = 0
        if micro_switch == 1:
            Pmicro=((V_micro/2)-(V_micro2/2))/controlPars.C_micro
            O_int = controlPars.n_int*controlPars.R_gas*controlPars.Temp/V_micro   # Gas-law
            O_cap = controlPars.n_cap*controlPars.R_gas*controlPars.Temp/(sum_v*.6) # times .6 since only 60 percent of the blood is plasma.
            P_cap_lower = P[11] - F[15]*.9*R[0,12] # set the capillary pressure, based on the flow from art to vein and the precap resistance to be 90%
            P_cap_upper = P[2] - F[3]*.9*R[0,3]    # Or alternatively;    Pc = ((.1/.9)*P[11]+P[12])/(1+(.1/.9));
            F_micro_lower = controlPars.Kf * (P_cap_lower-Pmicro-controlPars.sigma*(O_cap-O_int))/controlPars.R_transcap
            F_micro_upper = controlPars.Kf * (P_cap_upper-Pmicro-controlPars.sigma*(O_cap-O_int))/controlPars.R_transcap
            F_lymphatic = (5+Pmicro-P[15])/controlPars.R_lymph; # the 5 is small constant positive pressure term that accounted for the intrinsic pumping action of the initial lymphatics
                #(meaning that lymphatic return is possible even when interstitial tissue pressures were slightly negative).
            if F_lymphatic<0:
                F_lymphatic=0
        
        ### ORIGINAL ###
        #------------------------------------------------------#
        #---------- Left side of the heart --------------------#
        F = np.zeros(24)
        if P[20]>P[0]+Pgrav[0]:
            F[0]=(P[20]-P[0]-Pgrav[0])/R[0,0]; #q0
        else:
            F[0]=0; # no insufficiency
        #-----------Intrathoracic flow rates-------------------#
        F[1]=(P[0]-P[1]-Pgrav[1])/R[0,1]
        
        if F[1]<0:
            F[1]=0
        F[2]=(P[1]-P[2]-Pgrav[2])/R[0,2]
        if F[2]<0:
            F[2]=0
        F[3]=(P[2]-P[3])/R[0,3]; # No influence of gravity for this flow.
        if F[3]<0:
            F[3]=0
        # Starling resistor defines the flow into the superior vena cava.
        
        if ((P[3]+Pgrav[3] > P[4]) and (P[4] > P_intra)):
            F[4] = (P[3] - P[4] + Pgrav[3]) / R[0,4]
        elif ((P[3]+Pgrav[3] > P_intra) and (P_intra > P[4])):
            F[4] = (P[3] - P_intra + Pgrav[3]) / R[0,4]
        elif (P_intra > P[3]+Pgrav[3]):
            F[4] = 0.0
        if P[4]-P[15]>0:
            F[5]=(P[4]-P[15]+Pgrav[4])/R[3,4]; #q5
        else:
            F[5]=(P[4]-P[15]+Pgrav[4])/(10*R[3,4]); #q5
        F[6]=(P[0]-P[5]+Pgrav[5])/R[0,5]
        if F[6]<0:
            F[6]=0
        F[7]=(P[5]-P[6]+Pgrav[6])/R[0,6]
        if F[7]<0:
            F[7]=0
        F[8]=(P[6]-P[7]+Pgrav[7])/R[0,7]
        F[9]=(P[7]-P[8])/R[0,8]; # No influence of gravity for this flow.
        F[10]=(P[8]-P[13]-Pgrav[8])/R[3,8]
        F[11]=(P[6]-P[9]+Pgrav[9])/R[0,9]
        F[12]=(P[9]-P[10])/R[0,10]; # No influence of gravity for this flow.
        if P[10]>P[13]+Pgrav[10]:
            F[13]=(P[10]-P[13]-Pgrav[10])/R[3,10]
        else: 
            F[13]=0
        F[14]=(P[6]-P[11]+Pgrav[11])/R[0,11]
        F[15]=(P[11]-P[12])/R[0,12]; # No influence of gravity for this flow.
        if P[12]>(P[13]+Pgrav[12]):
            F[16]=(P[12]-P[13]-Pgrav[12])/R[3,12]
        else:
            F[16]=0
        if P[13]>(P[14]+Pgrav[13]):
            F[17]=(P[13]-P[14]-Pgrav[13])/R[0,14]
        else:
            F[17]=0

        if (P[14]-P[15])>0:           # Blood flow comes from both the IVC and SVC
            F[18]=(P[14]-P[15]-Pgrav[14])/R[3,14]
        else:
            F[18]=(P[14]-P[15]-Pgrav[14])/(10*R[3,14])
        if (P[15]-P[16])>0:
            F[19]=(P[15]-P[16])/R[0,16]; #q19
        else:
            F[19]=0
        if P[16]-P[17]>0:
            F[20]=(P[16]-P[17])/R[3,16]; #20
        else:
            F[20]=0
        F[21]=(P[17]-P[18])/R[0,18]; # No influence of gravity for this flow.
        if P[18]-P[19]>0:
            F[22]=(P[18]-P[19])/R[0,19]; #q22
        else:
            F[22]=(P[18]-P[19])/(10*R[0,19]); # this allows for some insufficiency         
        if P[19]-P[20]>0:
            F[23]=(P[19]-P[20])/R[0,20]; #q23
        else:
            F[23]=0; # no insufficiency

        # Calculate the derivatives of volumes for cardiovascular model
        dVdt = np.zeros(22)
        dVdt[0] = F[0]-F[1]-F[6]

        dVdt[2] = F[2]-F[3]
        dVdt[3] = F[3]-F[4] - F_micro_upper

        dVdt[5] = F[6]-F[7]
        dVdt[6] = F[7]-F[8]-F[11]-F[14]
        dVdt[7] = F[8]-F[9]
        dVdt[8] = F[9]-F[10]
        dVdt[9] = F[11]-F[12]
        dVdt[10] = F[12]-F[13]
        dVdt[11] = F[14]-F[15]
        dVdt[12] = F[15]-F[16] - F_micro_lower
        dVdt[13] = F[10]+F[13]+F[16]-F[17]

        dVdt[15] = F[5]+F[18]-F[19]
        dVdt[16] = F[19]-F[20]
        dVdt[17] = F[20]-F[21]
        dVdt[18] = F[21]-F[22]
        dVdt[19] = F[22]-F[23]
        dVdt[20] = F[23]-F[0]
        dVdt[-1] = F_micro_lower + F_micro_upper - F_lymphatic

        if cerebralModelOn==1:
            if carotidOn==1:
                dVdt[1] = F[1]-F[2] - crb.Q_car
            if carotidOn==0 and cerebralVeinsOn==0: # if carotid is OFF
                dVdt[1] = F[1]-F[2] - crb.Q_la # Blood is now also flowing into the brain model of Patrick.
            if carotidOn==0 and cerebralVeinsOn==1:
                dVdt[1] = F[1]-F[2] - crb.Q_la - crb.Q_ex
            if cerebralVeinsOn==1:
                dVdt[4] = F[4]-F[5] + F_lymphatic + crb.Q_svc2 # Renal vein shouldn't enter compartment 4, also svc2 is the lower 
                dVdt[14] = F[17]-F[18] + crb.Q_rv # added Q_svc2
            if cerebralVeinsOn==0:
                dVdt[4] = F[4]-F[5] + F_lymphatic + crb.Q_svc # Renal vein shouldn't enter compartment 4, also svc2 is the lower 
                dVdt[14] = F[17]-F[18]
        if cerebralModelOn==0:
            dVdt[1] = F[1]-F[2]
            dVdt[4] = F[4]-F[5] + F_lymphatic
            dVdt[14] = F[17]-F[18]


        # Store values in the global arrays
        idx = np.searchsorted(t_eval, t)
        if 0 <= idx < len(t_eval):
            store_P[:, idx] = P
            #F_store[:, idx] = F
            store_HR[idx] = 1/HP*60
            store_P_intra[idx] = P_intra
            store_P_muscle2[:, idx] = [P_muscle_legs, P_muscle_abdom, P_muscle_thorax, P_intra]
            store_E[:, idx] = [E[15], E[16], E[19], E[20]]
            store_UV[:, idx] = UV
            if cerebralModelOn==1:
                store_crb[idx] = crb.Q_la
                """ store_crb_Q_ic:
                Q_f:    Cerebrospinal fluid formation rate
                Q_0:    Cerebrospinal fluid outflow rate
                Q:      Cerebral blood flow
                Q_vs:   Venous sinus flow
                Q_auto: average cerebral flow over crb_buffer_size [s]: np.mean(crb.Qbrain_buffer)
                Q_la:   large cerebral artery flow
                Q_pa:   pial arteriole flow
                Q_car:  carotid artery flow
                """
                store_crb_Q_ic[:, idx] = crb.Q_f, crb.Q_0, crb.Q_la, crb.Q_vs, crb.Q_auto, crb.Q_pa, crb.Q_car # intracranial flows
                store_crb_C[:, idx] = crb.C_vi, crb.C_ic, crb.C_pa # intracranial capacities
                store_crb_R[:, idx] = crb.R_pa, crb.R_vs # intracranial resistances
                store_crb_x[idx] = crb.x_aut # State variable of the autoregulation mechanism related to cerebral flow variations
                store_crb_mca[:, idx] = crb.v_mca, crb.r_mca # Flow velocity in the middle cerebral artery

                if cerebralVeinsOn==1:
                    store_crb_Q_j[:, idx] = crb.Q_c3, crb.Q_c2, crb.Q_c1, crb.Q_jr3, crb.Q_jr2, crb.Q_jr1, crb.Q_jl3, crb.Q_jl2, crb.Q_jl1 # jugular flows
                    store_crb_Q_v[:, idx] = crb.Q_vvr, crb.Q_vvl, crb.Q_vv # vertebral flows
                    store_crb_Q_out[:, idx] = crb.Q_lv, crb.Q_svc2, crb.Q_svc1, crb.Q_rv, crb.Q_azy2, crb.Q_out_net, crb.Q_tot # azygos, svc and out flows
                    store_crb_P[:, idx] = crb.P_v, crb.P_pa # Pressure in the cerebral veins and pial arterioles
                    store_crb_G[:, idx] = crb.G_jr3, crb.G_jr2, crb.G_jr1, crb.G_jl3, crb.G_jl2, crb.G_jl1, crb.G_c1, crb.G_cjr3, crb.G_cjr2, crb.G_cjl3, crb.G_cjl2, crb.G_c3 # jugular conductance

            # Autoregulation step I -> Get a constant array of the aortic arch and rap pressure of the most recent 4*S_GRAN seconds
            if idx_check == idx: # check if the same value is changed
                abp_buffer[-1] = store_P[0, idx] #Note, the baroreflex does not incorparate the baroreceptors in the carotid arteries 
                rap_buffer[-1] = store_P[15, idx]
                if cerebralModelOn==1:
                    crb.Qbrain_buffer[-1] = store_crb[idx] # Cerebral blood flow buffer
            else: # otherwise, move the filter
                abp_buffer = np.append(abp_buffer[1:], store_P[0, idx]) #Note, the baroreflex does not incorparate the baroreceptors in the carotid arteries 
                rap_buffer = np.append(rap_buffer[1:], store_P[15, idx])
                if cerebralModelOn==1:
                    crb.Qbrain_buffer = np.append(crb.Qbrain_buffer[1:], store_crb[idx]) # Cerebral blood flow buffer
            idx_check = idx
        store_t.append(t); # Check all calculated timesteps
        
        if cerebralModelOn==1:
            crb.Q_auto = np.mean(crb.Qbrain_buffer)
            crb.dx_autdt = (1.0/crb.tau_aut)*(-crb.x_aut + crb.G_aut*(crb.Q_auto-crb.Q_n)/crb.Q_n) 
            if carotidOn==0 and cerebralVeinsOn==1:
                crb_ddt = [crb.dx_autdt, crb.dP_vP_icdt, crb.dP_paP_icreduceddt, crb.dP_icreduceddt, crb.dP_vsdt, crb.dP_jr3dt, crb.dP_jl3dt, crb.dP_jr2dt,
                crb.dP_jl2dt, crb.dP_c3dt, crb.dP_c2dt, crb.dP_svc1dt, crb.dP_azydt, crb.dP_svcdt, crb.dP_vvdt]
            if carotidOn==1 and cerebralVeinsOn==1:
                crb_ddt = [crb.dx_autdt, crb.dP_vP_icdt, crb.dP_paP_icreduceddt, crb.dP_icreduceddt, crb.dP_vsdt, crb.dP_cardt, crb.dP_jr3dt, crb.dP_jl3dt, crb.dP_jr2dt,
                crb.dP_jl2dt, crb.dP_c3dt, crb.dP_c2dt, crb.dP_svc1dt, crb.dP_azydt, crb.dP_svcdt, crb.dP_vvdt]
            if carotidOn==0 and cerebralVeinsOn==0:
                crb_ddt = [crb.dx_autdt, crb.dP_vP_icdt, crb.dP_paP_icreduceddt, crb.dP_icreduceddt, crb.dP_vsdt]
            if carotidOn==1 and cerebralVeinsOn==0:
                crb_ddt = [crb.dx_autdt, crb.dP_vP_icdt, crb.dP_paP_icreduceddt, crb.dP_icreduceddt, crb.dP_vsdt, crb.dP_cardt]
            return np.concatenate([dVdt, crb_ddt])
        else:
            return dVdt

    
    if cerebralModelOn==0:
        y0 = np.zeros(22)
    if cerebralModelOn==1:
        if cerebralVeinsOn==1:
            y0 = np.zeros(37+carotidOn)
            y0[27+carotidOn] = crb.P_jr3 # 3rd segment of the right jugular vein pressure
            y0[28+carotidOn] = crb.P_jl3 # 3rd segment of the left jugular vein pressure
            y0[29+carotidOn] = crb.P_jr2 # 2nd segment of the right jugular vein pressure
            y0[30+carotidOn] = crb.P_jl2 # 2nd segment of the left jugular vein pressure
            y0[31+carotidOn] = crb.P_c3 # pressure in upper segment of the collateral network
            y0[32+carotidOn] = crb.P_c2 # pressure in middle segment of the collateral network
            y0[33+carotidOn] = crb.P_svc1 # pressure in the superior segment of the superior vena cava
            y0[34+carotidOn] = crb.P_azy # pressure in the azygos system
            y0[35+carotidOn] = crb.P_svc # testing?
            y0[36+carotidOn] = crb.P_vv # pressure in the vertebral vein
        else:
            y0 = np.zeros(27+carotidOn)
        
        y0[22] = crb.x_aut # autoregulation
        y0[23] = crb.P_vP_ic # pressure of cerebal veins vi from equation (5)
        y0[24] = crb.P_paP_ic # pressure of pial arterioles from equation (3)
        y0[25] = crb.P_ic # intracranial pressure from equation (15)
        y0[26] = crb.P_vs # cerebral sinus veins pressure
        if carotidOn==1:
            y0[27] = crb.P_car # pressure in the middle cerebral artery
    
    y0[0:22] = V # volumes
        
    t_span = (tmin, tmax)
    esse = esse_cerebral_NEW
    #solution = solve_ivp(esse_test, t_span, y0, t_eval=t_eval, method='RK45')
    # Create an instance of the tracker
    #sol = solve_ivp(esse, t_span, y0, first_step=T*0.1, max_step=T, t_eval=t_eval, method='RK23', rtol=1e-4, atol=1e-4)
    sol = solve_ivp(esse, t_span, y0, first_step=T*0.1, max_step=T, t_eval=t_eval, method='RK23', rtol=1e-4, atol=1e-4)

    def find_extremes_of_cardio_cycles(pressures, pressure_times, onset_times):
        """
        Find the maximum and minimum pressures of each cardiac cycle for a given compartment.
        
        :param pressures: np.ndarray of shape (t, 21) with pressures
        :param pressure_times: np.ndarray of shape (t,) with times corresponding to pressure measurements
        :param onset_times: list or np.ndarray of onset times (in seconds) for each cardiac cycle
        :param compartment_index: int, the index of the compartment to analyze (default is 2 for compartment 3)
        :return: list of tuples, each containing (max_pressure, min_pressure) for each cycle
        """
        # Ensure onset_times are integers
        onset_indices = np.searchsorted(pressure_times, onset_times)
        
        num_cycles = len(onset_times)-2
        extremes = []
        
        for i in range(num_cycles):
            # Determine the start and end of the current cycle
            start_idx = onset_indices[i]
            end_idx = onset_indices[i + 1] if i + 1 < num_cycles else len(pressures)
            
            if start_idx >= end_idx:
                # Skip invalid cycles where the start index is not less than the end index
                print(f"Skipping invalid cycle from {start_idx} to {end_idx}")
                continue
            
            # Extract the pressures for the current cycle and the given compartment
            cycle_pressures = pressures[start_idx:end_idx]
            
            if cycle_pressures.size == 0:
                # Skip empty slices
                print(f"Skipping empty cycle from {start_idx} to {end_idx}")
                continue
            
            # Find the maximum and minimum pressures for the current cycle
            max_pressure = np.max(cycle_pressures)
            min_pressure = np.min(cycle_pressures)
            mean_t = np.mean(pressure_times[onset_indices[i:i+2]])
            # Store the result
            extremes.append((mean_t, max_pressure, min_pressure))
        
        return np.array(extremes)

    # Find the extremes for compartment 3 (index 2)
    extremes_BP = find_extremes_of_cardio_cycles(store_P[2], t_eval, t_cc_onset)
    t_mean = extremes_BP[:,0]
    store_BP_max = extremes_BP[:,1]
    store_BP_min = extremes_BP[:,2]
    store_finap = store_BP_max*1/3+2/3*store_BP_min
    HR_list = HR_list[:-2]
    if cerebralModelOn==1:
        global tmean_mca, store_V_mca_max, store_V_mca_min
        extremes_V_mca = find_extremes_of_cardio_cycles(store_crb_mca[0], t_eval, t_cc_onset)
        tmean_mca = extremes_V_mca[:,0]
        store_V_mca_max = extremes_V_mca[:,1]
        store_V_mca_min = extremes_V_mca[:,2]
        Out_av.append([t_mean, map, store_finap, store_HR, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra,
        store_P_muscle, tmean_mca, store_V_mca_max, store_V_mca_min, store_P_muscle2])
    else:
        Out_av.append([t_mean, map, store_finap, store_HR, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra,
        store_P_muscle, store_P_muscle2])
    Timer.stop()

    outputP = output9 = outputP_intra = output10 = outputPg = outputE = []
    Out_wave.append([store_t, outputP, output9, outputP_intra, output10, outputPg, outputE])
    Out_solver = [sol.t, sol.y]
    return Out_av, Out_wave, Out_solver


### MORE DEF ###
def find_index_of_time(time_array, target_time, tol=1e-5):
    indices = np.where(np.isclose(time_array, target_time, atol=tol))[0]  # Get indices where target_time occurs within the tolerance
    if indices.size > 0:  # Check if indices array is not empty
        return indices[0]  # Return the first index where the target_time occurs
    else:
        print(f"Time value {target_time} not found in the array.")
        return None

def lin_interp(t, values, desired_interval):
    # Create new time array with the desired frequency, starting from the first whole number in the range
    start_time = np.ceil(t[0] * 10) / 10  # Adjusting to start from the next whole 0.1 increment
    end_time = np.floor((t[-1] - desired_interval) * 10) / 10  # Adjusting to stay within bounds
    new_times = np.arange(start_time, end_time + desired_interval, desired_interval)

    # Perform linear interpolation
    linear_interp = interp1d(t, values, kind='linear')
    new_values = linear_interp(new_times)
    return new_times, new_values

# Function used by the optimization script
def solve2(inputs):
    """
    This function solves the optimization problem using the inputs provided.
    
    Args:
        inputs (list): A list of input parameters to be mapped to the model's input vector.
    
    Returns:
        The first three elements of the result from the 'solve' function.
    """
    ### INPUTS ###
    global sliding_window_size, baro_buffer, crb_buffer_size
    
    baro_buffer = 4 # int
    crb_buffer_size = 2 # seconds, ORIGINAL
    #crb_buffer_size = 0.1 # seconds

    scaling = {
        "RRsgain": inputs[0], # RRsgain gain 1 (HP), maybe combine scaling with RRsgain scaling, vec[2] -> doesn't do anything -> now it does
        "RRpgain": inputs[1], # RRpgain gain 2 (HP), maybe combine scaling with RRpgain scaling, vec[3] -> doesn't do anything -> now it does
        "beta": inputs[2], # beta (UV)
        "alpha": inputs[3], # alpha (UV)
        "beta_resp_Erv": inputs[4], # beta_resp (ErvMAX)
        "beta_resp_Elv": inputs[5], # beta_resp (ElvMAX)
        "alpha_resp": inputs[6], # alpha_resp (R)
        "alphav_resp": inputs[7], # alphav_resp (R)
        "RAP_setp": inputs[8], # RAP_setp -> some influence
        "ABP_setp": inputs[9], # ABP_setp -> is first set though data averaging, then modified by the optimizer
        "max_RV_E": inputs[10], # max right ventricle elastance
        "min_RV_E": inputs[11], # min right ventricle elastance
        "max_LV_E": inputs[22], # max left ventricle elastance
        "min_LV_E": inputs[23], # min left ventricle elastance
        "SD_ratio": inputs[12], # systole-diastole ratio scaling
        "HR_t0": inputs[13], # HR @ t=0
        "STS_pressure": inputs[14], # STS pressure
        "Stand_muscleP": inputs[15], # muscle P after STS (artificial value)
        "Global_R": inputs[16], # Global R
        "Global_UV": inputs[17], # Global UV
        "Rp": inputs[18]*1.4, # Rp. age regression 1.1%/year = 1.0111^(70-40) = 1.4, LANDOWNE 1955
        "E_arteries": inputs[19]*1.6, # E arteries. age regression 1.58%/year = 1.0158^(70-40) = 1.6, LANDOWNE 1955
        "E_veins": inputs[20]*1.6, # E veins. age regression 1.58%/year = 1.0158^(70-40) = 1.6, LANDOWNE 1955
        "v_ratio": inputs[21], # v_ratio scaling
        "baro_delay2": 0, # baro_delay
        "baro_delay_para": inputs[24], # parasympathetic baro_delay
        "baro_delay_sympa": inputs[25], # sympathetic baro_delay
        "baro_delay_BR_R": inputs[26], # baro_delay BR R
        "baro_delay_BR_UV": inputs[27], # baro_delay BR UV
        "baro_delay_CP_R": inputs[28], # baro_delay CP_R
        "baro_delay_CP_UV": inputs[29], # baro_delay CP_UV
        # Cerebral model
        "k_v": 1, # Cerebral model: doppler correction factor, correct doppler angle and converts maximal to average velocity
        "G_aut": 1 # Cerebral model: Gain of the autoregulation mechanism related to CBF variations
        }
    
    print(scaling)
    
    ### RUN ###
    print("Input parameters are:\n", inputs)
    return solve(scaling)[:3]

# Function to do just 1 test-run with a given set of scalars
def run_solve2():
    inp_ones = np.ones(27)
    inp_ones[-1] = 0.0

    inp_opti = [ 0.88521808, 0.57638988, 1.23869191, 0.63037545, 1.04719004, 1.52528282, 1.38392606,
    1.20657392, 0.81749262, 1.04000896, 0.77240868, 1.38909163, 1.1387788, 1.19498142, 0.36249268,
    3.10236141, 1.00832986, 0.81639036, 1.08779616, 1.0769972, 1.3786385, 1.03288004, 1.14197793,
    1.36268182, 0, 0, 0, 0, 0, 0] 
    """
    inp_opti = np.array([0.5826683082261311, 0.5638604531557075, 0.5318674159872925, 0.6304879791429777,
    0.7242952295412761, 1.0467043690477267, 0.7604962987638669, 1.4466729673118304,
    1.3502707053869294, 0.9701380402162948, 1.084693169545678, 1.3695976134802526,
    0.8357467676121084, 0.9164333830338648, 1.104359123141875, 0.9076092600865857,
    1.1112503540760827, 0.8609885864116209,
    0.7109305146197087, 0.8766995852060988, 0.9434873025258455, 1.0594999868228965,
    0.8974756227694911, 1.1424840201763422, 2.0])
    
    inp_opti = np.array([ 0.99712327,  0.74658897,  1.31748387,  0.59933618,  1.10942531,
        1.80500194,  1.31122843,  0.74436624,  1.30534411,  0.82168507,
        1.28433895,  1.42161834,  0.84986681,  1.18873776,  1.51981114,
        2.41849041,  1.05983693,  0.91024237,  1.18705277,  1.07158593,
        0.91086076,  1.12141536,  1.14041107,  1.22997517, -0.25175373,
        1, 1])
    """
    Out_av, Out_wave, Out_solver = solve2(inp_opti)
    global t_mean
    if cerebralModelOn==1:
        t_mean, MAP, Finap, HR_model, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra, store_P_muscle, tmean_mca, store_V_mca_max, store_V_mca_min, store_P_muscle2 = Out_av[0][0], Out_av[0][1], Out_av[0][2], Out_av[0][3], Out_av[0][4], Out_av[0][5], Out_av[0][6], Out_av[0][7], Out_av[0][8], Out_av[0][9], Out_av[0][10], Out_av[0][11], Out_av[0][12], Out_av[0][13]
    else:
        t_mean, MAP, Finap, HR_model, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra, store_P_muscle, store_P_muscle2 = Out_av[0][0], Out_av[0][1], Out_av[0][2], Out_av[0][3], Out_av[0][4], Out_av[0][5], Out_av[0][6], Out_av[0][7], Out_av[0][8], Out_av[0][9], Out_av[0][10]
    t, wave_f, alpha_tilt, p_intra, p_muscle, p_grav, Elas = Out_wave[0][0], Out_wave[0][1], Out_wave[0][2], Out_wave[0][3], Out_wave[0][4], Out_wave[0][5], Out_wave[0][6]
    t = Out_wave[0][0]
    global t_solver, y_solver
    t_solver, y_solver = Out_solver
        
    return