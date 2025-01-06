'''
Lex M. van Loon (lexmaxim.vanloon@anu.edu.au)
College of Health and Medicine
Australian National University (ANU)

MIT License, Copyright (c) 2024 Lex M. van Loon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Schematic overview of the 21-comaprtment model with:
---
|x| -> the compartments, and -x- -> the flows
---
                    ---         ---
----4--------------|3|----3----|2|--------------2-----
|                  ---         ---                   |
---                #0 Ascending Aorta                ---
|4|                #1 Upper thoracic artery          |1|
---                #2 Upper body arteries            ---
|                 #3 Upper body veins                |
5                 #4 Super vena cava                 1
|                 #5 Thoracic aorta                  |
| ----    ----    ----    ----    ----    ----   --- |
|-|15|-19-|16|-20-|17|-21-|18|-22-|19|-23-|20|-0-|0|-|
| ----    ----    ----    ----    ----    ----   --- |
|                 #6 Abdominal aorta                 |
18                 #7 Renal arteries                  6
|                 #8 Renal veins                     |
---                #9 Splanchnic arteries            ---
|14|               #10 Splanchnic veins              |5|
---                #11 Lower body arteries           ---
|                 #12 Lower body veins               |
17                 #13 Abdominal veins                7
|                 #14 Inferioir vena cava            |
---                #15 Right atrium                  ---
|13|               #16 Right ventricle               |6|
---                #17 Pulmonoary arteries           ---
|                 #18 Pulmonary veins                |
|                 #19 Left atrium                    |
|                 #20 left ventricle                 |
|                   ---         ---                  |
----10--------------|8|----9----|7|-------------8-----
|                   ---         ---                  |
|                   ---         ---                  |
----13--------------|10|---12---|9|------------11-----
|                   ---         ---                  |
|                   ---         ---                  |
----16--------------|12|---15---|11|-----------14-----
                    ---         ---

This file runs the model
-----------------------------------------------------------------

version 1.0 - initial


-----------------------------------------------------------------'''

def solve(scaling, solve_config):

    # CVSIM - IMPORT 
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
    import pandas as pd
    import csv
    import platform
    from CVSIM_plot import CVSIM_plot as cvplt
    import CVSIM_utils as utils
    from adultPars_carotid import _init_pars # Get the parameters for resistance, elastance and uvolume
    from controlPars import _init_control # Get the control parameters loaded.
    from reflexPars import _init_reflex # Get the control parameters loaded.

    Timer = timerz.Timer()
    Timer.start()

    global subjectPars, controlPars, reflexPars
    reflexPars = _init_reflex(scaling); # Get all the reflex parameters stored to the list 'reflexPars'.
    subjectPars = _init_pars(scaling); # Here the compartments parameters are assigned
    controlPars = _init_control(); # Here the control parameters are assigned
    
    global delta_t, HP, tmax, T, resting_time, limit_R, limit_UV, HR, abp_hist,pp_hist, rap_hist,para_resp,beta_resp,alpha_resp,alpha_respv,alphav_resp,alphav_respv
    global s_abp, abp_ma,rap_ma,s_rap,abp_pp,pp_abp, ErvMAX, ElvMAX, V_micro, t_eval, store_P, store_HR, sts_n_end, strainTime
    t_span = solve_config.get("t_span", [0, 250])
    T = solve_config.get("T", 0.01)
    t_eval = solve_config.get("t_eval", [])
    weight = solve_config.get("weight", [])
    length = solve_config.get("length", [])
    alpha_0 = solve_config.get("alpha_0", 0)
    planet = solve_config.get("planet", 1)
    StStest = solve_config.get("StStest", 0)
    TimeToStand = solve_config.get("TimeToStand", 5)
    startTimeStS = solve_config.get("startTimeStS", 60)
    strainTime = solve_config.get("strainTime", 0)
    micro_switch = solve_config.get("micro_switch", 0)
    cerebralVeinsOn = solve_config.get("cerebralVeinsOn", 0)
    carotidOn = solve_config.get("carotidOn", 0)
    ABRreflexOn = solve_config.get("ABRreflexOn", 0)
    CPRreflexOn = solve_config.get("CPRreflexOn", 0)
    fluidLoading = solve_config.get("fluidLoading", 0)
    ABP_setp = solve_config.get("ABP_setp", [])

    # Get all the pars from the different parameters values
    """
    sit_index = utils.find_index_of_time(data_time, sts0)
    stand_index = utils.find_index_of_time(data_time, sts1)
    #reflexPars.ABP_setp = sum(data_BP[sit_index:stand_index])/(stand_index-sit_index)*scaling["ABP_setp"]

    sit_index_10Hz = utils.find_index_of_time(data_time_10Hz, round(sts0, 1))
    stand_index_10Hz = utils.find_index_of_time(data_time_10Hz, round(sts1, 1))
    reflexPars.PP_setp = sum(data_SBP[sit_index_10Hz:stand_index_10Hz]-data_DBP[sit_index_10Hz:stand_index_10Hz])/(stand_index_10Hz-sit_index_10Hz)
    """
    reflexPars.ABP_setp = ABP_setp*scaling["ABP_setp"]
    #print(reflexPars.ABP_setp)
    
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
    limit_R = 100 * scaling["limit_R"] # What is an appropriate limit?  -> scales 1/limit^4 with resistance
    limit_UV = 100 * scaling["limit_UV"] # What is an appropriate limit?  -> scales ??
    
    #controlPars.BW = 77
    controlPars.BW = weight
    BW = controlPars.BW
    controlPars.C_micro=BW*2.9; # ref. ?
    #controlPars.H = 1.76
    controlPars.H = length/100
    H = controlPars.H
    controlPars.tbv = 70/(np.sqrt((BW/(22*H**2)))) * BW # Lemmens-Bernstein-Brodsky Equation
    TBV=controlPars.tbv*scaling["v_ratio"]
    
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
    Vmax10=1500 * scaling["Vmax"]
    Vmax12=1000 * scaling["Vmax"]
    Vmax13=150 * scaling["Vmax"]

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
    
    global HP_min, HP_max, R3, R8, R10, R12, UV3, UV8, UV10, UV12, ErvNorm, ElvNorm, ElaMIN, ElaMAX, ElvMIN, ErvMAX, ElvMAX, EraMIN, EraMAX, ErvMIN, ErvMAX_0, ElvMAX_0
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
    ElvMAX_0=subjectPars.elastance[0,20].copy()

    EraMIN=subjectPars.elastance[1,15].copy()
    EraMAX=subjectPars.elastance[0,15].copy()
    ErvMIN=subjectPars.elastance[1,16].copy()
    ErvMAX=subjectPars.elastance[0,16].copy()
    ErvMAX_0=subjectPars.elastance[0,16].copy()

    HP_min = 1.5 # minimal heart period
    HP_max = 0.3 # maximal heart period
    """
    ### IMPULSE FACTORS ###
    # More setpoints
    reflexPars.RAP_setp = 3 * scaling["RAP_setp"] # vec[15] -> some influence
    reflexPars.ABRsc = 18 # vec[1] -> little influence
    reflexPars.RAPsc = 5  # vec[16] -> very little influence

    # Gains
    reflexPars.RRsgain = reflexPars.RRsgain * scaling["RRsgain"] # vec[2] = 0.012
    reflexPars.RRpgain = reflexPars.RRpgain * scaling["RRpgain"] # vec[3] = 0.005
    reflexPars.beta = reflexPars.beta * scaling["beta"] # = 1
    reflexPars.alpha = reflexPars.alpha * scaling["alpha"] # = 1
    gain1 = 0.021 * scaling["beta_resp_Erv"] 
    gain2 = 0.014 * scaling["beta_resp_Elv"]
    gain3 = -.13 * scaling["alpha_resp"]
    gain4 = -.3 * scaling["alphav_resp"]
    """
    # More setpoints
    reflexPars.RAP_setp = 6 * scaling["RAP_setp"] # vec[15] -> some influence
    reflexPars.ABRsc = 18 # vec[1] -> little influence
    reflexPars.RAPsc = 5  # vec[16] -> very little influence

    reflexPars.RRsgain = reflexPars.RRsgain * scaling["RRsgain"] # vec[2] = 0.012
    reflexPars.RRpgain = reflexPars.RRpgain * scaling["RRpgain"] # vec[3] = 0.005
    reflexPars.beta = reflexPars.beta * scaling["beta"] # = 1, UV venous snsors
    reflexPars.alpha = reflexPars.alpha * scaling["alpha"] # = 1, UV arterial sensors
    gain1 = 0.022 * scaling["beta_resp_Erv"] # Elastance right ventricle
    gain2 = 0.007 * scaling["beta_resp_Elv"] # Elastance left ventricle
    gain3 = -0.05 * scaling["alpha_resp"] # Resistance arterial sensors
    gain4 = 0.05 * scaling["alphav_resp"] # Resistance venous sensors

    global baro_buffer, crb_buffer_size
    baro_buffer = 4 # int
    crb_buffer_size = 2 # seconds, ORIGINAL
    #crb_buffer_size = 0.1 # seconds
    
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
    global store_crb_Q_ic, store_crb_Q_j, store_crb_Q_v, store_crb_Q_out, store_crb_P, store_crb_C, store_crb_R, store_crb_G, store_crb_x, store_crb_mca, store_impulse, store_TBV
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
    store_TBV = np.zeros(len(t_eval))
    store_impulse = np.zeros((6, len(t_eval)))
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

    global abp_buffer, rap_buffer, abp_ma, abp_pp, rap_ma, abp_hist, rap_hist, abp_hist, rap_hist, pp_hist
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
    
    def impulseFunction(abp_buff,rap_buff):
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
        abp_ma = np.append(abp_ma[1:], np.mean(abp_buff))
        #abp_pp = np.append(abp_pp[1:],np.max(abp_buff)-np.min(abp_buff)) # THIS IS WRONG, the PP cannot be determined in 0.25 seconds
        rap_ma = np.append(rap_ma[1:], np.mean(rap_buff))
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
            HP=(60/HR) + beta_resp*reflexPars.RRsgain + para_resp*reflexPars.RRpgain
            # Check if HP is NaN and set to HP_max if it is
            if math.isnan(HP):
                HP = HP_max
            # Set some limits
            if HP < HP_max: # the heart rate can't be higher then 300
                HP = HP_max
            if HP > HP_min: # the heart rate cannot be lower then 40
                HP = HP_min
        # Contractility feedback. Limit contractility feedback so end-systolic ventricular elastances do not become too large during severe stress.
        ErvMAX=1/(1/ErvMAX_0+beta_resp*gain1)
        if 1/ErvMAX < .01:
            ErvMAX=1/.01
        ElvMAX=1/(1/ElvMAX_0+beta_resp*gain2)
        if 1/ElvMAX<.05:
            ElvMAX=1/.05
        return ErvMAX, ElvMAX, HP
    
    def CPRreflexDef():
        # Upper body compartment
        R[0,3] = R3 + gain3*alpha_resp + gain4*alphav_resp #  vec[4]=-.13  vec[17]=-0.3
        # R kidney compartment
        R[0,8] = R8 + gain3*alpha_resp + gain4*alphav_resp #  vec[5]=-.13  vec[18]=-.3
        # R splanchnic compartment
        R[0,10] = R10 + gain3*alpha_resp + gain4*alphav_resp #  vec[6]=-.13  vec[19]=-.3
        # R lower body compartment
        R[0,12] = R12 + gain3*alpha_resp + gain4*alphav_resp #  vec[7]=-0.13  vec[20]=-.3
        
        # Limit R, what is an appropriate limit?
        R[0,3] = utils.lim(R[0,3], R3, limit_R)
        R[0,8] = utils.lim(R[0,8], R8, limit_R)
        R[0,10] = utils.lim(R[0,10], R10, limit_R)
        R[0,12] = utils.lim(R[0,12], R12, limit_R)
        
        # Venous tone feedback implementation.
        # Upper body veins compartment 
        UV[3] = UV3 + alpha_respv*5.3*reflexPars.alpha + alphav_respv*13.5*reflexPars.beta #vec[8]=5.3, vec[21]=13.5
        # Renal veins
        UV[8] = UV8 + alpha_respv*1.3*reflexPars.alpha + alphav_respv*2.7*reflexPars.beta #vec[9]=1.3, vec[22]=2.7
        # Splanchnic veins
        UV[10] = UV10 + alpha_respv*13.3*reflexPars.alpha + alphav_respv*64*reflexPars.beta #vec[10]=13.3, vec[23]=64
        # Lower body veins
        UV[12] = UV12 + alpha_respv*6.7*reflexPars.alpha + alphav_respv*30*reflexPars.beta #vec[11]=6.7, vec[24]=30
        
        UV[3] = utils.lim(UV[3], UV3, limit_UV)
        UV[8] = utils.lim(UV[8], UV8, limit_UV)
        UV[10] = utils.lim(UV[10], UV10, limit_UV)
        UV[12] = utils.lim(UV[12], UV12, limit_UV)
        

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
        if t_rf >= reflexPars.S_GRAN and (ABRreflexOn==1 or CPRreflexOn==1): # Call every 0.0625 s
            t_rf_onset = t # New onset time reflex function
            t_rf = 0 # New time in new reflex function
            rf_switch = 1 # Reflex function switch ON
            # create impulse response
            impulseFunction(abp_buffer, rap_buffer)

        if t >= 4*reflexPars.S_GRAN: # Linear interpolation between impulse activations
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
                alpha_rad_full = utils.alpha_rad_full_func(tilt_time2, tau_STS, p=power)
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
        global grav_switch
        grav_switch = 1

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
        P[6]=(E[6]*(V[6]-UV[6]))+P_muscle_abdom
        P[7]=(E[7]*(V[7]-UV[7]))+P_muscle_abdom
        P[8]=(E[8]*(V[8]-UV[8]))+P_muscle_abdom
        P[9]=(E[9]*(V[9]-UV[9]))+P_muscle_abdom
        P[11]=(E[11]*(V[11]-UV[11]))+P_muscle_legs

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

        if grav_switch == 0: 

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

        if grav_switch == 1: 

            ### NEW ###
            #------------------------------------------------------#
            #---------- Left side of the heart --------------------#
            F = np.zeros(24)
            if P[20]>P[0]+Pgrav[20]:
                F[0]=(P[20]-P[0]-Pgrav[20])/R[0,0]; #q0
            else:
                F[0]=0; # no insufficiency
            #-----------Intrathoracic flow rates-------------------#
            F[1]=(P[0]-P[1]-Pgrav[0])/R[0,1]
            if F[1]<0:
                F[1]=0
            F[2]=(P[1]-P[2]-Pgrav[1])/R[0,2]
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
            F[6]=(P[0]-P[5]+Pgrav[0])/R[0,5]
            if F[6]<0:
                F[6]=0
            F[7]=(P[5]-P[6]+Pgrav[5])/R[0,6]
            if F[7]<0:
                F[7]=0
            F[8]=(P[6]-P[7]+Pgrav[6])/R[0,7]
            F[9]=(P[7]-P[8])/R[0,8]; # No influence of gravity for this flow.
            F[10]=(P[8]-P[13]-Pgrav[8])/R[3,8]
            F[11]=(P[6]-P[9]+Pgrav[6])/R[0,9]
            F[12]=(P[9]-P[10])/R[0,10]; # No influence of gravity for this flow.
            if P[10]>P[13]+Pgrav[10]:
                F[13]=(P[10]-P[13]-Pgrav[10])/R[3,10]
            else: 
                F[13]=0
            F[14]=(P[6]-P[11]+Pgrav[6])/R[0,11]
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
        dVdt[1] = F[1]-F[2]
        dVdt[2] = F[2]-F[3]
        dVdt[3] = F[3]-F[4] - F_micro_upper
        dVdt[4] = F[4]-F[5] + F_lymphatic
        dVdt[5] = F[6]-F[7]
        dVdt[6] = F[7]-F[8]-F[11]-F[14]
        dVdt[7] = F[8]-F[9]
        dVdt[8] = F[9]-F[10]
        dVdt[9] = F[11]-F[12]
        dVdt[10] = F[12]-F[13]
        dVdt[11] = F[14]-F[15]
        dVdt[12] = F[15]-F[16] - F_micro_lower
        dVdt[13] = F[10]+F[13]+F[16]-F[17]
        dVdt[14] = F[17]-F[18]
        dVdt[15] = F[5]+F[18]-F[19]
        dVdt[16] = F[19]-F[20]
        dVdt[17] = F[20]-F[21]
        dVdt[18] = F[21]-F[22]
        dVdt[19] = F[22]-F[23]
        dVdt[20] = F[23]-F[0]
        dVdt[-1] = F_micro_lower + F_micro_upper - F_lymphatic

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
            store_TBV[idx] = sum_v
            store_impulse[:, idx] = [para_resp, beta_resp, alpha_resp, alpha_respv, alphav_resp, alphav_respv]

            # Autoregulation step I -> Get a constant array of the aortic arch and rap pressure of the most recent 4*S_GRAN seconds
            carotid_recep = 1 * scaling["carotid_r_switch"]
            if idx_check == idx: # check if the same value is changed
                if carotid_recep == 0:
                    abp_buffer[-1] = store_P[0, idx] #Note, the baroreflex does not incorparate the baroreceptors in the carotid arteries 
                if carotid_recep == 1:
                    abp_buffer[-1] = (store_P[0, idx] + store_P[2, idx])/2 #Note, NOW it does
                rap_buffer[-1] = store_P[15, idx]
            else: # otherwise, move the filter
                if carotid_recep == 0:
                    abp_buffer = np.append(abp_buffer[1:], store_P[0, idx]) #Note, the baroreflex does not incorparate the baroreceptors in the carotid arteries 
                if carotid_recep == 1:
                    abp_buffer = np.append(abp_buffer[1:], (store_P[0, idx] + store_P[2, idx])/2) #Note, NOW it does 
                rap_buffer = np.append(rap_buffer[1:], store_P[15, idx])
            idx_check = idx
        store_t.append(t); # Check all calculated timesteps
        
        return dVdt

    y0 = np.zeros(22)
    y0[0:22] = V # volumes
        
    esse = esse_cerebral_NEW
    #solution = solve_ivp(esse_test, t_span, y0, t_eval=t_eval, method='RK45')
    # Create an instance of the tracker
    #sol = solve_ivp(esse, t_span, y0, first_step=T*0.1, max_step=T, t_eval=t_eval, method='RK23', rtol=1e-4, atol=1e-4)
    sol = solve_ivp(esse, t_span, y0, first_step=T*0.1, max_step=T, t_eval=t_eval, method='RK23', rtol=1e-3, atol=1e-3)

    # Find the extremes for compartment 3 (index 2)
    extremes_BP = utils.find_extremes_of_cardio_cycles(store_P[2], t_eval, t_cc_onset)
    mean_t = extremes_BP[:,0]
    store_BP_max = extremes_BP[:,1]
    store_BP_min = extremes_BP[:,2]
    store_finap = store_BP_max*1/3+2/3*store_BP_min
    HR_list = HR_list[:-2]
    Out_av.append([mean_t, map, store_finap, store_HR, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra, store_P_muscle, store_P_muscle2, store_E, store_UV, store_TBV, store_impulse])
    Timer.stop()

    outputP = output9 = outputP_intra = output10 = outputPg = outputE = []
    Out_wave.append([store_t, outputP, output9, outputP_intra, output10, outputPg, outputE])
    Out_solver = [sol.t, sol.y]
    return Out_av, Out_wave, Out_solver