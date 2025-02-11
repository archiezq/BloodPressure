""" !!! Start the simulation !!! """
def esse_cerebral_NEW(t, x_esse):
    global nrr, ncc, nrf, n, n4, HP, V_micro, ElvMIN, ErvMIN, ErvMAX, ElvMAX
    global abp_buffer, rap_buffer, abp_ma, abp_pp, rap_ma, abp_hist, rap_hist, abp_hist, rap_hist, pp_hist
    global V, P, F, Pgrav, E, R, UV, vl, t_temp, cvp_temp, co_temp, hr_temp, store_finap_temp
    global hydroP, alpha_tilt, Tav, Tsa, Tsv, P_muscle_thorax, P_muscle_abdom, P_muscle_legs, P_intra, F_micro_lower, F_micro_upper, F_lymphatic
    global cc_switch, t_rf, t_rf_onset, t_cc, t_cc_onset, t_resp_onset, impulse
    global para_resp,beta_resp,alpha_resp,alpha_respv,alphav_resp,alphav_respv, idx_check
    global oxy_switch, k_c, s_v, phi_c, h_c, phi_t, alpha_b, alpha_t, M_max, C_50, C_t, C_c
    global last_oxy_reset
    """ Update values """
    V = x_esse[:22]  # Volumes of cardiovascular model
    V_micro = V[-1] # V_micro
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
    if oxy_switch == 1 and cerebralVeinsOn==1:
        C_oxy = x_esse[37+carotidOn] # oxygen concentration in the brain
    if oxy_switch == 1 and cerebralVeinsOn==0:
        C_oxy = x_esse[27+carotidOn]
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
    


    # ----------------------- same part --------------------------
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
    
    if oxy_switch == 1:
        # dcdt = q_in*C_oxy_inlet + dC_dt(C_oxy)
        dcdt = dC_dt(C_oxy)

        # print(dcdt)

        
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

    # ------------------------ Different parts ------------------------
    
    # calculate autoregulation
    #crb.x_aut += crb.dx_autdt*crb.dt 
    # conditions for vasoconstriction/vasodilation
    if crb.x_aut <= 0.0: # vasodilation, added OR EQUALS
        crb.DeltaC_pa = crb.DeltaC_pa1
        crb.kC_pa = crb.DeltaC_pa1/4.0
    if crb.x_aut > 0.0: #vasocontriction
        crb.DeltaC_pa = crb.DeltaC_pa2
        crb.kC_pa = crb.DeltaC_pa2/4.0
        
    # calculate intranical capacity equation (4)  2000
    crb.C_ic = 1.0/(crb.k_E*crb.P_ic)
    
    # calculate cerebral veins vi capacity from equation (6)  error? 2015
    crb.C_vi = 1.0/(crb.k_ven*(crb.P_v - crb.P_ic - crb.P_v1)) # initial
    # crb.C_vi = crb.k_ven/(crb.P_v - crb.P_ic - crb.P_v1)
    
    # calculate pial arteries capacity from equation (8) 2015
    crb.C_pa = (crb.C_pan-crb.DeltaC_pa/2.0 + (crb.C_pan + crb.DeltaC_pa/2.0)*np.exp(-crb.x_aut/crb.kC_pa))/(1.0+np.exp(-crb.x_aut/crb.kC_pa))

    # Formula uses dx_autdt from the previous timestep, but x_aut from the current timestep, is this correct?
    # crb.dC_padt = (-crb.DeltaC_pa*crb.dx_autdt)/(crb.kC_pa*(1+np.exp(-crb.x_aut/crb.kC_pa))) # a19 2000
    crb.dC_padt = (-crb.DeltaC_pa*np.exp(-crb.x_aut/crb.kC_pa)*crb.dx_autdt)/(crb.kC_pa*(1+np.exp(-crb.x_aut/crb.kC_pa))) # a19 2000
    
    # calculate pial arterial resistance from equation (11) 2015
    crb.R_pa = crb.k_R*crb.C_pan**2/(((crb.P_pa-crb.P_ic)*crb.C_pa)**2)
    
    # calculate capillary pressure from equation (4) 2015
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
        # 颈静脉-椎静脉血流
        
        # calculating flow in the collateral segments 
        # 侧支血管段
        crb.Q_c3 = crb.G_c3*(crb.P_vs - crb.P_c3)
        crb.Q_c2 = crb.G_c2*(crb.P_c3 - crb.P_c2)
        crb.Q_c1 = crb.G_c1*(crb.P_c2 - P[4]) # Qc1 Flow in the lower segment of the collateral network, does this flow into C[4]?

        # calculating flow in the anastomotic connections
        # 吻合连接
        crb.Q_cjr3 = crb.G_cjr3*(crb.P_c3 - crb.P_jr3)
        crb.Q_cjr2 = crb.G_cjr2*(crb.P_c2 - crb.P_jr2)
        crb.Q_cjl3 = crb.G_cjl3*(crb.P_c3 - crb.P_jl3)
        crb.Q_cjl2 = crb.G_cjl2*(crb.P_c2 - crb.P_jl2)

        # calculating flow in the jugular segments
        # 颈静脉节段
        crb.Q_jr3 = crb.G_jr3*(crb.P_vs - crb.P_jr3) 
        crb.Q_jl3 = crb.G_jl3*(crb.P_vs - crb.P_jl3) 
        crb.Q_jr2 = crb.G_jr2*(crb.P_jr3 - crb.P_jr2)
        crb.Q_jl2 = crb.G_jl2*(crb.P_jl3 - crb.P_jl2)
        crb.Q_jr1 = crb.G_jr1*(crb.P_jr2 - crb.P_svc1)
        crb.Q_jl1 = crb.G_jl1*(crb.P_jl2 - crb.P_svc1)

        # calculating flow in the vertebral veins
        # 椎静脉
        crb.Q_vv = crb.G_vvr*(crb.P_vs - crb.P_vv) + crb.G_vvl*(crb.P_vs - crb.P_vv)
        crb.Q_vvr = crb.G_vvr*(crb.P_vs - crb.P_vv)
        crb.Q_vvl = crb.G_vvl*(crb.P_vs - crb.P_vv)

        # Calculating flow in the superior vena cava (superior tract)
        # 上腔静脉
        #crb.Q_svc1 = crb.G_jr1*(crb.P_jr2 - crb.P_svc1) + crb.G_jl1*(crb.P_jl2 - crb.P_svc1) # why?
        crb.Q_svc1 = crb.Q_jr1 + crb.Q_jl1 # doesn't use P[4]

        # Calculating flow in the renal vein
        # 肾静脉
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

    if grav_switch == 0: 

        ### ORIGINAL ###
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
        store_oxygen[:, idx] = dcdt

        if cerebralVeinsOn==1:
            store_crb_Q_j[:, idx] = crb.Q_c3, crb.Q_c2, crb.Q_c1, crb.Q_jr3, crb.Q_jr2, crb.Q_jr1, crb.Q_jl3, crb.Q_jl2, crb.Q_jl1 # jugular flows
            store_crb_Q_v[:, idx] = crb.Q_vvr, crb.Q_vvl, crb.Q_vv # vertebral flows
            store_crb_Q_out[:, idx] = crb.Q_lv, crb.Q_svc2, crb.Q_svc1, crb.Q_rv, crb.Q_azy2, crb.Q_out_net, crb.Q_tot # azygos, svc and out flows
            store_crb_P[:, idx] = crb.P_v, crb.P_pa # Pressure in the cerebral veins and pial arterioles
            store_crb_G[:, idx] = crb.G_jr3, crb.G_jr2, crb.G_jr1, crb.G_jl3, crb.G_jl2, crb.G_jl1, crb.G_c1, crb.G_cjr3, crb.G_cjr2, crb.G_cjl3, crb.G_cjl2, crb.G_c3 # jugular conductance

        # Autoregulation step I -> Get a constant array of the aortic arch and rap pressure of the most recent 4*S_GRAN seconds
        carotid_recep = 1 * scaling["carotid_r_switch"]
        if idx_check == idx: # check if the same value is changed
            if carotid_recep == 0:
                abp_buffer[-1] = store_P[0, idx] #Note, the baroreflex does not incorparate the baroreceptors in the carotid arteries 
            if carotid_recep == 1:
                abp_buffer[-1] = (store_P[0, idx] + store_P[2, idx])/2 #Note, NOW it does
            rap_buffer[-1] = store_P[15, idx]
            crb.Qbrain_buffer[-1] = store_crb[idx] # Cerebral blood flow buffer
        else: # otherwise, move the filter
            if carotid_recep == 0:
                abp_buffer = np.append(abp_buffer[1:], store_P[0, idx]) #Note, the baroreflex does not incorparate the baroreceptors in the carotid arteries 
            if carotid_recep == 1:
                abp_buffer = np.append(abp_buffer[1:], (store_P[0, idx] + store_P[2, idx])/2) #Note, NOW it does 
            rap_buffer = np.append(rap_buffer[1:], store_P[15, idx])
            crb.Qbrain_buffer = np.append(crb.Qbrain_buffer[1:], store_crb[idx]) # Cerebral blood flow buffer
        idx_check = idx
    store_t.append(t); # Check all calculated timesteps

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
    return np.concatenate([dVdt, crb_ddt, [dcdt]])

