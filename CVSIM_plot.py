def CVSIM_plot(config):

        # Extract necessary data from config
        data_time_10Hz = config.get("data_time_10Hz", [])
        data_MAP = config.get("data_MAP", [])
        data_HR = config.get("data_HR", [])
        cerebralModelOn = config.get("cerebralModelOn", 0)
        time_after_sts = config.get("time_after_sts", 179.9)
        resting_time = config.get("resting_time", 180)
        sts_n = config.get("sts_n", 180)
        t_mean = config.get("t_mean", [])
        t_eval = config.get("t_eval", [])
        store_BP_min = config.get("store_BP_min", [])
        store_BP_max = config.get("store_BP_max", [])
        data_DBP = config.get("data_DBP", [])
        data_SBP = config.get("data_SBP", [])
        delta_t = config.get("delta_t", 0.1)
        HR_list = config.get("HR_list", [])
        store_P_muscle2 = config.get("store_P_muscle2", [])
        store_impulse = config.get("store_impulse", [])
        store_TBV = config.get("store_TBV", [])
        store_P = config.get("store_P", [])
        store_UV = config.get("store_UV", [])
        store_E = config.get("store_E", [])
        scaling = config.get("scaling", [])
        y_solver = config.get("y_solver", [])
        t_solver = config.get("t_solver", [])
        # store_oxygen = config.get("store_oxygen", [])
        Cc_store = config.get("Cc_store", [])
        Ct_store = config.get("Ct_store", [])

        import matplotlib.pyplot as plt
        import CVSIM_utils as utils
        import numpy as np
        import matplotlib.pyplot as plt
        from reflexPars import _init_reflex # Get the control parameters loaded.
        from adultPars_carotid import _init_pars # Get the parameters for resistance, elastance and uvolume
        from matplotlib.ticker import ScalarFormatter

        reflexPars = _init_reflex(scaling); # Get all the reflex parameters stored to the list 'reflexPars'.
        subjectPars = _init_pars(scaling); # Here the compartments parameters are assigned

        if cerebralModelOn==1:
                tmean_mca = config.get("tmean_mca", [])
                store_crb_Q_ic = config.get("store_crb_Q_ic", [])
                store_crb_mca = config.get("store_crb_mca", [])
                store_V_mca_max = config.get("store_V_mca_max", [])
                store_V_mca_min = config.get("store_V_mca_min", [])
                carotidOn = config.get("carotidOn", 0)
                oxy_switch = config.get("oxy_switch", 0)

                if oxy_switch == 1:
                        store_oxygen = config.get("store_oxygen", [])

        """
        #IMPORT
        if dataset != 2:
                data_time_10Hz = np.squeeze(data['data']['time_10Hz'][0][0])
                data_MAP = np.squeeze(data['data']['map'][0][0])
                data_HR = np.squeeze(data['data']['hr'][0][0])
        """
        # magic with indices and time for allignment
        if time_after_sts > 179.9:
                time_after_sts = 179.9
        
        dt_data = round(data_time_10Hz[1]-data_time_10Hz[0],1)
        start_index_model = int(resting_time/2/dt_data)
        index_end = int((time_after_sts+resting_time)/dt_data)
        
        index_data_begin = utils.find_index_of_time(data_time_10Hz, round(sts_n,1))-start_index_model
        index_data_end = index_data_begin+index_end-start_index_model
        x_data = data_MAP[index_data_begin:index_data_end]
        
        t_translated = t_mean + sts_n - resting_time
        t_eval_trans = t_eval + sts_n - resting_time
        
        # INTERPOLATE FINAP #
        # Desired sampling frequency
        desired_frequency = 10  # Hz
        desired_interval = 1 / desired_frequency  # Interval in seconds
        
        # Sys and Dia
        new_times, new_minP = utils.lin_interp(t_mean, store_BP_min, desired_interval)
        new_times, new_maxP = utils.lin_interp(t_mean, store_BP_max, desired_interval)
        t_translated_new = (new_times + sts_n - resting_time)
        window_t_trans = t_translated_new[start_index_model:index_end]
        window_new_minP = new_minP[start_index_model:index_end]
        window_new_maxP = new_maxP[start_index_model:index_end]
        
        # Set default font size for all plots
        plt.rcParams['font.size'] = 15  # Replace 14 with your desired font size

        # Plotting
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(window_t_trans, window_new_minP, label=r'$Model_{dia}$', color='navy')
        plt.plot(window_t_trans, window_new_maxP, label=r'$Model_{sys}$', color='darkred')
        plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_DBP[index_data_begin:index_data_end], label=r'$Data_{dia}$', color='cornflowerblue', ls="--")
        plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_SBP[index_data_begin:index_data_end], label=r'$Data_{sys}$', color='lightcoral', ls="--")
        plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
        plt.axvline(x=sts_n+delta_t, color='grey', linestyle='--', label=r'$t_0 + \Delta t$')  # Add vertical line at UV
        plt.xlabel('$t$ (s)')
        plt.ylabel(r'$P$ (mmHg)')
        plt.title(r'Plot of systolic and diastolic $P$ vs time')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()
        
        # HR 1
        lip_HR = utils.lin_interp(t_mean, HR_list, desired_interval)[1][start_index_model:index_end]
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_HR[index_data_begin:index_data_end], label=r'$HR_{data}$', color='cornflowerblue', ls="--")
        plt.plot(t_translated_new[start_index_model:index_end], lip_HR, label=r'$HR_{model}$', color='navy', linestyle="-")
        plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
        #plt.axvline(x=sts_n+delta_t, color='grey', linestyle='--', label=r'$t_0 + \Delta t$')  # Add vertical line at UV
        plt.xlabel('$t$ (s)')
        plt.ylabel(r'$HR$ ($min^{-1}$)')
        plt.title(r'Plot of heart rate ($HR$) vs time ($t$)')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

        if cerebralModelOn==1:
                ### CRB PLOTS ###
                t_trans_mca = tmean_mca + sts_n - resting_time
                names_store_crb_Q_ic = ['$Q_{f}$', '$Q_{0}$', '$Q_{la}$', '$Q_{vs}$', '$Q_{auto}$', '$Q_{pa}$']
                """ store_crb_Q_ic:
                Q_f:    Cerebrospinal fluid formation rate
                Q_0:    Cerebrospinal fluid outflow rate
                Q_la:   Flow in the large cerebral arteries
                Q_vs:   Venous sinus flow
                Q_auto: average cerebral flow over crb_buffer_size [s]: np.mean(crb.Qbrain_buffer)
                Q_pa:   Flow in the pial arterioles
                """

                # crb Q (Cerebral inflow)
                plt.figure(figsize=(10, 6), dpi=300)
                plt.plot(t_eval_trans[9000:12000], store_crb_Q_ic[5, 9000:12000], label=names_store_crb_Q_ic[5], linewidth=0.5, linestyle="-")
                plt.plot(t_eval_trans[9000:12000], store_crb_Q_ic[2, 9000:12000], label=names_store_crb_Q_ic[2], linewidth=0.5, linestyle="-")
                # Add vertical line at specific time points (sts_n)
                plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')
                plt.xlabel('t (s)')
                plt.ylabel(r'$Q$ ($mL \times s^{-1}$)')
                plt.title(r'Plot of cerebral inflow vs time ($t$)')
                plt.legend(loc='upper left')
                plt.grid(True)
                plt.show()

                # crb MCA (Middle cerebral artery) velocity
                plt.figure(figsize=(10, 6), dpi=300)
                plt.plot(t_eval_trans[9000:12000], store_crb_mca[0, 9000:12000], label=r'$V_{MCA}$', color='navy', linewidth=0.5, linestyle="-")
                plt.plot(t_trans_mca, store_V_mca_max, label=r'$V_{MCA_{sys}}$', color='lightcoral', linestyle='--')
                plt.plot(t_trans_mca, store_V_mca_min, label=r'$V_{MCA_{dia}}$', color='cornflowerblue', linestyle='--')
                plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
                plt.xlabel('t (s)')
                plt.ylabel(r'$V$ ($cm \times s^{-1}$)')
                plt.title(r'Plot of middle cerebral artery velocity ($V_{MCA}$) vs time')
                plt.legend(loc='upper left')
                plt.grid(True)
                plt.show()

                # crb MCA (Middle cerebral artery) velocity
                plt.figure(figsize=(10, 6), dpi=300)
                plt.plot(t_eval_trans[9000:12000], store_crb_mca[0, 9000:12000], label=r'$V_{MCA}$', color='navy', linewidth=0.5, linestyle="-")
                plt.plot(t_trans_mca[60:170], store_V_mca_max[60:170], label=r'$V_{MCA_{sys}}$', color='lightcoral', linestyle='--')
                plt.plot(t_trans_mca[60:170], store_V_mca_min[60:170], label=r'$V_{MCA_{dia}}$', color='cornflowerblue', linestyle='--')
                plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
                plt.xlabel('t (s)')
                plt.ylabel(r'$V$ ($cm \times s^{-1}$)')
                plt.title(r'Plot of middle cerebral artery velocity ($V_{MCA}$) vs time')
                plt.legend(loc='upper left')
                plt.grid(True)
                plt.show()

                # crb MCA (Middle cerebral artery) radius
                plt.figure(figsize=(10, 6), dpi=300)
                plt.plot(t_eval_trans[9000:12000], store_crb_mca[1, 9000:12000], label=r'$V_{MCA}$', color='b', linewidth=0.5, linestyle="-")
                plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
                plt.xlabel('t (s)')
                plt.ylabel(r'$r$ ($cm$)')
                plt.title(r'Plot of middle cerebral artery radius ($r_{MCA}$) vs time')
                plt.legend(loc='upper left')
                plt.grid(True)
                plt.show()
                
                print("This is the oxy_switch: ", oxy_switch)
                
                if oxy_switch == 1:
                        # print(store_oxygen)
                        print(f"The length of oxygen is:, {len(store_oxygen)}")
                        plt.figure(figsize=(10, 6), dpi=300)

                        # plt.plot(t_eval_trans[9000:12000], store_oxygen[0, 9000:12000], label='C_t (Oxygen Concentration in Tissue)')
                        # plt.plot(t_eval_trans[:1000], store_oxygen[0][:1000], label='C_t (Oxygen Concentration in Tissue)')
                        plt.plot(t_eval_trans[7000:], store_oxygen[0][7000:], label='C_t (Oxygen Concentration in Tissue)')
                        plt.xlabel('Time (s)')
                        plt.ylabel('C_t (m3 O2/m3 tissue)')
                        plt.title('Tissue Oxygen Concentration vs Time')
                        plt.legend(loc='upper left')
                        plt.grid(True)
                        ax = plt.gca()
                        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                        plt.show()

                        plt.figure(figsize=(10, 6), dpi=300)
                        plt.plot(t_eval_trans[10000:16000], store_oxygen[0][10000:16000], label='C_t (Oxygen Concentration in Tissue)')  
                        plt.xlabel('Time (s)')
                        plt.ylabel('C_t (m3 O2/m3 tissue)')
                        plt.title('Tissue Oxygen Concentration vs Time')
                        plt.legend(loc='upper left')
                        plt.grid(True)
                        ax = plt.gca()
                        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                        plt.show()

                if carotidOn==1:
                        # plot carotid pressure
                        plt.figure(figsize=(10, 6), dpi=300)
                        plt.plot((t_solver + sts_n - resting_time), y_solver[27], label=r'$P_{carotid}$', color='#1f77b4', linewidth=1)  # Soft blue
                        plt.plot((t_solver + sts_n - resting_time), y_solver[24]+y_solver[25], label=r'$P_{PA}$', color='#2ca02c', linewidth=1)  # Soft green
                        plt.axvline(x=sts_n, color='gray', linestyle='--', label=r'$t_0$')
                        plt.xlabel('t (s)')
                        plt.ylabel(r'$P$ (mmHg)')
                        plt.title(r'Plot of carotid pressure ($P_{carotid}$) vs time')
                        plt.legend(loc='upper left')
                        plt.grid(True)
                        plt.show()

                
        # Plot the stand-up pressures
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(t_eval_trans[11000:13500], store_P_muscle2[0][11000:13500], 
                '-', label='$P_{Legs}$', linewidth=2, color='steelblue')
        plt.plot(t_eval_trans[11000:13500], store_P_muscle2[1][11000:13500], 
                '-', label='$P_{Abdomen}$', linewidth=2, color='darkorange')
        #plt.plot(t_eval_trans[11000:13500], store_P_muscle2[2][11000:13500], 
        #        '-', label='$P^{muscle}Thorax$', linewidth=2, color='teal')
        plt.plot(t_eval_trans[11000:13500], store_P_muscle2[3][11000:13500], 
                '-', label='$P_{Thorax}$', linewidth=2, color='mediumseagreen')

        #plt.plot(t_eval_trans[10000:16000], store_P[12][10000:16000], '-', label='$P_{12}$', linewidth=.5)
        plt.legend(loc='upper left')
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (mmHg)')
        #plt.vlines(x=[sts_n], ymin=[-10], ymax=[20], colors='k', ls='--', lw=1, label='stand-up')
        plt.axvline(x=sts_n, color='black', linestyle='--', linewidth=2, label='stand-up')  
        plt.grid(True)
        #plt.ylim(-5,50)
        plt.show()


        p = reflexPars.p
        s = reflexPars.s
        a = reflexPars.a
        v = reflexPars.v
        cpv = reflexPars.cpv
        cpa = reflexPars.cpa
        x = np.linspace(0,60,960)
        
        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(x, p, '-', label='$I_p$', linewidth=2) # parasympathetic
        plt.plot(x, s, '-', label='$I_s$', linewidth=2) # sympathetic
        plt.plot(x, a, '-', label='$I_a$', linewidth=2) # arterial
        plt.plot(x, v, '-', label='$I_v$', linewidth=2) # venous
        plt.plot(x, cpa, '-', label='$I_{cpa}$', linewidth=2) # cardiopulmonary arterial
        plt.plot(x, cpv, '-', label='$I_{cpv}$', linewidth=2) # cardiopulmonary venous
        plt.legend()
        # Set y-axis ticks to display only three specific values
        plt.yticks([0.00, 0.01, 0.02])  # Replace with the values you want to display
        plt.yticks([0.00, 0.01, 0.02])  # Replace with the values you want to display
        plt.xlabel('t (s)')
        plt.ylabel('Normalized Response')
        plt.ylim(0,0.02)
        plt.xlim(0,45)
        plt.show()

        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(t_eval_trans , store_impulse[0] , 
                '-', label='$BRParaResp$', linewidth=2)
        plt.plot(t_eval_trans , store_impulse[1] , 
                '-', label='$BRSympResp$', linewidth=2)
        plt.plot(t_eval_trans , store_impulse[2] , 
                '-', label='$BRRResp$', linewidth=2)
        plt.plot(t_eval_trans , store_impulse[3] , 
                '-', label='$BRVResp$', linewidth=2)
        plt.plot(t_eval_trans , store_impulse[4] , 
                '-', label='$CPRResp$', linewidth=2)
        plt.plot(t_eval_trans , store_impulse[5] , 
                '-', label='$CPVResp$', linewidth=2)
        plt.legend(loc='upper left')
        plt.xlabel('Time (s)')
        plt.ylabel('Impulse')
        plt.axvline(x=sts_n, color='black', linestyle='--', linewidth=2, label='stand-up')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(t_eval_trans, store_TBV, '-', label='TBV', linewidth=2)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('TBV (mL)')
        plt.axvline(x=sts_n, color='black', linestyle='--', linewidth=2, label='stand-up')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(t_eval_trans, store_P[12], '-', label='PLowerBody', linewidth=2)
        plt.plot(t_eval_trans, store_P[14], '-', label='PInfVC', linewidth=2)
        plt.plot(t_eval_trans, store_P[15], '-', label='PRightAtr', linewidth=2)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (mmHg)')
        plt.axvline(x=sts_n, color='black', linestyle='--', linewidth=2, label='stand-up')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6), dpi=300)
        plt.plot(t_eval_trans, store_P[0], '-', label='PAscAo', linewidth=0.5)
        plt.plot(t_eval_trans, store_P[2], '-', label='PUppBo', linewidth=0.5)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Pressure (mmHg)')
        plt.axvline(x=sts_n, color='black', linestyle='--', linewidth=2, label='stand-up')
        plt.grid(True)
        plt.show()


        """
        fig = plt.figure()
        fig.figure(figsize=(10, 6), dpi=300)
        ax = fig.add_subplot(1,1,1)
        ax.plot(x, p, '-', label='p', linewidth=2)
        ax.plot(x, s, '-', label='s', linewidth=2)
        ax.plot(x, a, '-', label='a', linewidth=2)
        ax.plot(x, v, '-', label='v', linewidth=2)
        ax.plot(x, cpv, '-', label='cpv', linewidth=2)
        ax.plot(x, cpa, '-', label='cpa', linewidth=2)
        
        ax.legend()
        ax.set_xlabel('T (s)')
        ax.set_ylabel('Normalized Response')
        #ax.set_ylim(0,0.3)
        ax.set_ylim(0,0.015)
        plt.show()
        
        SUMZ = reflexPars.SUMZ
        """
        # Non-lin compliance plot
        P_muscle = 0
        #V = y_solver[12]
        UV12 = store_UV[12][0]
        Vmax12 = 1000
        E = subjectPars.elastance[0,12]
        V = np.linspace(0, 0.99*(Vmax12+UV12), 1000)
        P_nonlin=(np.tan((V-UV12)/(2*Vmax12/np.pi)))/((np.pi*1/E)/(2*Vmax12))+P_muscle;
        #P_nonlin=np.tan(np.pi*(V-UV12)/(2*Vmax12))*(2*Vmax12*E)/np.pi+P_muscle;
        
        # Plotting
        plt.figure(figsize=(10, 6), dpi=300)
        # Plot with thicker line
        plt.plot(V, P_nonlin, label=r'$P_{\text{12}}$', color='b', linewidth=2)

        # Vertical line for UV with increased thickness
        plt.axvline(x=UV12, color='g', linestyle='--', linewidth=2, label=r'$UV_{\text{12}}$')  

        # Vertical line for UV + Vmax12 with increased thickness
        plt.axvline(x=UV12 + Vmax12, color='r', linestyle='--', linewidth=2, label=r'$UV_{\text{12}} + V^{max}_{\text{12}}$')  

        # Add labels, title, legend, and grid
        plt.xlabel(r'$V_{\text{12}}$ (mL)')
        plt.ylabel(r'$P_{\text{12}}$ (mmHg)')
        plt.title(r'Plot of $V_{\text{12}}$ vs $P_{\text{12}}$')
        plt.legend(loc='upper left')

        # Set axis limits
        plt.xlim(580, 1600)
        plt.ylim(-10, 300)

        plt.grid(True)
        plt.show()


        # Cardiac E
        Tav=0.12*np.sqrt(1)
        Tsa=0.2*np.sqrt(1)
        Tsv=0.3*np.sqrt(1)
        t_start = 0
        t_end = t_start + 80
        # Assuming t_eval is your time vector and you want to plot up to t_end
        min_time = t_eval[0]
        max_time = t_eval[t_end - 1]
        mid_time = (min_time + max_time) / 2
        # Plotting
        plt.figure(figsize=(8, 4), dpi=300)
        plt.plot(t_eval[:t_end], store_E[0,:t_end], linewidth=2, label=r'$RA$', color='cornflowerblue', ls="--")
        plt.plot(t_eval[:t_end], store_E[1,:t_end], linewidth=2, label=r'$RV$', color='navy')
        plt.plot(t_eval[:t_end], store_E[2,:t_end], linewidth=2, label=r'$LA$', color = 'lightcoral', ls="--")
        plt.plot(t_eval[:t_end], store_E[3,:t_end], linewidth=2, label=r'$LV$', color='darkred')
        #plt.axvline(x=Tav, color='chocolate', linestyle='--', label=r'$Tav$')  # Add vertical line at UV
        #plt.axvline(x=Tsa, color='lightgreen', linestyle='--', label=r'$Tsa$')  # Add vertical line at UV
        #plt.axvline(x=Tav+Tsv, color='darkred', linestyle='--', label=r'$Tsv$')  # Add vertical line at UV
        plt.xlabel('t (s)')
        plt.ylabel(r'E (mmHg/mL)')
        plt.title(r'Plot of Cardiac Elastance (E) vs Time (t)')
        # Set x-ticks to min, mid, and max
        plt.xticks([min_time, mid_time, max_time], [f"{min_time:.1f}", f"{mid_time:.1f}", f"{max_time:.1f}"])
        plt.legend()
        plt.grid(False)
        plt.show()

        
        return