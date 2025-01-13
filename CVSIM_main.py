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
import pandas as pd
import csv
import platform
from CVSIM_plot import CVSIM_plot as cvplt
import CVSIM_utils as utils
import CVSIM_Brain
import CVSIM
from controlPars import _init_control # Get the control parameters loaded.

#import PySimpleGUI as sg

#%%
# General settings

print('Healthy adult sim run')

planet=1
fluidLoading=0
StStest=1
cerebralModelOn=1 # 1 = cerebral model on, 0 = cerebral model off
if cerebralModelOn==1: # Change these
    cerebralVeinsOn=0
    carotidOn=1
else: # Do not change these
    cerebralVeinsOn=0
    carotidOn=0
ABRreflexOn=1; # ABR reflex
CPRreflexOn=1; # CPR reflex
micro_switch = 1 # 1 = microcirculation on, 0 = microcirculation off
bleedingOn,startTimeBleeding,totalTimeBleeding,bleedingVolume=0,150,180,2000;#  time in seconds
fillingOn,totalTimeFilling,fillingVolume=0,200,1000;#  time in seconds

if cerebralModelOn==1:
    print("BRAIN = ON")
    if cerebralVeinsOn==0:
        print("BRAIN VEINS = OFF")
    else:
        print("BRAIN VEINS = ON")
    if carotidOn==1:
        print("CAROTID = ON")
    else:
        print("CAROTID = OFF")
else:
    print("BRAIN = OFF")

controlPars = _init_control(); # Get all the control parameters stored to the list 'control'.You can access the heart rate for instance by contorl.hr.
        
### IMPORT ###
dataset = 2 # 0=NIVLAD, 1=PROHEALTH, 2=PROHEALTH_AVERAGE
standUp_n = 1 # 1, 2 or 3

# Determine if running on Snellius based on hostname
if "snellius" in platform.node():
    snellius = 1
else:
    snellius = 0

global data_SBP, data_DBP, data_HR, data_BP, data_MAP, data_time, data_time_10Hz, sts_n_end
# Dataset path logic
if dataset == 0:
    filename  = r"C:\Users\jsande\Documents\UvA\Data\AMC_OneDrive_4-5-2024\Data_NILVAD\NILVAD_preprocessed_data\Preprocessed_PIN43028_T2_SSS.mat"
    data, data_DBP, data_SBP, data_time, data_BP, data_time_10Hz, data_MAP = utils.get_data(filename)
    marker_name = 'markers'
    markers = np.squeeze(data['data'][marker_name][0][0][0][0])
    sts0 = markers[0][0][0]; # start (for calibratin of setpoints)
    sts1 = markers[4][0][0]; # 1st stand-up (for calibratin of setpoints)
    sts_n = markers[3+standUp_n][0][0]
    sit_n = markers[standUp_n][0][0]
    # Add automatic data extraction, done as for PROHEALTH

if dataset == 1:
    if snellius == 1:
        # Paths for when running on Snellius
        filename = "/gpfs/home1/jvandesande/Preprocessed_data/PHI022/Preprocessed_PHI022_FSup.mat"
        csv_file_path = "/gpfs/home1/jvandesande/Preprocessed_data/PROHEALTH-I_export_olderadults.csv"
    else:
        # Paths for when running locally
        filename = r"C:\Users\jsande\Documents\UvA\Data\AMC_OneDrive_4-5-2024\Data validation study PROHEALTH\Preprocessed data\PHI022\Preprocessed_PHI022_FSup.mat"
        csv_file_path = r"C:\Users\jsande\Documents\UvA\Data\AMC_OneDrive_4-5-2024\Data validation study PROHEALTH\PROHEALTH-I_export_olderadults.csv"

    data, data_DBP, data_SBP, data_time, data_BP, data_time_10Hz, data_MAP = utils.get_data(filename)
    patient_id = utils.extract_patient_id(filename)
    
    # Read the CSV file using the dynamically determined column names
    try:
        df = pd.read_csv(
            csv_file_path,
            header=0,                # Use the first row as the header
            na_values=['', 'NA'],    # Treat empty strings and 'NA' as NaN
            delimiter=';',           # Specify the delimiter
        )
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
    
    # Find patient with ID
    #patient_id = 'PHI022'
    patient_data = df[df['Record Id'] == patient_id]
    
    global sts_n_end
    # Extract the required columns
    if not patient_data.empty:
        length = patient_data['length'].values[0]
        weight = patient_data['weight'].values[0]
        bmi = patient_data['bmi'].values[0]
        age = patient_data['Age'].values[0]
        sex = patient_data['Sex'].values[0]
        print(f'Length: {length}, Weight: {weight}, BMI: {bmi}, Age: {age}, Sex: {sex}')
    else:
        print(f'Patient with ID {patient_id} not found.')
    marker_name = 'marker'
    markers = np.squeeze(data['data'][marker_name][0][0][0][0])
    sts0 = round(markers[6][0][0], 3); # 1st start (for calibration of setpoints)
    sts1 = round(markers[0][0][0], 3); # 1st stand-up (for calibration of setpoints)
    sts_n = round(markers[-1+standUp_n][0][0], 3); # 
    # Set a default value for sts_n_end before assigning from markers
    sts_n_end = float('nan')  # Initialize to NaN explicitly
    try:
        # Try to assign sts_n_end from the data file (it might be NaN or a number)
        sts_n_end = round(markers[11+standUp_n][0][0], 3)
    except (IndexError, TypeError):
        # Handle cases where accessing the markers data fails
        print("Error accessing markers data, sts_n_end set to NaN")
    sit_n = round(markers[17+standUp_n][0][0], 3); # 
    print("sts_n: ", sts_n)
    print("sts_n_end: ", sts_n_end)

if dataset == 2:
    fallers = 0
    if snellius == 1:
        # Paths for when running on Snellius
        csv_SBP = "/gpfs/home1/jvandesande/Preprocessed_data/Populations/mean_data_fallers_non_fallers_SBP.csv"
        csv_DBP = "/gpfs/home1/jvandesande/Preprocessed_data/Populations/mean_data_fallers_non_fallers_DBP.csv"
        csv_HR = "/gpfs/home1/jvandesande/Preprocessed_data/Populations/mean_data_fallers_non_fallers_HR.csv"
    if snellius == 0:
        # Paths for when running locally
        csv_SBP = r"D:\CLS\Thesis\Qi_CODE\mean_data_fallers_non_fallers_SBP.csv"
        csv_DBP = r"D:\CLS\Thesis\Qi_CODE\mean_data_fallers_non_fallers_DBP.csv"
        csv_HR = r"D:\CLS\Thesis\Qi_CODE\mean_data_fallers_non_fallers_HR.csv"
    if fallers == 1:
        data_SBP = pd.read_csv(csv_SBP)['Mean SBP Fallers (mmHg)']
        data_DBP = pd.read_csv(csv_DBP)['Mean DBP Fallers (mmHg)']
        data_HR = pd.read_csv(csv_HR)['Mean HR Fallers (bpm)']
    if fallers == 0:
        data_SBP = pd.read_csv(csv_SBP)['Mean SBP Non-Fallers (mmHg)']
        data_DBP = pd.read_csv(csv_DBP)['Mean DBP Non-Fallers (mmHg)']
        data_HR = pd.read_csv(csv_HR)['Mean HR Non-Fallers (bpm)']
    data_BP = data_MAP = (1/3)*data_SBP + (2/3)*data_DBP
    data_time = data_time_10Hz = pd.read_csv(csv_HR)['Time (s)']
    length = 169 # cm, men and women avg.
    weight = 78 # kg, men and women avg.
    sts0 = -120
    sts1 = 0
    sts_n = 0
    sts_n_end = float('nan') # Initialize to NaN explicitly
    sit_n = 180

sit_index = utils.find_index_of_time(data_time, sts0)
stand_index = utils.find_index_of_time(data_time, sts1)
ABP_setp = sum(data_BP[sit_index:stand_index])/(stand_index-sit_index)




#%%
# =============================================================================
# Assign the control parameters to set for how long you want to run the simulation
# =============================================================================
global resting_time, time_after_sts, tmax, T, resting_time, t_eval
resting_time = 120 # s resting time
time_buffer = 10 # s, have the simulation run a little longer to complete the cardiac cycle, don't touch this parameter
tmin=controlPars.tmin
time_after_sts = sit_n-sts_n
if time_after_sts > 179.9: # Limit the simulation to 100 seconds ater STS for speed-up, SHOULD REMOVE LATER
    time_after_sts = 179.9
tmax = resting_time + time_after_sts + time_buffer; # Here on can set for how long you the simulation want to run in seconds
#tmax=controlPars.tmax; # Here on can set for how long you the simulation want to run in seconds
t_span = (tmin, tmax)
#T=controlPars.T; # Sample frequency
T=0.01; # Sample frequency
N = round((tmax-tmin)/T)+1
#N=controlPars.N; # 
#t_values = np.linspace(tmin,tmax,int(tmax/T)) # create t-axis with (start point, end point, number of points)
t_eval = np.arange(tmin,tmax,T) # create t-axis with (start point, end point, delta T)

# =============================================================================
# Interventions
# =============================================================================

global delta_t, HP, limit_R, limit_UV, HR, abp_hist,pp_hist, rap_hist,para_resp,beta_resp,alpha_resp,alpha_respv,alphav_resp,alphav_respv
global s_abp, abp_ma,rap_ma,s_rap,abp_pp,pp_abp, ErvMAX, ElvMAX, V_micro, store_P, store_HR

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

global solve_config
solve_config = {'t_span': t_span,
                'T': T,
                't_eval': t_eval,
                'weight': weight,
                'length': length,
                'alpha_0': alpha_0,
                'planet': planet,
                'StStest': StStest,
                'TimeToStand': TimeToStand,
                'startTimeStS': startTimeStS,
                'strainTime': strainTime,
                'micro_switch': micro_switch,
                'cerebralVeinsOn': cerebralVeinsOn,
                'carotidOn': carotidOn,
                'ABRreflexOn': ABRreflexOn,
                'CPRreflexOn': CPRreflexOn,
                'fluidLoading': fluidLoading,
                'ABP_setp': ABP_setp,
                }

#%%
# Solve the model

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
    global scaling
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
        "baro_delay_para": 0, # parasympathetic baro_delay
        "baro_delay_sympa": 0, # sympathetic baro_delay
        "baro_delay_BR_R": 0, # baro_delay BR R
        "baro_delay_BR_UV": 0, # baro_delay BR UV
        "baro_delay_CP_R": 0, # baro_delay CP_R
        "baro_delay_CP_UV": 0, # baro_delay CP_UV
        "baro_stretch_para": inputs[24], # parasympathetic baro_delay
        "baro_stretch_sympa": inputs[25], # sympathetic baro_delay
        "baro_stretch_BR_R": inputs[26], # baro_delay BR R
        "baro_stretch_BR_UV": inputs[27], # baro_delay BR UV
        "baro_stretch_CP_R": inputs[28], # baro_delay CP_R
        "baro_stretch_CP_UV": inputs[29], # baro_delay CP_UV
        "baro_stretch2": 0, # baroreflex "stretch"
        "Rp_lungs": 1, # Rp lungs
        "limit_R": 0.5, # limit R
        "limit_UV": 0.1, # limit UV
        "Vmax": 1, # V max
        "venous_UV": 1, # venous UV
        "Rp_upper": 1, # Rp upper
        "carotid_r_switch": 1, # carotid receptor switch
        # Cerebral model
        "k_v": 1, # Cerebral model: doppler correction factor, correct doppler angle and converts maximal to average velocity
        "G_aut": 1 # Cerebral model: Gain of the autoregulation mechanism related to CBF variations
        }
    """
    "baro_delay_para": 0, # parasympathetic baro_delay
    "baro_delay_sympa": 0, # sympathetic baro_delay
    "baro_delay_BR_R": 0, # baro_delay BR R
    "baro_delay_BR_UV": 0, # baro_delay BR UV
    "baro_delay_CP_R": 0, # baro_delay CP_R
    "baro_delay_CP_UV": 0, # baro_delay CP_UV
    "baro_stretch_para": inputs[24], # parasympathetic baro_delay
    "baro_stretch_sympa": inputs[25], # sympathetic baro_delay
    "baro_stretch_BR_R": inputs[26], # baro_delay BR R
    "baro_stretch_BR_UV": inputs[27], # baro_delay BR UV
    "baro_stretch_CP_R": inputs[28], # baro_delay CP_R
    "baro_stretch_CP_UV": inputs[29], # baro_delay CP_UV

    "baro_delay_para": inputs[24], # parasympathetic baro_delay
    "baro_delay_sympa": inputs[25], # sympathetic baro_delay
    "baro_delay_BR_R": inputs[26], # baro_delay BR R
    "baro_delay_BR_UV": inputs[27], # baro_delay BR UV
    "baro_delay_CP_R": inputs[28], # baro_delay CP_R
    "baro_delay_CP_UV": inputs[29], # baro_delay CP_UV
    "baro_stretch_para": 0, # parasympathetic baro_delay
    "baro_stretch_sympa": 0, # sympathetic baro_delay
    "baro_stretch_BR_R": 0, # baro_delay BR R
    "baro_stretch_BR_UV": 0, # baro_delay BR UV
    "baro_stretch_CP_R": 0, # baro_delay CP_R
    "baro_stretch_CP_UV": 0, # baro_delay CP_UV
    """

    #scaling = {'RRsgain': np.float64(0.4247090909389059), 'RRpgain': np.float64(1.4023006555585769), 'beta': np.float64(0.9676064829760379), 'alpha': np.float64(0.30801155417727755), 'beta_resp_Erv': np.float64(1.039831396037958), 'beta_resp_Elv': np.float64(1.0244036523443205), 'alpha_resp': np.float64(1.5917116199240897), 'alphav_resp': np.float64(1.622268479254073), 'RAP_setp': np.float64(1.1895675824789789), 'ABP_setp': np.float64(0.996788766284258), 'max_RV_E': np.float64(1.1637383714750085), 'min_RV_E': np.float64(1.0736641350414131), 'max_LV_E': np.float64(1.583812838782982), 'min_LV_E': np.float64(1.3378645724234), 'SD_ratio': np.float64(0.8404091815244861), 'HR_t0': np.float64(1.1706438822250247), 'STS_pressure': np.float64(0.3144591586237568), 'Stand_muscleP': np.float64(4.118319390737045), 'Global_R': np.float64(0.7491135936792918), 'Global_UV': np.float64(1.3773794880382608), 'Rp': np.float64(1.1697461315030573), 'E_arteries': np.float64(1.1508263987923764), 'E_veins': np.float64(2.2831944910401747), 'v_ratio': np.float64(1.3547166415262744), 'baro_delay2': 0, 'baro_delay_para': np.float64(0.814349229568542), 'baro_delay_sympa': np.float64(2.2616582887833676), 'baro_delay_BR_R': np.float64(7.95160029023435), 'baro_delay_BR_UV': np.float64(0.498555752776751), 'baro_delay_CP_R': np.float64(1.158187699655198), 'baro_delay_CP_UV': np.float64(9.009277957063352), 'Rp_lungs': 1, 'limit_R': 0.5, 'limit_UV': 0.1, 'Vmax': 1, 'venous_UV': 1, 'Rp_upper': 1, 'carotid_r_switch': 1, 'k_v': 1, 'G_aut': 1}

    #print(scaling)
    
    ### RUN ###
    #print("Input parameters are:\n", inputs)
    if cerebralModelOn==1:
        return CVSIM_Brain.solve(scaling, solve_config)
    if cerebralModelOn==0:
        return CVSIM.solve(scaling, solve_config)

# Function to do just 1 test-run with a given set of scalars
def run_solve2():
    inp_ones = np.ones(27)
    inp_ones[-1] = 0.0
    """
    inp_opti = [ 0.88521808, 0.57638988, 1.23869191, 0.63037545, 1.04719004, 1.52528282, 1.38392606,
    1.20657392, 0.81749262, 1.04000896, 0.77240868, 1.38909163, 1.1387788, 1.19498142, 0.36249268,
    3.10236141, 1.00832986, 0.81639036, 1.08779616, 1.0769972, 1.3786385, 1.03288004, 1.14197793,
    1.36268182, 0, 0, 0, 0, 0, 0] """
    inp_opti = np.array([0.5338655458886326, 1.1646889924699022, 1.388136476959439, 1.1029384719878366, 1.849453154224727, 
                       1.7807378378407384, 1.7785534533269085, 0.6892790441322125, 1.5678698036480323, 0.9557837667952847, 
                       1.6294108295378877, 1.3747639496457904, 0.8070552787020846, 1.0912946770062129, 0.21376104865104018, 
                       2.1181284046215705, 0.6634135544351104, 0.7267133061898551, 0.8347913616772134, 0.6333487807405132, 
                       0.838087185428039, 0.8538897393068012, 0.9095615520482808, 1.8975416051109735, 1.627988500432018, 
                       2.4033823036710564, 5.978162588044352, 0.31288780143627126, 7.590963263607815, 6.284070547306135])

    Out_av, Out_wave, Out_solver = solve2(inp_opti)
    global t_solver, y_solver
    global t_mean, MAP, Finap, HR_model, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra, store_P_muscle, tmean_mca, store_V_mca_max, store_V_mca_min, store_P_muscle2, store_E, store_UV, store_TBV, store_impulse, store_finap, store_HR, store_crb_Q_ic, store_crb_mca
    if cerebralModelOn==1:
        t_mean, MAP, Finap, HR_model, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra, store_P_muscle, tmean_mca, store_V_mca_max, store_V_mca_min, store_P_muscle2, store_E, store_UV, store_TBV, store_impulse, store_crb_Q_ic, store_crb_mca = Out_av[0]
    if cerebralModelOn==0:
        t_mean, MAP, Finap, HR_model, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra, store_P_muscle, store_P_muscle2, store_E, store_UV, store_TBV, store_impulse = Out_av[0]
    t, wave_f, alpha_tilt, p_intra, p_muscle, p_grav, Elas = Out_wave[0]
    t_solver, y_solver = Out_solver
    return

#%%
# Do 1 test run   
run_switch = True
if run_switch == True:
    run_solve2()

#%%
# Plot
plot_switch = True
if plot_switch == True:
    if cerebralModelOn==1:
        config = {  "cerebralModelOn": cerebralModelOn,
                    "carotidOn": carotidOn,             #
                    "data_time_10Hz": data_time_10Hz,
                    "data_MAP": data_MAP,
                    "data_HR": data_HR,
                    "time_after_sts": time_after_sts,
                    "resting_time": resting_time,
                    "sts_n": sts_n,
                    "t_mean": t_mean,
                    "t_eval": t_eval,
                    "store_BP_min": store_BP_min,
                    "store_BP_max": store_BP_max,
                    "data_DBP": data_DBP,
                    "data_SBP": data_SBP,
                    "delta_t": delta_t,
                    "HR_list": HR_list,
                    "store_P_muscle2": store_P_muscle2,
                    "store_impulse": store_impulse,
                    "store_P": store_P,
                    "store_UV": store_UV,
                    "store_TBV": store_TBV,
                    "store_E": store_E,
                    "scaling": scaling,
                    "tmean_mca": tmean_mca,             #
                    "store_crb_Q_ic": store_crb_Q_ic,   #
                    "store_crb_mca": store_crb_mca,     #
                    "store_V_mca_max": store_V_mca_max, #
                    "store_V_mca_min": store_V_mca_min, #
                    "y_solver": y_solver,
                    "t_solver": t_solver
                    }
    if cerebralModelOn==0:
        config = {  "cerebralModelOn": cerebralModelOn,
                    "data_time_10Hz": data_time_10Hz,
                    "data_MAP": data_MAP,
                    "data_HR": data_HR,
                    "time_after_sts": time_after_sts,
                    "resting_time": resting_time,
                    "sts_n": sts_n,
                    "t_mean": t_mean,
                    "t_eval": t_eval,
                    "store_BP_min": store_BP_min,
                    "store_BP_max": store_BP_max,
                    "data_DBP": data_DBP,
                    "data_SBP": data_SBP,
                    "delta_t": delta_t,
                    "HR_list": HR_list,
                    "store_P_muscle2": store_P_muscle2,
                    "store_impulse": store_impulse,
                    "store_P": store_P,
                    "store_UV": store_UV,
                    "store_TBV": store_TBV,
                    "store_E": store_E,
                    "scaling": scaling,
                    "y_solver": y_solver,
                    "t_solver": t_solver
                    }

    cvplt(config)





#%%
# Parallel runs
parallel_switch = False
if parallel_switch == True:        

    def run_model_with_params(param_set):
        # Run your model using these parameters
        result = solve2(param_set)
        return result

    inp = np.ones(27)
    inp[24] = 0
    # Modify the 20th index with values from 0.5 to 1.5 in increments of 0.1
    parameter_values = np.arange(0.5, 1.6, 0.1)  # Generates [0.5, 0.6, ..., 1.5]

    # Create a list to hold the modified arrays
    modified_arrays = []

    for value in parameter_values:
        new_array = inp.copy()  # Copy the original array to avoid modifying it
        new_array[20] = value   # Change the value at index 20
        modified_arrays.append(new_array)

    # Output results
    print("Modified Arrays:")
    for arr in modified_arrays:
        print(arr)
    """
    import itertools

    # Define ranges of parameters for the analysis
    param1_values = [1, 2, 3]  # Example values for param1
    param2_values = [0.1, 0.2, 0.3]  # Example values for param2

    # Create a list of parameter combinations
    param_combinations = list(itertools.product(param1_values, param2_values))
    """
    import multiprocessing

    if __name__ == "__main__":
        # Adjust the number of processes based on available cores
        num_cores = multiprocessing.cpu_count()  # Or specify a number
        #num_cores = len(modified_arrays)  # Or specify a number
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(run_model_with_params, modified_arrays)

        
        """
        #IMPORT
        data_time_10Hz = np.squeeze(data['data']['time_10Hz'][0][0])
        data_MAP = np.squeeze(data['data']['map'][0][0])
        """
        plt.figure(figsize=(10, 6), dpi=300)

        # Process results as needed
        for result in results:
    
            Out_av, Out_wave, Out_solver = result
            t_mean, MAP, Finap, HR_model, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra, store_P_muscle, tmean_mca, store_V_mca_max, store_V_mca_min, store_P_muscle2 = Out_av[0][0], Out_av[0][1], Out_av[0][2], Out_av[0][3], Out_av[0][4], Out_av[0][5], Out_av[0][6], Out_av[0][7], Out_av[0][8], Out_av[0][9], Out_av[0][10], Out_av[0][11], Out_av[0][12], Out_av[0][13]
            t, wave_f, alpha_tilt, p_intra, p_muscle, p_grav, Elas = Out_wave[0][0], Out_wave[0][1], Out_wave[0][2], Out_wave[0][3], Out_wave[0][4], Out_wave[0][5], Out_wave[0][6]
            t = Out_wave[0][0]
            t_solver, y_solver = Out_solver
            
            # magic with indices and time for allignment
            if time_after_sts > 100:
                time_after_sts = 100
            dt_data = round(data_time_10Hz[1]-data_time_10Hz[0],1)
            start_index_model = int(resting_time/2/dt_data)
            index_end = int((time_after_sts+resting_time)/dt_data)
            
            index_data_begin = utils.find_index_of_time(data_time_10Hz, round(sts_n,1))-start_index_model
            index_data_end = index_data_begin+index_end-start_index_model
            x_data = data_MAP[index_data_begin:index_data_end]
            
            t_translated = t_mean + sts_n - resting_time
            t_eval_trans = t_eval + sts_n - resting_time
            #t_translated2 = t + sts_n - resting_time
            
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
            
            # Plotting
            plt.plot(window_t_trans, window_new_minP, label=r'$Model_{dia}$', color='navy')
            plt.plot(window_t_trans, window_new_maxP, label=r'$Model_{sys}$', color='darkred')
        plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_DBP[index_data_begin:index_data_end], label=r'$Data_{dia}$', color='cornflowerblue', ls="--")
        plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_SBP[index_data_begin:index_data_end], label=r'$Data_{sys}$', color='lightcoral', ls="--")
        plt.axvline(x=sts_n, color='grey', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
        plt.xlabel('$t$ (s)')
        plt.ylabel(r'$P$ (mmHg)')
        plt.title(r'Plot of systolic and diastolic $P$ vs time')
        plt.legend(loc='upper left')
        plt.grid(True)
        # Save the figure to a file (instead of showing it on Snellius)
        plt.savefig('plot_systolic_diastolic_BP.png', format='png')

        # Optionally, if you still want to display it in your local environment:
        # plt.show()  # Comment this out if you don’t want to display it locally
    print("SUCCES!")

#%%
# Compare 2 model runs
compare_switch = False
if compare_switch == True:
    inputs = [ 0.88521808, 0.57638988, 1.23869191, 0.63037545, 1.04719004, 1.52528282, 1.38392606,
    1.20657392, 0.81749262, 1.04000896, 0.77240868, 1.38909163, 1.1387788, 1.19498142, 0.36249268,
    3.10236141, 1.00832986, 0.81639036, 1.08779616, 1.0769972, 1.3786385, 1.03288004, 1.14197793,
    1.36268182, 0, 0, 0, 0, 0, 0, 1] 

    global grav_switch, sliding_window_size, baro_buffer, crb_buffer_size
    baro_buffer = 4 # int
    crb_buffer_size = 2 # seconds, ORIGINAL
    grav_switch = 1
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
        "Rp_lungs": 1, # Rp lungs
        "limit_R": 1, # limit R
        "limit_UV": 1, # limit UV
        "Vmax": 1, # V max
        "venous_UV": 1, # venous UV
        "Rp_upper": 1, # Rp upper
        "carotid_r_switch": 1, # carotid receptor switch
        # Cerebral model
        "k_v": 1, # Cerebral model: doppler correction factor, correct doppler angle and converts maximal to average velocity
        "G_aut": 1 # Cerebral model: Gain of the autoregulation mechanism related to CBF variations
        }

    print(scaling)

    ### RUN ###
    print("Input parameters are:\n", inputs)

    Out_av, Out_wave, Out_solver = solve(scaling)[:3]
    t_mean, MAP, Finap, HR_model, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra, store_P_muscle, tmean_mca, store_V_mca_max, store_V_mca_min, store_P_muscle2 = Out_av[0][0], Out_av[0][1], Out_av[0][2], Out_av[0][3], Out_av[0][4], Out_av[0][5], Out_av[0][6], Out_av[0][7], Out_av[0][8], Out_av[0][9], Out_av[0][10], Out_av[0][11], Out_av[0][12], Out_av[0][13]
    t, wave_f, alpha_tilt, p_intra, p_muscle, p_grav, Elas = Out_wave[0][0], Out_wave[0][1], Out_wave[0][2], Out_wave[0][3], Out_wave[0][4], Out_wave[0][5], Out_wave[0][6]
    t_solver, y_solver = Out_solver

    # magic with indices and time for allignment
    if time_after_sts > 100:
        time_after_sts = 100
    dt_data = round(data_time_10Hz[1]-data_time_10Hz[0],1)
    start_index_model = int(resting_time/2/dt_data)
    index_end = int((time_after_sts+resting_time)/dt_data)
    index_data_begin = utils.find_index_of_time(data_time_10Hz, round(sts_n,1))-start_index_model
    index_data_end = index_data_begin+index_end-start_index_model

    # INTERPOLATE FINAP #
    # Desired sampling frequency
    desired_frequency = 10  # Hz
    desired_interval = 1 / desired_frequency  # Interval in seconds

    # Sys and Dia
    new_times, new_minP = utils.lin_interp(t_mean, store_BP_min, desired_interval)
    new_times, new_maxP = utils.lin_interp(t_mean, store_BP_max, desired_interval)
    lip_HR1 = utils.lin_interp(t_mean, HR_list, desired_interval)[1][start_index_model:index_end]
    t_translated_new = (new_times + sts_n - resting_time)
    window_t_trans1 = t_translated_new[start_index_model:index_end]
    window_new_minP1 = new_minP[start_index_model:index_end]
    window_new_maxP1 = new_maxP[start_index_model:index_end]

    inputs = [ 0.88521808, 0.57638988, 1.23869191, 0.63037545, 1.04719004, 1.52528282, 1.38392606,
    1.20657392, 0.81749262, 1.04000896, 0.77240868, 1.38909163, 1.1387788, 1.19498142, 0.36249268,
    3.10236141, 1.00832986, 0.81639036, 1.08779616, 1.0769972, 1.3786385, 1.03288004, 1.14197793,
    1.36268182, 0, 0, 0, 0, 0, 0, 1] 

    grav_switch = 1
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
        "Rp_lungs": 1, # Rp lungs
        "limit_R": 1, # limit R
        "limit_UV": 1, # limit UV
        "Vmax": 1, # V max
        "venous_UV": 1, # venous UV
        "Rp_upper": 1, # Rp upper
        "carotid_r_switch": 1, # carotid receptor switch
        # Cerebral model
        "k_v": 1, # Cerebral model: doppler correction factor, correct doppler angle and converts maximal to average velocity
        "G_aut": 1 # Cerebral model: Gain of the autoregulation mechanism related to CBF variations
        }

    print(scaling)

    ### RUN ###
    print("Input parameters are:\n", inputs)

    Out_av, Out_wave, Out_solver = solve(scaling)[:3]
    t_mean, MAP, Finap, HR_model, store_BP_max, store_BP_min, HR_list, store_P, store_P_intra, store_P_muscle, tmean_mca, store_V_mca_max, store_V_mca_min, store_P_muscle2 = Out_av[0][0], Out_av[0][1], Out_av[0][2], Out_av[0][3], Out_av[0][4], Out_av[0][5], Out_av[0][6], Out_av[0][7], Out_av[0][8], Out_av[0][9], Out_av[0][10], Out_av[0][11], Out_av[0][12], Out_av[0][13]
    t, wave_f, alpha_tilt, p_intra, p_muscle, p_grav, Elas = Out_wave[0][0], Out_wave[0][1], Out_wave[0][2], Out_wave[0][3], Out_wave[0][4], Out_wave[0][5], Out_wave[0][6]
    t_solver, y_solver = Out_solver

    # Sys and Dia
    new_times, new_minP = utils.lin_interp(t_mean, store_BP_min, desired_interval)
    new_times, new_maxP = utils.lin_interp(t_mean, store_BP_max, desired_interval)
    lip_HR2 = utils.lin_interp(t_mean, HR_list, desired_interval)[1][start_index_model:index_end]
    t_translated_new = (new_times + sts_n - resting_time)
    window_t_trans2 = t_translated_new[start_index_model:index_end]
    window_new_minP2 = new_minP[start_index_model:index_end]
    window_new_maxP2 = new_maxP[start_index_model:index_end]
    
    # Plotting
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(window_t_trans1, window_new_minP1, label=r'$Model1_{dia}$', color='navy')
    plt.plot(window_t_trans1, window_new_maxP1, label=r'$Model1_{sys}$', color='darkred')
    plt.plot(window_t_trans2, window_new_minP2, label=r'$Model2_{dia}$', color='cornflowerblue', ls="--")
    plt.plot(window_t_trans2, window_new_maxP2, label=r'$Model2_{sys}$', color='lightcoral', ls="--")
    plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_DBP[index_data_begin:index_data_end], label=r'$Data_{dia}$', color='blue', ls="-.")
    plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_SBP[index_data_begin:index_data_end], label=r'$Data_{sys}$', color='red', ls="-.")
    plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
    plt.axvline(x=sts_n+delta_t, color='grey', linestyle='--', label=r'$t_0 + \Delta t$')  # Add vertical line at UV
    plt.xlabel('$t$ (s)')
    plt.ylabel(r'$P$ (mmHg)')
    plt.title(r'Plot of systolic and diastolic $P$ vs time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    # HR 1
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(window_t_trans1, lip_HR1, label=r'$HR1_{model}$', color='darkred')
    plt.plot(window_t_trans2, lip_HR2, label=r'$HR2_{model}$', color='lightcoral', linestyle="--")
    plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_HR[index_data_begin:index_data_end], label=r'$HR_{data}$', color='red', ls="-.")
    plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
    #plt.axvline(x=sts_n+delta_t, color='grey', linestyle='--', label=r'$t_0 + \Delta t$')  # Add vertical line at UV
    plt.xlabel('$t$ (s)')
    plt.ylabel(r'$HR$ ($min^{-1}$)')
    plt.title(r'Plot of heart rate ($HR$) vs time ($t$)')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

#%%
# Compare addition of carotid
"""
carotidOn = 1
run_solve2()

t_trans_mca = tmean_mca + sts_n - resting_time
store_V_mca_max_1 = store_V_mca_max[50:200]
store_V_mca_min_1 = store_V_mca_min[50:200]
t_trans_mca_1 = t_trans_mca[50:200]

carotidOn = 0
run_solve2()

t_trans_mca = tmean_mca + sts_n - resting_time
store_V_mca_max_2 = store_V_mca_max[50:200]
store_V_mca_min_2 = store_V_mca_min[50:200]
t_trans_mca_2 = t_trans_mca[50:200]

# PLOT
# Comparing two model runs for MCA velocity
plt.figure(figsize=(10, 6), dpi=300)

# First model run
plt.plot(t_trans_mca_1, store_V_mca_max_1, label=r'$V_{MCA_{max, WITH}}$', color='lightcoral', linestyle='--')
plt.plot(t_trans_mca_1, store_V_mca_min_1, label=r'$V_{MCA_{min, WITH}}$', color='cornflowerblue', linestyle='--')

# Second model run
plt.plot(t_trans_mca_2, store_V_mca_max_2, label=r'$V_{MCA_{max, W/O}}$', color='darkred', linestyle='-')
plt.plot(t_trans_mca_2, store_V_mca_min_2, label=r'$V_{MCA_{min, W/O}}$', color='navy', linestyle='-')

# Add vertical line (common to both runs)
plt.axvline(x=sts_n, color='grey', linestyle='--', label=r'$t_0$') 

# Labels and title
plt.xlabel('t (s)')
plt.ylabel(r'$V$ ($cm \times s^{-1}$)')
plt.title(r'Comparison of middle cerebral artery velocity ($V_{MCA}$) vs time with and without (W/O) carotid')

# Legend and grid
plt.legend(loc='upper left')
plt.grid(True)

# Show the plot
plt.show()
"""


#%%
# SA (Sensitivity Analysis) test, with fewer parameters
SA = 0
if SA == 1:

    # bounds
    n_opti_var = 2
    bounds = np.ones([n_opti_var,2])
    l_bound = 0.5
    u_bound = 1/l_bound

    bounds[:,0] = bounds[:,0] * l_bound
    bounds[:,1] = bounds[:,1] * u_bound

    # SA
    def wrapped_solver(X, func=solve2_test):
        global buffer_front, buffer_back
        # Desired sampling frequency
        desired_frequency = 10  # Hz
        desired_interval = 1 / desired_frequency  # Interval in seconds
        buffer_front = 1 # s
        buffer_back = 3 # s
        N, D = X.shape
        print("Size N = ", np.size(N))
        print("Model inputs D = ", np.size(D))
        results = np.zeros((X.shape[0], int(np.size(t_eval)/desired_frequency-(buffer_back+1)*desired_frequency)))
        print(np.size(results))
        for i in range(N):
            X_row = X[i, :]
            output = func(X_row)[0][0]
            t_mean = output[0]
            output_Pmax = output[4]
            output_Pmin = output[5]
            output_HR = output[6]
            # Linear Interpolation
            new_times, new_minP = utils.lin_interp(t_mean, output_Pmin, desired_interval)
            new_times, new_maxP = utils.lin_interp(t_mean, output_Pmax, desired_interval)
            index_start = utils.find_index_of_time(new_times, buffer_front, tol=1e-5)
            index_end = utils.find_index_of_time(new_times, tmax-buffer_back, tol=1e-5)
            results[i] = new_maxP[index_start:index_end]
            print(str(i) + " of 320")

        return results

    # Sensitivity analysis (SA)
    PS = ProblemSpec({
        'num_vars': n_opti_var,
        'names': ['RRsgain (HP)', 'RRpgain (HP)'],
        'bounds': bounds,
    })

    fac = 1 # Samples = N*(2D+2), N=2**fac, D=model inputs=4
    (
        PS.sample_sobol(2**fac)
        .evaluate(wrapped_solver)
        .analyze_sobol(print_to_console=True)
    )
    

    # Get first order sensitivities for all outputs
    S1s = np.array([PS.analysis[_y]['S1'] for _y in PS['outputs']])

    # Get model outputs
    y = PS.results

    # Set up figure
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    buffer_front = 1 # s
    buffer_back = 3 # s
    tmeans = np.linspace(buffer_front, tmax-buffer_back, int((tmax - buffer_back - buffer_front)*10))

    # Populate figure subplots
    for i, ax in enumerate([ax0, ax1, ax2, ax3]):
        ax.plot(tmeans, S1s[:, i],
                label=r'S1$_\mathregular{{{}}}$'.format(PS["names"][i]),
                color='black')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("First-order Sobol index")

        ax.set_ylim(-0.1, 1.1)

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.legend(loc='upper right')

    plt.show()

    # Get total sensitivities for all outputs
    STs = np.array([PS.analysis[_y]['ST'] for _y in PS['outputs']])

    # Set up figure
    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # Populate figure subplots
    for i, ax in enumerate([ax0, ax1, ax2, ax3]):
        ax.plot(tmeans, STs[:, i],
                label=r'ST$_\mathregular{{{}}}$'.format(PS["names"][i]),
                color='black')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Total Sobol index")

        ax.set_ylim(-0.1, 1.1)

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.legend(loc='upper right')

    plt.show()
    
    # Get second order sensitivities for all outputs
    S2s = np.array([PS.analysis[_y]['S2'] for _y in PS['outputs']])

    # Set up figure
    fig1 = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig1.add_gridspec(2, 3)

    ax0 = fig1.add_subplot(gs[0, 0])
    ax1 = fig1.add_subplot(gs[0, 1])
    ax2 = fig1.add_subplot(gs[1, 0])
    ax3 = fig1.add_subplot(gs[1, 1])
    ax4 = fig1.add_subplot(gs[0, 2])
    ax5 = fig1.add_subplot(gs[1, 2])

    j=0
    k=1

    # Populate figure subplots
    for i, ax in enumerate([ax0, ax1, ax2, ax3, ax4, ax5]):
        print(str(j),str(i+k))
        ax.plot(tmeans, S2s[:, j, i+k],
                label=r'S2$_\mathregular{{{}}}$'.format(PS["names"][i+k])+'$_\mathregular{{{}}}$'.format(PS["names"][j]),
                color='black')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("second-order Sobol index")

        ax.set_ylim(-1.1, 1.1)

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.legend(loc='upper right')
        
        if i+k==3:
            k=k-2+j
            j=j+1
        
    plt.show()
    
    # Plotting the results    
    plt.plot(tmeans, np.mean(y, axis=0), label="Mean", color='black')

    # in percent
    prediction_interval = 95

    plt.fill_between(tmeans,
                    np.percentile(y, 50 - prediction_interval/2., axis=0),
                    np.percentile(y, 50 + prediction_interval/2., axis=0),
                    alpha=0.5, color='black',
                    label=f"{prediction_interval} % prediction interval")

    plt.xlabel("Time (s)")
    plt.ylabel("MAP (mmHg)")
    plt.legend(title=r"MAP",
            loc='lower right')._legend_box.align = "left"

    plt.show()


    # SA with all parameters
    # bounds
    n_opti_var = 23
    bounds = np.ones([n_opti_var,2])
    l_bound = 0.5
    u_bound = 1/l_bound

    bound = 9
    bounds[0:bound,0] = bounds[0:bound,0] * l_bound
    bounds[0:bound,1] = bounds[0:bound,1] * u_bound

    l_bound1 = 0.7
    u_bound1 = 1/l_bound1
    bounds[bound:,0] = bounds[bound:,0] * l_bound1
    bounds[bound:,1] = bounds[bound:,1] * u_bound1

    bounds[9,0] = 0.8
    bounds[9,1] = 1/0.8
    bounds[12,0] = 0.8
    bounds[12,1] = 1/0.8

    bounds[16,0] = 0.2
    bounds[16,1] = 1/0.2
    bounds[17,0] = 0.2
    bounds[17,1] = 1/0.2

    # SA
    def wrapped_solver(X, func=solve2):
        
        # Desired sampling frequency
        desired_frequency = 10  # Hz
        desired_interval = 1 / desired_frequency  # Interval in seconds
        buffer_front = 1 # s
        buffer_back = 3 # s
        N, D = X.shape
        print("Size N = ", np.size(N))
        print("Model inputs D = ", np.size(D))
        results = np.zeros((X.shape[0], int(np.size(t_eval)/desired_frequency-(buffer_back+1)*desired_frequency)))
        print(np.size(results))
        for i in range(N):
            X_row = X[i, :]
            output = func(X_row)[0][0]
            t_mean = output[0]
            output_Pmax = output[4]
            output_Pmin = output[5]
            output_HR = output[6]
            # Linear Interpolation
            new_times, new_minP = utils.lin_interp(t_mean, output_Pmin, desired_interval)
            new_times, new_maxP = utils.lin_interp(t_mean, output_Pmax, desired_interval)
            index_start = utils.find_index_of_time(new_times, buffer_front, tol=1e-5)
            index_end = utils.find_index_of_time(new_times, tmax-buffer_back, tol=1e-5)
            results[i] = new_maxP[index_start:index_end]
            print(str(i) + " of 320")

        return results

    # Sensitivity analysis (SA)
    PS = ProblemSpec({
        'num_vars': n_opti_var,
        'names': ['RRsgain (HP)', 'RRpgain (HP)', 'beta (UV)', 'alpha (UV)', 'beta_resp (ErvMAX)', 'beta_resp (ElvMAX)', 'alpha_resp (R)', 'alphav_resp (R)', 'RAP_setp', 
                        'ABP_setp', 'max_heart_E', 'min_heart_E', 'sys-dias_ratio', 'HR_t0', 'STS_P', 'Stand_P', 'grav_strong', 'grav_weak', 'global_R', 'global_UV', 
                        'R_p', 'E_arteries', 'E_veins'],
        'bounds': bounds,
    })

    fac = 1 # Samples = N*(2D+2), N=2**fac, D=model inputs=4

    (
        PS.sample_sobol(2**fac)
        .evaluate(wrapped_solver)
        .analyze_sobol(print_to_console=True)
    )

    PS.to_df()

