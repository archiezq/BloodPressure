"""
Created on 11-9-2024
@author: joerivandesande@live.nl
"""


#%% IMPORTS
from scipy.io import loadmat
import numpy as np
import CVSIM_Brain as model
import matplotlib.pyplot as plt # Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
import math
from scipy.optimize import least_squares
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from pymoo.core.evaluator import Evaluator
from scipy.interpolate import interp1d
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import multiprocessing
from pymoo.core.problem import StarmapParallelization


#%% CLASSES AND FUNCTIONS
class MyProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=n_variables,
                         n_obj=n_objectives,
                         xl=xl,
                         xu=xu) # Look into constraint handling for dependent parameters

    def _evaluate(self, model_input, out, *args, **kwargs):
        global iteration
        print("Iteration:" , iteration, "/", iter_tot)
        iteration = iteration+1
        Out_av, Out_wave, Out_solver = model.solve2(model_input)
        t_mean, BP_max, BP_min = Out_av[0][0], Out_av[0][4], Out_av[0][5]

        lip_minP, lip_minP_t = lin_interp(t_mean, BP_min, desired_interval)
        lip_minP = lip_minP[find_index_of_time(lip_minP_t, model_start_t):
        find_index_of_time(lip_minP_t, model_end_t)+1]
        lip_maxP, lip_maxP_t = lin_interp(t_mean, BP_max, desired_interval)
        lip_maxP = lip_maxP[find_index_of_time(lip_maxP_t, model_start_t):
        find_index_of_time(lip_maxP_t, model_end_t)+1]
                
        rmse_av_p = 0.5*(rmse_norm(x_data_max, lip_maxP)+rmse_norm(x_data_min, lip_minP))
        print("RMSE =", rmse_av_p)
        
        opti_hist.append([rmse_av_p, model_input])

        out["F"] = rmse_av_p

class MyProblem_HR(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=n_variables,
                         n_obj=n_objectives,
                         xl=xl,
                         xu=xu) # Look into constraint handling for dependent parameters

    def _evaluate(self, model_input, out, *args, **kwargs):
        global iteration
        print("Iteration:" , iteration, "/", iter_tot)
        iteration = iteration+1
        Out_av, Out_wave, Out_solver = model.solve2(model_input)
        t_mean, BP_max, BP_min, HR_list = Out_av[0][0], Out_av[0][4], Out_av[0][5], Out_av[0][6]

        lip_minP_t, lip_minP = lin_interp(t_mean, BP_min, desired_interval)
        lip_minP = lip_minP[find_index_of_time(lip_minP_t, model_start_t):
        find_index_of_time(lip_minP_t, model_end_t)+1]
        lip_maxP_t, lip_maxP = lin_interp(t_mean, BP_max, desired_interval)
        lip_maxP = lip_maxP[find_index_of_time(lip_maxP_t, model_start_t):
        find_index_of_time(lip_maxP_t, model_end_t)+1]
        lip_HR_t, lip_HR = lin_interp(t_mean, HR_list, desired_interval)
        lip_HR = lip_HR[find_index_of_time(lip_HR_t, model_start_t):
        find_index_of_time(lip_HR_t, model_end_t)+1]
        """
        print(lip_HR_t[0], lip_HR_t[-1])
        print(lip_minP_t[0], lip_minP_t[-1])
        print(lip_maxP_t[0], lip_maxP_t[-1])
        print(lip_HR_t[find_index_of_time(lip_HR_t, model_start_t):
        find_index_of_time(lip_HR_t, model_end_t)])
        
        print(lip_minP_t[start_index_model], lip_minP_t[index_end])
        print(lip_maxP_t[start_index_model], lip_maxP_t[index_end])
        print(lip_HR_t[start_index_model], lip_HR_t[index_end])
        """       
        rmse_av = (1/3) * (rmse_norm(x_data_max, lip_maxP) + rmse_norm(x_data_min, lip_minP) + rmse_norm(x_data_HR, lip_HR))
        print("RMSE =", rmse_av)
        
        opti_hist.append([rmse_av, model_input])

        out["F"] = rmse_av

class MyProblem_HR_parallel(ElementwiseProblem):
    def __init__(self, elementwise_runner=None):
        super().__init__(
            n_var=n_variables,
            n_obj=n_objectives,
            xl=xl,
            xu=xu,
            elementwise_runner=elementwise_runner,
        )
    
    def _evaluate(self, model_input, out, *args, **kwargs):
        global iteration
        print(f"Iteration: {iteration}/{iter_tot}")
        iteration += 1

        # Run your model and calculate RMSE
        rmse_av = run_model_and_calculate_rmse(model_input)
        opti_hist.append([rmse_av, model_input])

        out["F"] = rmse_av

def run_model_and_calculate_rmse(model_input):
    Out_av, Out_wave, Out_solver = model.solve2(model_input)
    t_mean, BP_max, BP_min, HR_list = Out_av[0][0], Out_av[0][4], Out_av[0][5], Out_av[0][6]

    lip_minP_t, lip_minP = lin_interp(t_mean, BP_min, desired_interval)
    lip_minP = lip_minP[find_index_of_time(lip_minP_t, model_start_t):
    find_index_of_time(lip_minP_t, model_end_t)+1]
    lip_maxP_t, lip_maxP = lin_interp(t_mean, BP_max, desired_interval)
    lip_maxP = lip_maxP[find_index_of_time(lip_maxP_t, model_start_t):
    find_index_of_time(lip_maxP_t, model_end_t)+1]
    lip_HR_t, lip_HR = lin_interp(t_mean, HR_list, desired_interval)
    lip_HR = lip_HR[find_index_of_time(lip_HR_t, model_start_t):
    find_index_of_time(lip_HR_t, model_end_t)+1]

    rmse_av = (1/3) * (rmse_norm(x_data_max, lip_maxP) + rmse_norm(x_data_min, lip_minP) + rmse_norm(x_data_HR, lip_HR))
    print("RMSE =", rmse_av)
    return rmse_av

class MyProblem_BRAIN(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=n_variables,
                         n_obj=n_objectives,
                         xl=xl,
                         xu=xu) # Look into constraint handling for dependent parameters

    def _evaluate(self, model_input, out, *args, **kwargs):
        global iteration
        print("Iteration:" , iteration, "/", iter_tot)
        iteration = iteration+1
        Out_av, Out_wave, Out_solver = model.solve2(model_input)
        print("size of Out_av", len(Out_av[0]))
        t_mean, BP_max, BP_min, HR_list, mca_t, mca_max, mca_min = Out_av[0][0], Out_av[0][4], Out_av[0][5], Out_av[0][6], Out_av[0][10], Out_av[0][11], Out_av[0][12]

        lip_minP_t, lip_minP = lin_interp(t_mean, BP_min, desired_interval)
        lip_minP = lip_minP[find_index_of_time(lip_minP_t, model_start_t):
        find_index_of_time(lip_minP_t, model_end_t)+1]
        lip_maxP_t, lip_maxP = lin_interp(t_mean, BP_max, desired_interval)
        lip_maxP = lip_maxP[find_index_of_time(lip_maxP_t, model_start_t):
        find_index_of_time(lip_maxP_t, model_end_t)+1]
        lip_HR_t, lip_HR = lin_interp(t_mean, HR_list, desired_interval)
        lip_HR = lip_HR[find_index_of_time(lip_HR_t, model_start_t):
        find_index_of_time(lip_HR_t, model_end_t)+1]
    
        lip_min_mca_t, lip_min_mca = lin_interp(mca_t, mca_min, desired_interval)
        print(lip_min_mca_t)
        lip_min_mca = lip_min_mca[find_index_of_time(lip_min_mca_t, model_start_t):
        find_index_of_time(lip_min_mca, model_end_t)+1]
        lip_max_mca_t, lip_max_mca = lin_interp(mca_t, mca_max, desired_interval)
        print(lip_min_mca_t)
        lip_max_mca = lip_max_mca[find_index_of_time(lip_max_mca_t, model_start_t):
        find_index_of_time(lip_max_mca_t, model_end_t)+1]
                
        rmse_av = (1/5) * (rmse_norm(x_data_max, lip_maxP) + rmse_norm(x_data_min, lip_minP) + rmse_norm(x_data_HR, lip_HR) +
        rmse_norm(av_min_v, lip_min_mca) + rmse_norm(av_max_v, lip_max_mca))
        print("RMSE =", rmse_av)
        
        opti_hist.append([rmse_av, model_input])

        out["F"] = rmse_av

def model_function(model_input):
    Out_av, Out_wave, Out_solver = model.solve2(model_input)
    t_mean, Finap, BP_max, BP_min = Out_av[0][0], Out_av[0][2], Out_av[0][4], Out_av[0][5]
    
    lip_times, new_values = lin_interp(t_mean, Finap, desired_interval)
    
    residual = new_values[start_index_model:index_end]-x_data
    rmse_av_p = rmse_norm(x_data, new_values[start_index_model:index_end])
    print("RMSE =", rmse_av_p)

    return residual

def model_function2(model_input):
    Out_av, Out_wave, Out_solver = model.solve2(model_input)
    t_mean, BP_max, BP_min = Out_av[0][0], Out_av[0][4], Out_av[0][5]
    
    lip_minP = lin_interp(t_mean, BP_min, desired_interval)[1][start_index_model:index_end]
    lip_maxP = lin_interp(t_mean, BP_max, desired_interval)[1][start_index_model:index_end]
    
    residual1 = lip_minP-x_data_min
    residual2 = lip_maxP-x_data_max

    rmse_av_p = 0.5*(rmse_norm(x_data_min, lip_minP)+rmse_norm(x_data_max, lip_maxP))
    print("RMSE =", rmse_av_p)

    return residual1+residual2

def find_index_of_time(time_array, target_time, tol=1e-2):
    indices = np.where(np.isclose(time_array, target_time, atol=tol))[0]  # Get indices where target_time occurs within the tolerance
    if indices.size > 0:  # Check if indices array is not empty
        return indices[0]  # Return the first index where the target_time occurs
    else:
        print(f"Time value {target_time} not found in the array.")
        return None
    
def find_index_of_lowest_value(elements):
    if not elements:
        return -1  # Return -1 if the list is empty
    
    min_index = 0
    min_value = elements[0][0]
    
    for i in range(1, len(elements)):
        if elements[i][0] < min_value:
            min_value = elements[i][0]
            min_index = i
    
    return min_index     

def find_indices_of_nan_and_inf(elements):
    nan_inf_indices = []
    
    for i, element in enumerate(elements):
        number = element[0]
        if math.isnan(number) or math.isinf(number):
            nan_inf_indices.append(i)
    
    return nan_inf_indices

def plot_arrays(elements, indices):
    plt.figure(figsize=(10, 6))
    
    for i in indices:
        array = elements[i][1]
        plt.plot(array, label=f'Index {i}')
    
    plt.xlabel('Index in array')
    plt.ylabel('Value')
    plt.title('Arrays with NaN or inf in their first index')
    plt.legend()
    plt.show()
    
def plot_scatter(elements, indices):
    plt.figure(figsize=(10, 6))
    
    for i in indices:
        array = elements[i][1]
        plt.scatter(range(len(array)), array, label=f'NaN {i}')
    
    plt.xlabel('Index in array')
    plt.ylabel('Value')
    plt.title('NaN or Inf RMSE parameter values scatter')
    plt.legend()
    plt.show()
    
def plot_scatter_opti(elements, indices, optimal, variable_names):
    plt.figure(figsize=(10, 6))
    
    for i in indices:
        array = elements[i][1]
        plt.scatter(range(len(array)), array, label=f'NaN {i}')
    
    plt.scatter(range(len(array)), optimal, label='Optimum', marker='x')
    
    plt.xticks(range(len(variable_names)), variable_names, rotation=45, ha='right')
    plt.xlabel('Index in array')
    plt.ylabel('Value')
    plt.title('NaN or Inf RMSE parameter values scatter')
    plt.legend()
    plt.tight_layout()
    plt.show()

def rmse(array1, array2):
    # Ensure both arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")

    # Calculate the squared differences between corresponding elements of the arrays
    squared_diff = (array1 - array2) ** 2

    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Calculate the square root of the mean squared difference
    rmse1 = np.sqrt(mean_squared_diff)

    return rmse1

def rmse_norm(array1, array2):
    # Ensure both arrays have the same length
    if len(array1) != len(array2):
        print(len(array1), len(array2))
        raise ValueError("Arrays must have the same length")
        
    # Normalize
    array1_norm, array2_norm = normalize(array1, array2)

    # Calculate the squared differences between corresponding elements of the arrays
    squared_diff = (array1_norm - array2_norm) ** 2

    # Calculate the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Calculate the square root of the mean squared difference
    rmse_n = np.sqrt(mean_squared_diff)

    return rmse_n

def rmse_time(array1, array2):
    # Ensure both arrays have the same length
    if len(array1) != len(array2):
        print(len(array1), len(array2))
        raise ValueError("Arrays must have the same length")
        
    # Normalize
    array1_norm, array2_norm = normalize(array1, array2)

    # Calculate the squared differences between corresponding elements of the arrays
    squared_diff = (array1_norm - array2_norm) ** 2

    # Calculate the square root of the mean squared difference
    rmse_t = np.sqrt(squared_diff)

    return rmse_t

def rmse_weighted2(array1, array2):
    # Ensure both arrays have the same length
    if len(array1) != len(array2):
        print(len(array1), len(array2))
        raise ValueError("Arrays must have the same length")
    
    # Determine the length and halfway point of the arrays
    length = len(array1)
    
    # Normalize
    array1_norm, array2_norm = normalize(array1, array2)

    # Calculate the squared differences between corresponding elements of the arrays
    squared_diff = (array1_norm - array2_norm) ** 2

    weights = np.ones(length)
    weight_start = int(sts_model/dt_data-start_index_model)
    weigth_end = int(weight_start+60/dt_data) # Classical OH is defined as a reduction of BP within 3 min of STS. -> Change later.
    weights[weight_start:weigth_end] = 2

    # Calculate the weighted mean of the squared differences
    weighted_mean_squared_diff = np.average(squared_diff, weights=weights)

    # Calculate the square root of the weighted mean squared difference
    rmse_w = np.sqrt(weighted_mean_squared_diff)

    return rmse_w

def lin_interp(t_translated, values, desired_interval):
    # Create new time array with the desired frequency, starting from the first whole number in the range
    start_time = np.ceil(t_translated[0] * 10) / 10  # Adjusting to start from the next whole 0.1 increment
    end_time = np.floor((t_translated[-1] - desired_interval) * 10) / 10  # Adjusting to stay within bounds
    lip_times = np.arange(start_time, end_time + desired_interval, desired_interval)

    # Perform linear interpolation
    linear_interp = interp1d(t_translated, values, kind='linear')
    new_values = linear_interp(lip_times)
    return lip_times, new_values

def lin_interp_index(t_translated, values, desired_interval, index_begin, index_end):
    # Create new time array with the desired frequency, starting from the first whole number in the range
    start_time = np.ceil(t_translated[0] * 10) / 10  # Adjusting to start from the next whole 0.1 increment
    end_time = np.floor((t_translated[-1] - desired_interval) * 10) / 10  # Adjusting to stay within bounds
    print(start_time, end_time)
    lip_times = np.arange(start_time, end_time + desired_interval, desired_interval)

    # Perform linear interpolation
    linear_interp = interp1d(t_translated, values, kind='linear')
    new_values = linear_interp(lip_times)
    return lip_times[index_begin:index_end], new_values[index_begin:index_end]

def normalize(arr1, arr2):
    # Normalize model results and data based on max and min in data (arr1)
    arr_min = arr1.min()
    arr_max = arr1.max()
    norm1 = (arr1 - arr_min) / (arr_max - arr_min)
    norm2 = (arr2 - arr_min) / (arr_max - arr_min)
    return norm1, norm2

def normalize_param(arr, xu, xl):
    # Normalize optimal parameters based on xu and xl (upper and lower bounds)
    norm = (arr - xl) / (xu - xl)
    return norm

def write_results(X, F):

    # Assuming res.X is a list or array of optimal parameters
    # and res.F is the normalized RMSE

    optimal_params = X
    normalized_rmse = F

    # Specify the file name
    output_file = 'optimization_results.txt'

    # Open the file in write mode and write the values
    with open(output_file, 'w') as f:
        f.write('Optimal Parameters:\n')
        for i, param in enumerate(optimal_params):
            f.write(f'Parameter {i + 1}: {param}\n')
        
        f.write('\nNormalized RMSE:\n')
        f.write(f'{normalized_rmse}\n')

    print(f'Results saved to {output_file}')

def average_velocity_on_matching_times(time_list1, vel_list1, time_list2, vel_list2):
    # Convert the time-value pairs into dictionaries for fast lookup
    dict1 = {t: v for t, v in zip(time_list1, vel_list1)}
    dict2 = {t: v for t, v in zip(time_list2, vel_list2)}
    
    # Find the common time points between the two lists
    common_times = sorted(set(time_list1) & set(time_list2))
    
    # Initialize a list to store the averaged velocity values
    averaged_velocities = []
    
    # For each common time, average the velocity values from both lists
    for t in common_times:
        avg_velocity = (dict1[t] + dict2[t]) / 2
        averaged_velocities.append(avg_velocity)
    
    return common_times, averaged_velocities


#%% PARAMETERS AND THEIR OPTIMIZATION RANGES

global n_variables, n_objectives, sliding_window_size, desired_interval, opti_hist, iteration, iter_tot
n_variables = 31
n_objectives = 1

sliding_window_size = 1 # periods
sts_model = 120 # (s)

# Desired sampling frequency
desired_frequency = 10  # Hz
desired_interval = 1 / desired_frequency  # Interval in seconds

# Global search parameters
offspring = 20
iterations = 20
iter_tot = (offspring-1)*iterations
seed = 1

# Init
opti_hist = []
iteration = 0

### INPUTS ###
global xl, xu # lower and upper bound
inputs = np.ones([n_variables,2])
l_bound = 0.2
u_bound = 2
"""
inputs[:,0] = inputs[:,0] * l_bound
inputs[:,1] = inputs[:,1] * u_bound
"""
bound = 8
inputs[0:bound,0] = inputs[0:bound,0] * l_bound
inputs[0:bound,1] = inputs[0:bound,1] * u_bound

l_bound1 = 0.7
u_bound1 = 1/l_bound1
inputs[bound:,0] = inputs[bound:,0] * l_bound1
inputs[bound:,1] = inputs[bound:,1] * u_bound1
# ABP_setp -> is first set though data averaging, then modified by the optimizer
inputs[9,0] = 0.8
inputs[9,1] = 1/0.8
# systole-diastole ratio scaling
inputs[12,0] = 0.8
inputs[12,1] = 1/0.8
# v_ratio scaling
inputs[21,0] = 0.8
inputs[21,1] = 1/0.8
# STS pressure and muscle pressure after STS
inputs[14,0] = 0.2
inputs[14,1] = 1/0.2
inputs[15,0] = 0.2
inputs[15,1] = 1/0.2
# Baroreflex delay
inputs[24,0] = 0
inputs[24,1] = 5

xl = np.array(inputs[:,0]) # lower bounds
xu = np.array(inputs[:,1]) # upper bounds

# copy from CVSIM.py and adjust variable_names accordingly: in order of inputs[x]
scaling_factors = {
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
    "STS_muscleP": inputs[15], # muscle P after STS (artificial value)
    "Global_R": inputs[16], # Global R
    "Global_UV": inputs[17], # Global UV
    "Rp": inputs[18]*1.4, # Rp. age regression 1.1%/year = 1.0111^(70-40) = 1.4, LANDOWNE 1955
    "E_arteries": inputs[19]*1.6, # E arteries. age regression 1.58%/year = 1.0158^(70-40) = 1.6, LANDOWNE 1955
    "E_veins": inputs[20]*1.6, # E veins. age regression 1.58%/year = 1.0158^(70-40) = 1.6, LANDOWNE 1955
    "v_ratio": inputs[21], # v_ratio scaling
    "baro_delay_para": inputs[24], # parasympathetic baro_delay
    "baro_delay_sympa": inputs[25], # sympathetic baro_delay
    "baro_delay_BR_R": inputs[26], # baro_delay BR R
    "baro_delay_BR_UV": inputs[27], # baro_delay BR UV
    "baro_delay_CP_R": inputs[28], # baro_delay CP_R
    "baro_delay_CP_UV": inputs[29] # baro_delay CP_UV
    }

variable_names = ['RRsgain (HP)', 'RRpgain (HP)', 'beta (UV)', 'alpha (UV)', 'beta_resp (ErvMAX)', 'beta_resp (ElvMAX)', 'alpha_resp (R)', 'alphav_resp (R)', 'RAP_setp', 
                  'ABP_setp', 'max_RV_E', 'min_RV_E', 'sys-dias_ratio', 'HR_t0', 'STS_pressure', 'STS_muscleP', 'grav_strong', 'global_R', 'global_UV', 
                  'R_p', 'E_arteries', 'E_veins', 'v_ratio', 'max_LV_E', 'min_LV_E', "baro_delay_para", "baro_delay_sympa", "baro_delay_BR_R",
                  "baro_delay_BR_UV", "baro_delay_CP_R", "baro_delay_CP_UV"]


#%% IMPORT

snellius = False # 
dataset = 2 # 0=NIVLAD, 1=PROHEALTH, 2=PROHEALTH_AVERAGE
standup_n = 1 # n stand-up
include_doppler = False # 0=No, 1=Yes

fallers = 0

if snellius == False:
    if dataset == 0:
        filename  = r"C:\Users\jsande\Documents\UvA\Data\AMC_OneDrive_4-5-2024\Data_NILVAD\NILVAD_preprocessed_data\Preprocessed_PIN43028_T2_SSS.mat"
    if dataset == 1:
        filename = r"C:\Users\jsande\OneDrive - UvA\Documenten\UvA\Data\AMC_OneDrive_4-5-2024\Data validation study PROHEALTH\Preprocessed data\PHI022\Preprocessed_PHI022_FSup.mat"
    if dataset == 2:
        csv_SBP = r"D:\CLS\Thesis\Qi_CODE\mean_data_fallers_non_fallers_SBP.csv"
        csv_DBP = r"D:\CLS\Thesis\Qi_CODE\mean_data_fallers_non_fallers_DBP.csv"
        csv_HR = r"D:\CLS\Thesis\Qi_CODE\mean_data_fallers_non_fallers_HR.csv"

if snellius == True:
    # Paths for when running on Snellius
    csv_SBP = "/gpfs/home1/jvandesande/Preprocessed_data/Populations/mean_data_fallers_non_fallers_SBP.csv"
    csv_DBP = "/gpfs/home1/jvandesande/Preprocessed_data/Populations/mean_data_fallers_non_fallers_DBP.csv"
    csv_HR = "/gpfs/home1/jvandesande/Preprocessed_data/Populations/mean_data_fallers_non_fallers_HR.csv"
if dataset == 2 or snellius == True:
    if fallers == 1:
        data_SBP = pd.read_csv(csv_SBP)['Mean SBP Fallers (mmHg)']
        data_DBP = pd.read_csv(csv_DBP)['Mean DBP Fallers (mmHg)']
        data_HR = pd.read_csv(csv_HR)['Mean HR Fallers (bpm)']
    if fallers == 0:
        data_SBP = pd.read_csv(csv_SBP)['Mean SBP Non-Fallers (mmHg)']
        data_DBP = pd.read_csv(csv_DBP)['Mean DBP Non-Fallers (mmHg)']
        data_HR = pd.read_csv(csv_HR)['Mean HR Non-Fallers (bpm)']
    data_time = data_time_10Hz = pd.read_csv(csv_HR)['Time (s)']
    data_MAP = (1/3)*data_SBP + (2/3)*data_DBP
    sts0 = -120
    sts1 = 0
    sts_n = 0
    sit_n = 180

if dataset == 0 or dataset == 1:
    # Load .mat file
    data = loadmat(filename)

    data_time = np.squeeze(data['data']['time'][0][0])
    data_BP = np.squeeze(data['data']['BP'][0][0])
    data_cbfv_l = np.squeeze(data['data']['cbfv_l'][0][0])
    data_cbfv_r = np.squeeze(data['data']['cbfv_r'][0][0])

    data_time_10Hz = np.squeeze(data['data']['time_10Hz'][0][0])
    data_MAP = np.squeeze(data['data']['map'][0][0])
    data_SBP = np.squeeze(data['data']['sbp'][0][0])
    data_DBP = np.squeeze(data['data']['dbp'][0][0])
    data_HR = np.squeeze(data['data']['hr'][0][0])


    if dataset == 0:
        marker_name = 'markers'
        markers = np.squeeze(data['data'][marker_name][0][0][0][0])
        sts_n = markers[standup_n+3][0][0]
        sit_n = markers[standup_n][0][0]
    if dataset == 1:
        marker_name = 'marker'
        markers = np.squeeze(data['data'][marker_name][0][0][0][0])
        sts0 = round(markers[6][0][0], 3); # 1st start (for calibration of setpoints)
        sts1 = round(markers[0][0][0], 3); # 1st stand-up (for calibration of setpoints)
        sts_n = round(markers[-1+standup_n][0][0], 3); # 
        sit_n = round(markers[17+standup_n][0][0], 3); # 


# magic with indices and time for allignment
time_after_sts = sit_n-sts_n # (s)
if time_after_sts > 180:
    time_after_sts = 180
global dt_data, index_end, index_start, x_data, x_data_min, x_data_max, start_index_model, x_data_HR, model_start_t, model_end_t
dt_data = round(data_time_10Hz[1]-data_time_10Hz[0],5)
start_index_model = int(sts_model/2/dt_data)
index_end = int((time_after_sts+sts_model)/dt_data)

index_data_begin = find_index_of_time(data_time_10Hz, round(sts_n))-start_index_model
index_data_end = index_data_begin+index_end-start_index_model
index_data_begin_200Hz = index_data_begin*20
index_data_end_200Hz = index_data_end*20

x_data_time_10Hz = data_time_10Hz[index_data_begin:index_data_end]
x_data = data_MAP[index_data_begin:index_data_end]
x_data_min = data_DBP[index_data_begin:index_data_end]
x_data_max = data_SBP[index_data_begin:index_data_end]
x_data_HR = data_HR[index_data_begin:index_data_end]
x_data_time_200Hz = data_time[index_data_begin_200Hz:index_data_end_200Hz]
if include_doppler == 1:
    x_data_cbfv_l = data_cbfv_l[index_data_begin_200Hz:index_data_end_200Hz]
    x_data_cbfv_r = data_cbfv_r[index_data_begin_200Hz:index_data_end_200Hz]

model_start_t = 60.0 # s
model_end_t = 299.9 # s


#%% 
# Find optima of Doppler
if include_doppler == 1:
    def find_extrema_using_hr(data, heart_rate_data, time_values):
        filtered_peaks = []
        filtered_valleys = []
        filtered_peak_times = []
        filtered_valley_times = []
        filtered_peak_values = []
        filtered_valley_values = []

        i = 0  # Start index for heart rate data
        while i < len(heart_rate_data):
            hr = heart_rate_data[i]
            
            # Estimate the number of Doppler samples per beat based on the heart rate
            #samples_per_beat = int((60 / (hr)) * 200)  # Multiply by 200 because the Doppler signal is sampled at 200 Hz, only works if the heart rate data is clean
            samples_per_beat = int((60 / (60)) * 200)  # Multiply by 200 because the Doppler signal is sampled at 200 Hz, assume the heartrate to be 60

            
            # Define the window for this heartbeat cycle
            start_idx = i * 20  # Convert heart rate time (10Hz) to Doppler time (200Hz)
            end_idx = min(len(data), start_idx + samples_per_beat)

            # Find local maximum in this window
            local_max_idx = np.argmax(data[start_idx:end_idx]) + start_idx
            local_min_idx = np.argmin(data[start_idx:end_idx]) + start_idx
            
            # Ensure there's only one maximum and minimum in the window
            if local_max_idx not in filtered_peaks:
                filtered_peaks.append(local_max_idx)
                filtered_peak_values.append(data[local_max_idx])
                filtered_peak_times.append(time_values[local_max_idx])
            
            if local_min_idx not in filtered_valleys:
                filtered_valleys.append(local_min_idx)
                filtered_valley_values.append(data[local_min_idx])
                filtered_valley_times.append(time_values[local_min_idx])

            # Shift the window to the right: Start the next window from the end of this one
            i += samples_per_beat // 20  # Move to the next heart rate window
            
        return filtered_peaks, filtered_valleys, filtered_peak_times, filtered_valley_times, filtered_peak_values, filtered_valley_values

    # Assuming `x_data_HR` is your heart rate data (10Hz), and `x_data_time_200Hz` is the corresponding time data (200Hz)
    peaks_l, valleys_l, peak_times_l, valley_times_l, peak_velocities_l, valley_velocities_l = find_extrema_using_hr(data_cbfv_l, data_HR, data_time)
    peaks_r, valleys_r, peak_times_r, valley_times_r, peak_velocities_r, valley_velocities_r = find_extrema_using_hr(data_cbfv_r, data_HR, data_time)
    """
    lip_times_max_l, lip_max_mca_l = lin_interp_index(peak_times_l, peak_velocities_l, desired_interval, index_data_begin, index_data_end)
    lip_times_max_r, lip_max_mca_r = lin_interp_index(peak_times_r, peak_velocities_r, desired_interval, index_data_begin, index_data_end)
    lip_times_min_l, lip_min_mca_l = lin_interp_index(valley_times_l, valley_velocities_l, desired_interval, index_data_begin, index_data_end)
    lip_times_min_r, lip_min_mca_r = lin_interp_index(valley_times_r, valley_velocities_r, desired_interval, index_data_begin, index_data_end)
    """
    lip_times_max_l, lip_max_mca_l = lin_interp(peak_times_l, peak_velocities_l, desired_interval)
    lip_times_max_r, lip_max_mca_r = lin_interp(peak_times_r, peak_velocities_r, desired_interval)
    lip_times_min_l, lip_min_mca_l = lin_interp(valley_times_l, valley_velocities_l, desired_interval)
    lip_times_min_r, lip_min_mca_r = lin_interp(valley_times_r, valley_velocities_r, desired_interval)

    lip_max_mca_l = lip_max_mca_l[find_index_of_time(lip_times_max_l, x_data_time_10Hz[0] ):
    find_index_of_time(lip_times_max_l, x_data_time_10Hz[-1] )]
    lip_max_mca_r = lip_max_mca_r[find_index_of_time(lip_times_max_r, x_data_time_10Hz[0] ):
    find_index_of_time(lip_times_max_r, x_data_time_10Hz[-1] )]
    lip_min_mca_l = lip_min_mca_l[find_index_of_time(lip_times_min_l, x_data_time_10Hz[0] ):
    find_index_of_time(lip_times_min_l, x_data_time_10Hz[-1] )]
    lip_min_mca_r = lip_min_mca_r[find_index_of_time(lip_times_min_r, x_data_time_10Hz[0] ):
    find_index_of_time(lip_times_min_r, x_data_time_10Hz[-1] )]

    lip_times_max_l = lip_times_max_l[find_index_of_time(lip_times_max_l, x_data_time_10Hz[0] ):
    find_index_of_time(lip_times_max_l, x_data_time_10Hz[-1] )]
    lip_times_max_r = lip_times_max_r[find_index_of_time(lip_times_max_r, x_data_time_10Hz[0] ):
    find_index_of_time(lip_times_max_r, x_data_time_10Hz[-1] )]
    lip_times_min_l = lip_times_min_l[find_index_of_time(lip_times_min_l, x_data_time_10Hz[0] ):
    find_index_of_time(lip_times_min_l, x_data_time_10Hz[-1] )]
    lip_times_min_r = lip_times_min_r[find_index_of_time(lip_times_min_r, x_data_time_10Hz[0] ):
    find_index_of_time(lip_times_min_r, x_data_time_10Hz[-1] )]

    # Plotting
    plt.figure(figsize=(10, 6), dpi=3000)
    plt.plot(x_data_time_200Hz, x_data_cbfv_l, label=r'$v_{l}$', color='r', linewidth=0.1, linestyle="-")
    #plt.plot(peak_times_l, peak_velocities_l, "x", color='g', label='Peaks')
    #plt.plot(valley_times_l, valley_velocities_l, "x", color='y', label='Valleys')
    plt.plot(lip_times_max_l, lip_max_mca_l, label='Interpolated Peaks', color='g', linestyle="--", linewidth=0.6)
    plt.plot(lip_times_min_l, lip_min_mca_l, label='Interpolated Valleys', color='y', linestyle="--", linewidth=0.6)
    plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
    plt.xlabel('t (s)')
    plt.ylabel(r'$v$ (cm/s)')
    plt.title(r'Plot of left MCA velocities vs time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=3000)
    plt.plot(x_data_time_200Hz, x_data_cbfv_r, label=r'$v_{r}$', color='r', linewidth=0.1, linestyle="-")
    #plt.plot(peak_times_r, peak_velocities_r, "x", color='g', label='Peaks')
    #plt.plot(valley_times_r, valley_velocities_r, "x", color='y', label='Valleys')
    plt.plot(lip_times_max_r, lip_max_mca_r, label='Interpolated Peaks', color='g', linestyle="--", linewidth=0.6)
    plt.plot(lip_times_min_r, lip_min_mca_r, label='Interpolated Valleys', color='y', linestyle="--", linewidth=0.6)
    plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
    plt.xlabel('t (s)')
    plt.ylabel(r'$v$ (cm/s)')
    plt.title(r'Plot of right MCA velocities vs time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    global av_max_v, av_min_v
    times_max, av_max_v = average_velocity_on_matching_times(lip_times_max_l, lip_max_mca_l, lip_times_max_r, lip_max_mca_r)
    times_min, av_min_v = average_velocity_on_matching_times(lip_times_min_l, lip_min_mca_l, lip_times_min_r, lip_min_mca_r)

    plt.figure(figsize=(10, 6), dpi=3000)
    plt.plot(lip_times_max_r, lip_max_mca_r, label='max R', color='g', linestyle="--", linewidth=0.6)
    plt.plot(lip_times_max_l, lip_max_mca_l, label='max L', color='y', linestyle="--", linewidth=0.6)
    plt.plot(times_max, av_max_v, label='max av', color='b', linestyle="--", linewidth=0.6)
    plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
    plt.xlabel('t (s)')
    plt.ylabel(r'$v$ (cm/s)')
    plt.title(r'Plot of maximal MCA velocities vs time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6), dpi=3000)
    plt.plot(lip_times_min_r, lip_min_mca_r, label='min R', color='g', linestyle="--", linewidth=0.6)
    plt.plot(lip_times_min_l, lip_min_mca_l, label='min L', color='y', linestyle="--", linewidth=0.6)
    plt.plot(times_min, av_min_v, label='min av', color='b', linestyle="--", linewidth=0.6)
    plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
    plt.xlabel('t (s)')
    plt.ylabel(r'$v$ (cm/s)')
    plt.title(r'Plot of minimal MCA velocities vs time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()



#%% 
# CLEAN HR DATA
clean_HR_data_switch = False
if clean_HR_data_switch == True:
    # Step 1: Create DataFrame
    df = pd.DataFrame({'timestamp': x_data_time_10Hz, 'heart_rate': x_data_HR})

    # Convert the 'timestamp' column to TimedeltaIndex for resampling
    df['timestamp'] = pd.to_timedelta(df['timestamp'], unit='s')

    # Step 2: Calculate the first derivative of heart rate
    df['derivative'] = df['heart_rate'].diff() / 0.1  # Assuming 0.1 seconds between samples

    # Define a threshold for the derivative
    derivative_threshold = 4  # Adjust this value based on your data

    # Step 3: Identify anomalies where the derivative exceeds the threshold
    anomalies = np.abs(df['derivative']) > derivative_threshold

    # Step 4: Replace detected anomalies with NaN
    df.loc[anomalies, 'heart_rate'] = np.nan

    # Step 5: Interpolate missing values (outliers) using linear interpolation
    df['heart_rate'] = df['heart_rate'].interpolate(method='linear')

    # Step 6: Resample data to ensure consistent 10Hz frequency
    resampled_df = df.set_index('timestamp').resample('100ms').interpolate('linear')

    # Step 7: Retrieve cleaned heart rate data
    clean_heart_rate_values = resampled_df['heart_rate'].values

    # Plot the original and cleaned heart rate data
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(x_data_time_10Hz, x_data_HR, label='Original HR', color='r', linewidth=0.1, linestyle="-")
    plt.plot(resampled_df.index.total_seconds(), clean_heart_rate_values, label='Cleaned HR', color='b', linewidth=0.3, linestyle="-")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heart Rate (bpm)')
    plt.title('Original and Cleaned Heart Rate Data')
    plt.legend()
    plt.grid(True)
    plt.show()

    x_data_HR = clean_heart_rate_values

#%% 
# RUN GLOBAL OPTIMIZATION

algorithm = ISRES(n_offsprings=offspring, rule=1.0 / 7.0, gamma=0.2, alpha=0.5)
""" rule (Rule for Ranking):
    This parameter determines the trade-off between objective function ranking and constraint violation ranking. The value of rule influences how the algorithm ranks the solutions based on their objective function value and constraint violations.
    A lower rule value gives more emphasis on minimizing constraint violations, while a higher value focuses more on optimizing the objective function.
    The value of rule = 1.0 / 7.0 suggests a balance where the algorithm slightly favors the objective function but still considers constraint violations.
    
    gamma (Constraint Handling Parameter):
    gamma is a parameter that influences the pressure on satisfying the constraints during the optimization process. It essentially controls how strongly the algorithm penalizes constraint violations.
    A higher gamma value increases the penalty for constraint violations, pushing the algorithm to focus more on finding feasible solutions.
    In this case, gamma = 0.85 indicates a moderate emphasis on constraint satisfaction.
    
    alpha (Step Size Control):
    alpha is a parameter that controls the step size or the amount of exploration during the search process. It affects the magnitude of the changes made to the candidate solutions in each iteration.
    A higher alpha value increases the exploratory behavior of the algorithm, potentially allowing it to escape local minima but at the risk of slower convergence.
    Here, alpha = 0.2 suggests a relatively cautious approach, with smaller changes being made to the candidate solutions in each iteration, leading to more focused and controlled exploration."""

if snellius == True:
    if __name__ == "__main__":
        n_processes = multiprocessing.cpu_count()  # Adjust the number of processes as needed
        print(f"Number of cores: {n_processes}")
        pool = multiprocessing.Pool(n_processes)
        runner = StarmapParallelization(pool.starmap)

        # Pass the runner to the problem
        problem = MyProblem_HR_parallel(elementwise_runner=runner)

        # Perform optimization
        res = minimize(
            problem,
            algorithm,
            ("n_gen", iterations),
            seed=seed,
            verbose=True,
            save_history=True,
        )
        print("SUCCESS")
        print("Threads:", res.exec_time)

        # Close the pool after the optimization
        pool.close()
        pool.join()

if snellius == False:
    if include_doppler == False:
        problem = MyProblem_HR()
    if include_doppler == True:
        problem = MyProblem_BRAIN()
    res = minimize(problem,
                algorithm,
                ("n_gen", iterations),
                seed=seed,
                verbose=False,
                save_history=True)

#%% WRITE RESULTS TO FILE

write_results(res.X, res.F)

#%%
# RMSE per generation
val = [e.opt.get("F")[0] for e in res.history]
plt.plot(np.arange(len(val)), val)
plt.xticks(np.arange(len(val)))
plt.xlabel(f"Generation (offspring = ${offspring}$)")
plt.ylabel("Normalized RMSE")
plt.show()

# Save plot as PNG and EPS
plt.savefig("RMSE.png", format='png')
plt.savefig("RMSE.eps", format='eps')

# Save plot data
data_to_save = pd.DataFrame({'x': np.arange(len(val)), 'y': val})
data_to_save.to_csv("RMSE.csv", index=False)

print("Best solution found: \n X = %s \n F = %s \n CV = %s " % (res.X, res.F, res.CV) )

#%% 
# Absolute optimal parameter values per generation
val_para = [e.opt.get("X")[0] for e in res.history]

# Get a list of distinct colors from the 'tab20' colormap using the new method
colors = plt.colormaps['tab20'](np.linspace(0, 1, len(variable_names)))
"""
# Choose a continuous colormap, e.g., 'viridis'
cmap = plt.cm.get_cmap('viridis', n_variables)
# Generate 21 distinct colors
colors = cmap(np.linspace(0, 1, n_variables))
"""
# Plot each parameter's values with its corresponding color
for i, name in enumerate(variable_names):
    plt.plot(np.arange(len(val_para)), [vp[i] for vp in val_para], label=name, color=colors[i])
plt.title(f'Optimal parameter values per optimization generation (offspring = {offspring})')
plt.xticks(np.arange(len(val_para)))
plt.xlabel("Generation")
plt.ylabel("Parameter value")
# Adjust legend position to outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# Adjust layout to fit the legend properly
plt.tight_layout()

# Save plot as PNG and EPS
plt.savefig("params_abs.png", format='png')
plt.savefig("params_abs.eps", format='eps')

# Save plot data
# Convert the data into a dictionary suitable for saving
data_dict = {name: [vp[i] for vp in val_para] for i, name in enumerate(variable_names)}
data_df = pd.DataFrame(data_dict)
data_df.to_csv("plot_data.csv", index=False)

plt.show()


#%% Normalized optimal parameter values per generation
normalized_val_para = normalize_param(val_para, xu, xl)
# Plot each normalized parameter's values with its corresponding color
for i, name in enumerate(variable_names):
    plt.plot(np.arange(len(normalized_val_para)), [vp[i] for vp in normalized_val_para], label=name, color=colors[i])
plt.title(f'Optimal parameter values per optimization generation (offspring = {offspring})')
plt.xticks(np.arange(len(normalized_val_para)))
plt.xlabel("Generation")
plt.ylabel("Normalized Parameter Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save plot as PNG and EPS
plt.savefig("params_norm.png", format='png')
plt.savefig("params_norm.eps", format='eps')

# Save plot data
# Convert the data into a dictionary suitable for saving
data_dict = {name: [vp[i] for vp in normalized_val_para] for i, name in enumerate(variable_names)}
data_df = pd.DataFrame(data_dict)
data_df.to_csv("params_norm.csv", index=False)

plt.show()


#%% RUN LOCAL OPTIMIZATION
"""
# Perform curve fitting using Levenberg-Marquardt algorithm
x0 = res.X  # Initial guess for parameters [a, b, c, d]

#results = least_squares(model_function, x0, bounds=([0.9, 1.1]), method='trf', ftol=1e-8, xtol=1e-8, gtol=1e-8, verbose=2)
results = least_squares(model_function, x0, method='lm', ftol=1e-3,
                        xtol=1e-3, gtol=1e-3, verbose=1, x_scale='jac')

opti = results.x
print("\n Optimized parameters: ", opti)
"""

#%%
#res.X = [ 0.88521808, 0.57638988, 1.23869191, 0.63037545, 1.04719004, 1.52528282, 1.38392606, 1.20657392, 0.81749262, 1.04000896, 0.77240868, 1.38909163, 1.1387788, 1.19498142, 0.36249268, 3.10236141, 1.00832986, 0.81639036, 1.08779616, 1.0769972, 1.3786385, 1.03288004, 1.14197793, 1.36268182, -0.08804694] 
#Out_av, Out_wave, Out_solver = model.solve2(X)
Out_av, Out_wave, Out_solver = model.solve2(res.X)

#%% Index magic

#Out_av, Out_wave, Out_solver = model.solve2(opti)
#Out_av, Out_wave, Out_solver = model.solve2(np.ones([n_variables]))
if include_doppler == 0:
    t_mean, Finap, HR_model, BP_max, BP_min, HR_list = Out_av[0][0], Out_av[0][2], Out_av[0][3], Out_av[0][4], Out_av[0][5], Out_av[0][6]
if include_doppler == 1:
    t_mean, Finap, HR_model, BP_max, BP_min, HR_list, mca_t, mca_max, mca_min = Out_av[0][0], Out_av[0][2], Out_av[0][3], Out_av[0][4], Out_av[0][5], Out_av[0][6], Out_av[0][10], Out_av[0][11], Out_av[0][12]
t, wave_f = Out_wave[0][0], Out_wave[0][1]

lip_times, lip_minP = lin_interp(t_mean, BP_min, desired_interval)
lip_times = lip_times[start_index_model:index_end]
lip_minP = lip_minP[start_index_model:index_end]
lip_maxP = lin_interp(t_mean, BP_max, desired_interval)[1][start_index_model:index_end]
lip_HR = lin_interp(t_mean, HR_list, desired_interval)[1][start_index_model:index_end]

t_translated = (lip_times + sts_n - sts_model)
t_translated2 = (np.array(t) + sts_n - sts_model)

if include_doppler == 1:
    lip_mca_max = lin_interp(mca_t, mca_max, desired_interval)[1][start_index_model:index_end] # probably adjust indices
    lip_mca_min = lin_interp(mca_t, mca_min, desired_interval)[1][start_index_model:index_end] # probably adjust indices
    t_translated_mca = (mca_t + sts_n - sts_model)

# average RMSE of systolic and diastolic BP and HR
rmse_av_p = (1/3) * (rmse_norm(x_data_max, lip_maxP) + rmse_norm(x_data_min, lip_minP) + rmse_norm(x_data_HR, lip_HR))

#%% PLOT
# Sys and Dia

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(t_translated, lip_minP, label=r'$Dia_{model}$', color='#4169E1')
plt.plot(t_translated, lip_maxP, label=r'$Sys_{model}$', color='#B22222')
plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_DBP[index_data_begin:index_data_end], label=r'$Dia_{data}$', color='cornflowerblue', linestyle="--")
plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_SBP[index_data_begin:index_data_end], label=r'$Sys_{data}$', color='lightcoral', linestyle="--")
plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
#plt.axvline(x=sts_n+delta_t, color='grey', linestyle='--', label=r'$t_0 + \Delta t$')  # Add vertical line at UV
plt.xlabel('t (s)')
plt.ylabel(r'$P$ (mmHg)')
plt.title(r'Plot of systolic and diastolic $P$ vs time')
plt.legend(loc='upper left')
plt.grid(True)

# Save the plot as PNG and EPS
plt.savefig("pressure_plot.png", format='png')
plt.savefig("pressure_plot.eps", format='eps')

# Save data
# Organize data into a dictionary
data_dict = {
    't_translated': t_translated,
    'lip_minP': lip_minP,
    'lip_maxP': lip_maxP,
    'data_time_10Hz': data_time_10Hz[index_data_begin:index_data_end],
    'data_DBP': data_DBP[index_data_begin:index_data_end],
    'data_SBP': data_SBP[index_data_begin:index_data_end]
}

# Convert to DataFrame for easy saving
data_df = pd.DataFrame(data_dict)
data_df.to_csv("pressure_data.csv", index=False)
plt.show()


#%%  MAP
"""
fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

plot_data_start_time=100; # s
plot_data_start_MAP = plot_data_start_time * 10

p1, = host.plot(data_time_10Hz[index_data_begin:index_data_end], x_data, 'red', linewidth=.5, label="MAP_data")
p1, = host.plot(t_translated, lip_Finap, 'green', linewidth=.5, label="MAP_model")
#p1, = host.plot(t_av_trans[start_index_model:index_end], map_av2[start_index_model:index_end], 'blue', linewidth=.25, label="MAP_model2")
#p1, = host.plot(Output[0][0][startpointplot:],Output[0][1][startpointplot:],'red', linewidth=2, label="MABP")

#p1, = host.plot(t,cvp_c,'turquoise', linewidth=.5, label="CVP")
#p1, = host.plot(Output[1][0][startpointplot:],Output[1][1][startpointplot:],'blue', linewidth=2, label="MCVP")


plt.vlines(x=[sts_n], ymin=[0], ymax=[300], colors='k', ls='--', lw=1, label='vline_multiple - partial height')
host.set_ylim(-5,140) # ylimits for the blood pressure

host.set_xlabel("Time (seconds)")
host.set_ylabel("Pressure (mmHg)")

host.legend(('Mean Finapres', 'Simulated mean'), loc=4);

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', **tkw)
host.tick_params(axis='x', **tkw)

plt.show()
"""
#%% Waveform
"""
fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

plot_start_time=225; # s
plot_end_time=400; # s
plot_data_start_BP = plot_start_time * 200
plot_data_end_BP = plot_end_time * 200
plot_model_start = find_index_of_time(t_translated2, plot_start_time)
plot_model_end = find_index_of_time(t_translated2, plot_end_time)

p1, = host.plot(data_time[plot_data_start_BP:plot_data_end_BP], data_BP[plot_data_start_BP:plot_data_end_BP], 'red', linewidth=.4, label="MAP_data")
#p1, = host.plot(t_translated2[plot_model_start:plot_model_end], wave_f[2,plot_model_start:plot_model_end], 'blue', linewidth=.4, label="MAP_model")
#p1, = host.plot(t_translated3, wave_f2, 'blue', linewidth=.1, label="MAP_model2")

plt.vlines(x=[sts_n], ymin=[0], ymax=[300], colors='k', ls='--', lw=1, label='vline_multiple - partial height')
host.set_ylim(-5,140) # ylimits for the blood pressure

host.set_xlabel("Time (seconds)")
host.set_ylabel("Pressure (mmHg)")

host.legend(('Finapres waveform', 'Simulated waveform comp. 2'), loc=4);

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', **tkw)
host.tick_params(axis='x', **tkw)

plt.show()
"""
#%% NaN and Inf

indices = find_indices_of_nan_and_inf(opti_hist)
print(f"The indices of elements with NaN or inf are: {indices}")
#plot_scatter(opti_hist, indices)

#variable_names = [f'Var{i}' for i in range(19)]
if indices != []:
    plot_scatter_opti(opti_hist, indices, res.X, variable_names)


#%% HR

plt.figure(figsize=(10, 6), dpi=300)
plt.plot(data_time_10Hz[index_data_begin:index_data_end], data_HR[index_data_begin:index_data_end], label=r'$HR_{data}$', color='#B22222', ls="--")
plt.plot(t_translated, lip_HR, label=r'$HR_{model}$', color='lightcoral', linestyle="-")
plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
#plt.axvline(x=sts_n+delta_t, color='grey', linestyle='--', label=r'$t_0 + \Delta t$')  # Add vertical line at UV
plt.xlabel('t (s)')
plt.ylabel(r'$HR$ ($min^{-1}$)')
plt.title(r'Plot of heart rate ($HR$) vs time ($t$)')
plt.legend(loc='upper left')
plt.grid(True)

# Save the plot as PNG and EPS
plt.savefig("heart_rate_plot.png", format='png')
plt.savefig("heart_rate_plot.eps", format='eps')

# Save data
# Organize data into a dictionary
data_dict = {
    'data_time_10Hz': data_time_10Hz[index_data_begin:index_data_end],
    'data_HR': data_HR[index_data_begin:index_data_end],
    't_translated': t_translated,
    'lip_HR': lip_HR
}

# Convert to DataFrame for easy saving
data_df = pd.DataFrame(data_dict)
data_df.to_csv("heart_rate_data.csv", index=False)
plt.show()


#%% RESULTS
"""
array = np.array([
    0.5826683082261311, 0.5638604531557075, 0.5318674159872925, 0.6304879791429777,
    0.7242952295412761, 1.0467043690477267, 0.7604962987638669, 1.4466729673118304,
    1.3502707053869294, 0.9701380402162948, 1.084693169545678, 1.3695976134802526,
    0.8357467676121084, 0.9164333830338648, 1.104359123141875, 0.9076092600865857,
    1.0487338456148674, 4.61726979150161, 1.1112503540760827, 0.8609885864116209,
    0.7109305146197087, 0.8766995852060988, 0.9434873025258455, 1.0594999868228965,
    0.8974756227694911, 1.1424840201763422
])
Out_av, Out_wave, Out_solver = model.solve2(array)
t_mean, BP_max, BP_min, HR_list = Out_av[0][0], Out_av[0][4], Out_av[0][5], Out_av[0][6]
"""
#%% Interpolate model results to match data frequency
lip_times, lip_minP = lin_interp(t_mean, BP_min, desired_interval)
lip_times = lip_times[start_index_model:index_end]
lip_minP = lip_minP[start_index_model:index_end]
lip_maxP = lin_interp(t_mean, BP_max, desired_interval)[1][start_index_model:index_end]
lip_HR = lin_interp(t_mean, HR_list, desired_interval)[1][start_index_model:index_end]
t_translated = (lip_times + sts_n - sts_model)
#%% RMSE per timepoint
rmse_time_SBP = rmse_time(x_data_max, lip_maxP)
rmse_time_DBP = rmse_time(x_data_min, lip_minP)
rmse_time_HR = rmse_time(x_data_HR, lip_HR)
rmse_time_av = (1/3) * (rmse_time_SBP + rmse_time_DBP + rmse_time_HR)
rmse_SBP = np.mean(rmse_time_SBP)
rmse_DBP = np.mean(rmse_time_DBP)
rmse_HR = np.mean(rmse_time_HR)
rmse_av = np.mean(rmse_time_av)
print(f"Average RMSE for SBP: {rmse_SBP}")
print(f"Average RMSE for DBP: {rmse_DBP}")
print(f"Average RMSE for HR: {rmse_HR}")
print(f"Average RMSE for all parameters: {rmse_av}")

# PLOT
plt.figure(figsize=(10, 6), dpi=300)
#plt.plot(t_translated, rmse_time_SBP, label=r'$RMSE_{SBP}$', color='r', linewidth=0.5)
#plt.plot(t_translated, rmse_time_DBP, label=r'$RMSE_{DBP}$', color='b', linewidth=0.5)
#plt.plot(t_translated, rmse_time_HR, label=r'$RMSE_{HR}$', color='g', linewidth=0.5)
plt.plot(t_translated, rmse_time_av, label=r'$RMSE_{av}$', color='k', linewidth=0.5)
plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
plt.xlabel('t (s)')
plt.ylabel('RMSE')
plt.title('RMSE per timepoint')
plt.legend(loc='upper left')
plt.grid(True)

# Save the plot as PNG and EPS
plt.savefig("RMSE_T.png", format='png')
plt.savefig("RMSE_T.eps", format='eps')

# Save data
# Organize data into a dictionary
data_dict = {
    "t_translated": t_translated,
    "rmse_time_SBP": rmse_time_SBP,
    "rmse_time_DBP": rmse_time_DBP,
    "rmse_time_HR": rmse_time_HR,
    "rmse_time_av": rmse_time_av
}
# Convert to DataFrame for easy saving
data_df = pd.DataFrame(data_dict)
data_df.to_csv("RMSE_T_data.csv", index=False)

plt.show()


#%% Plot Doppler

if include_doppler == 1:
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(data_time_10Hz[index_data_begin:index_data_end], x_data_cbfv_l[index_data_begin:index_data_end], label=r'$v_{l,data}$', color='r', ls="--")
    plt.plot(data_time_10Hz[index_data_begin:index_data_end], x_data_cbfv_r[index_data_begin:index_data_end], label=r'$v_{r,data}$', color='b', ls="--")
    plt.plot(t_translated_mca, lip_max_mca_l, label='Interpolated Peaks', color='g', linestyle="--", linewidth=0.6)
    plt.axvline(x=sts_n, color='black', linestyle='--', label=r'$t_0$')  # Add vertical line at UV
    #plt.axvline(x=sts_n+delta_t, color='grey', linestyle='--', label=r'$t_0 + \Delta t$')  # Add vertical line at UV
    plt.xlabel('t (s)')
    plt.ylabel(r'$HR$ ($min^{-1}$)')
    plt.title(r'Plot of heart rate ($HR$) vs time ($t$)')
    plt.legend(loc='upper left')
    plt.grid(True)

    # Save the plot as PNG and EPS
    plt.savefig("CBFV.png", format='png')
    plt.savefig("CBFV.eps", format='eps')

    # Save data
    # Organize data into a dictionary
    data_dict = {
        data_time_10Hz: data_time_10Hz[index_data_begin:index_data_end],
        x_data_cbfv_l: x_data_cbfv_l[index_data_begin:index_data_end],
        x_data_cbfv_r: x_data_cbfv_r[index_data_begin:index_data_end]
    }
    # Convert to DataFrame for easy saving
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv("CBFV_data.csv", index=False)

    plt.show()
