import os
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d

def find_index_of_time(time_array, target_time, tol=1e-5):
    indices = np.where(np.isclose(time_array, target_time, atol=tol))[0]  # Get indices where target_time occurs within the tolerance
    if indices.size > 0:  # Check if indices array is not empty
        return indices[0]  # Return the first index where the target_time occurs
    else:
        print(f"Time value {target_time} not found in the array.")
        return None
    
def round_time_series(time_series, decimals=4):
    return [round(element, decimals) for element in time_series]
        
def lim(compartment, comp_norm, limit):
    if compartment < 1/limit*comp_norm:
        compartment = 1/limit*comp_norm
        #print("R lower limit")
    if compartment > limit*comp_norm:
        compartment = limit*comp_norm
        #print("R upper limit")
    return compartment

def get_data(filename):
    # Load .mat file
    data = loadmat(filename)
    data_time = np.squeeze(data['data']['time'][0][0])
    data_BP = np.squeeze(data['data']['BP'][0][0])
    data_time_10Hz = np.squeeze(data['data']['time_10Hz'][0][0])
    data_MAP = np.squeeze(data['data']['map'][0][0])
    data_SBP = np.squeeze(data['data']['sbp'][0][0])
    data_DBP = np.squeeze(data['data']['dbp'][0][0])
    return data, data_DBP, data_SBP, data_time, data_BP, data_time_10Hz, data_MAP
    
def extract_patient_id(filename):
    # Extract the patient ID using string manipulation
    # Split the path into parts and find the part that contains 'PHI'
    parts = filename.split(os.sep)
    patient_id = None
    for part in parts:
        if 'PHI' in part:
            patient_id = part
            break
    
    # Extracted patient ID
    if patient_id:
        print(f'Extracted patient ID: {patient_id}')
        return patient_id
    else:
        print('Patient ID not found in the given file path.')
    return patient_id

def alpha_rad_full_func(tilt_time, tau_STS, p=0.25):
    """
    Calculate alpha_rad_full with flattened maxima and constant peak value.
    
    Parameters:
    tilt_time (array-like): Time values for the tilt motion.
    tau_STS (float): Period of the tilt motion.
    p (float): Power to flatten the maxima (p >= 1).
    
    Returns:
    numpy.ndarray: Computed alpha_rad_full values.
    """
    cos_term = 1.0 - np.cos(2 * np.pi * tilt_time / tau_STS)
    normalization = (1.0 - np.cos(np.pi)) ** p
    return cos_term ** p / normalization

def lin_interp(t, values, desired_interval):
    # Create new time array with the desired frequency, starting from the first whole number in the range
    start_time = np.ceil(t[0] * 10) / 10  # Adjusting to start from the next whole 0.1 increment
    end_time = np.floor((t[-1] - desired_interval) * 10) / 10  # Adjusting to stay within bounds
    new_times = np.arange(start_time, end_time + desired_interval, desired_interval)

    # Perform linear interpolation
    linear_interp = interp1d(t, values, kind='linear')
    new_values = linear_interp(new_times)
    return new_times, new_values

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
    