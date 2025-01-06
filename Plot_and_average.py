import matplotlib.pyplot as plt
import scipy.io
import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter, butter, filtfilt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler



#%%
### SETTINGS ###
global time_before_sts, time_after_sts, separate_fallers_non_fallers, include_sts1, include_sts2, include_sts3
time_before_sts = 120 # s
time_after_sts = 180 # s
separate_fallers_non_fallers = True
# Make sure to True all sts if choosing Doppler !
include_sts1 = True
include_sts2 = False
include_sts3 = False
one_sts = True # Toggle if only one STS is needed. This makes sure STS1 is used if available, otherwise STS2, then STS3.
data_type = "NIRS" # "MAP", "SBP", "DBP", "HR" or "Doppler" --> FIX NIRS
if data_type == "Doppler":
    smoothing = True
    if smoothing == True:
        cutoff = 0.5  # Cutoff frequency in Hz
individual_plots = True
study = "PROHEALTH" # "PROHEALTH" or "NILVAD" --> FIX NILVAD



#%%
### DEF ###
def get_data(filename):
    # Load .mat file
    data = loadmat(filename)
    data_time_200Hz = np.squeeze(data['data']['time'][0][0])
    data_BP = np.squeeze(data['data']['BP'][0][0])
    data_time_10Hz = np.squeeze(data['data']['time_10Hz'][0][0])
    data_MAP = np.squeeze(data['data']['map'][0][0])
    data_SBP = np.squeeze(data['data']['sbp'][0][0])
    data_DBP = np.squeeze(data['data']['dbp'][0][0])
    markers = np.squeeze(data['data']['marker'][0][0][0][0])
    HR = np.squeeze(data['data']['hr_bp'][0][0])
    data_cbfv_r = np.squeeze(data['data']['cbfv_r'][0][0])
    data_cbfv_l = np.squeeze(data['data']['cbfv_l'][0][0])
    data_O2 = np.squeeze(data['data']['absO2Hb'][0][0])
    return data_time_200Hz, data_BP, data_time_10Hz, data_MAP, data_SBP, data_DBP, markers, HR, data_cbfv_r, data_cbfv_l, data_O2

def find_index_of_time(time_array, target_time, tolerance=1e-6):
    # Find indices where the difference is within a tolerance
    indices = np.where(np.isclose(time_array, target_time, atol=tolerance))[0]
    
    if indices.size > 0:  # Check if indices array is not empty
        return indices[0]  # Return the first index where the target_time occurs
    else:
        print(f"Time value {target_time} not found in the array.")
        return None

def index_start_end(time, sts):
    sts_index = find_index_of_time(time, round(sts, 1))
    dt = time[1] - time[0]
    start_index = int(sts_index - time_before_sts*(1/dt))
    end_index = int(sts_index + time_after_sts*(1/dt))
    return start_index, end_index

# Define a helper function to interpret the quality labels
def interpret_quality(quality):
    if isinstance(quality, str):
        quality = quality.lower().strip()
        if quality == 'not good' or quality == 'absent' or quality == 'less good':
            return []
        elif quality == '' or quality == 'moderate':
            return [1, 2, 3]
        elif 'only' in quality:
            return [int(num) for num in quality.split() if num.isdigit()]
        else:
            return [int(num) for num in quality if num.isdigit()]
    return [1, 2, 3]

from scipy.signal import butter, filtfilt
import numpy as np

def butterworth_filter(data, cutoff, fs, order=4):
    """
    Applies a Butterworth low-pass filter to the data.

    Parameters:
        data (array-like): The input signal (e.g., blood flow velocity).
        cutoff (float): The cutoff frequency for the filter in Hz.
        fs (float): The sampling frequency of the signal in Hz.
        order (int): The order of the filter (default is 4).

    Returns:
        filtered_data (array): The filtered signal.
    """
    # Calculate the Nyquist frequency
    nyquist = 0.5 * fs
    # Normalize the cutoff frequency
    normalized_cutoff = cutoff / nyquist
    # Design the Butterworth filter
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    # Apply the filter to the data
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data

def clean_heart_rate(x_data_time_10Hz, x_data_HR):
    
    # Step 1: Create DataFrame
    df = pd.DataFrame({'timestamp': x_data_time_10Hz, 'heart_rate': x_data_HR})

    # Convert the 'timestamp' column to TimedeltaIndex for resampling
    df['timestamp'] = pd.to_timedelta(df['timestamp'], unit='s')

    # Step 2: Calculate the first derivative of heart rate
    df['derivative'] = df['heart_rate'].diff() / 0.1  # Assuming 0.1 seconds between samples

    # Define a threshold for the derivative
    derivative_threshold = 0.9  # Adjust this value based on your data

    # Step 3: Identify anomalies where the derivative exceeds the threshold
    anomalies = np.abs(df['derivative']) > derivative_threshold

    # Step 4: Replace detected anomalies with NaN
    df.loc[anomalies, 'heart_rate'] = np.nan

    # Step 5: Interpolate missing values (outliers) using linear interpolation
    df['heart_rate'] = df['heart_rate'].interpolate(method='linear')

    # Step 6: Resample data to ensure consistent 10Hz frequency
    resampled_df = df.set_index('timestamp').resample('100ms').interpolate('linear')

    # Step 7: Retrieve cleaned heart rate data and time values
    clean_heart_rate_values = resampled_df['heart_rate'].values
    time = np.array(resampled_df.index.total_seconds())
    """
    # Plot the original and cleaned heart rate data
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(x_data_time_10Hz, x_data_HR, label='Original HR', color='r', linewidth=0.4, linestyle="-")
    plt.plot(time, clean_heart_rate_values, label='Cleaned HR', color='b', linewidth=0.4, linestyle="-")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Heart Rate (bpm)')
    plt.title('Original and Cleaned Heart Rate Data')
    plt.legend()
    plt.grid(True)
    plt.show()
    """
    return clean_heart_rate_values, time



def preprocess_data(data):
    """Replace NaN or Inf with mean."""
    data = np.nan_to_num(data, nan=np.nanmean(data))
    data[data == np.inf] = np.nanmean(data)
    data[data == -np.inf] = np.nanmean(data)
    return data

def replace_nans_with_time_increments(timestamps, time_step=0.1):
    """Replace NaNs in the timestamps with time increments (e.g., 0.1 seconds)."""
    if np.isnan(timestamps[0]):
        timestamps[0] = 0.0  # Start at 0 or another chosen start time

    for i in range(1, len(timestamps)):
        if np.isnan(timestamps[i]):
            timestamps[i] = timestamps[i - 1] + time_step  # Increment by the time_step (0.1 sec)
    
    return timestamps

def detect_outliers_with_replacement(timestamps, data, contamination=0.08, time_step=0.1):
    """Detect and replace outliers in heart rate data and adjust timestamps."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_flags = iso_forest.fit_predict(data_scaled)
    
    cleaned_data = data.copy()
    cleaned_timestamps = timestamps.copy()
    
    # Replace outliers with NaN
    cleaned_data[outlier_flags == -1] = np.nan
    cleaned_timestamps[outlier_flags == -1] = np.nan  # Mark corresponding timestamps as NaN
    
    # Interpolate missing values for data (heart rate), but not for timestamps
    cleaned_data = pd.Series(cleaned_data).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
    
    # Replace NaNs in timestamps with time increments instead of interpolating
    cleaned_timestamps = replace_nans_with_time_increments(cleaned_timestamps, time_step)

    return cleaned_timestamps, cleaned_data, outlier_flags

def apply_butterworth_filter(data, cutoff=0.05, fs=10, order=1):
    """Apply a zero-phase Butterworth filter for noise reduction."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def clean_heart_rate_data(timestamps, heart_rate):
    """Complete cleaning pipeline for heart rate and timestamps."""
    
    # Step 1: Replace NaNs or infs
    timestamps = replace_nans_with_time_increments(timestamps)
    heart_rate = preprocess_data(heart_rate)
    
    # Step 2: Outlier detection and replacement
    cleaned_timestamps, cleaned_heart_rate, outliers = detect_outliers_with_replacement(timestamps, heart_rate)

    # Step 3: Apply Butterworth filter to the heart rate data
    denoised_heart_rate = apply_butterworth_filter(cleaned_heart_rate)
    
    return cleaned_timestamps, denoised_heart_rate, outliers





#%%
### RUN ###
# Initialize lists to store valid data and patient IDs
all_data = []
all_record_ids = []
quality = []

this_dir = r"C:\Users\jsande\OneDrive - UvA\Documenten\UvA\Code\CVSIM\Lex\Brain2"

if study == "PROHEALTH":
# Base directory and file pattern
    base_dir = r"C:\Users\jsande\OneDrive - UvA\Documenten\UvA\Data\AMC_OneDrive_4-5-2024\Data validation study PROHEALTH\Preprocessed data"
    file_pattern = "Preprocessed_PHI{:03d}_FSup.mat"
    csv_file = r"C:\Users\jsande\OneDrive - UvA\Documenten\UvA\Data\AMC_OneDrive_4-5-2024\Data validation study PROHEALTH\PROHEALTH-I_export_olderadults.csv"
    quality_file = r"C:\Users\jsande\OneDrive - UvA\Documenten\UvA\Data\AMC_OneDrive_4-5-2024\Data validation study PROHEALTH\Overview data quality.xlsx"

if study == "NILVAD":
    base_dir = r"C:\Users\jsande\OneDrive - UvA\Documenten\UvA\Data\AMC_OneDrive_4-5-2024\Data validation study PROHEALTH\Preprocessed data"
    file_pattern = "Preprocessed_PHI{:03d}_FSup.mat"
    csv_file = r"C:\Users\jsande\OneDrive - UvA\Documenten\UvA\Data\AMC_OneDrive_4-5-2024\Data validation study PROHEALTH\PROHEALTH-I_export_olderadults.csv"
    quality_file = r"C:\Users\jsande\OneDrive - UvA\Documenten\UvA\Data\AMC_OneDrive_4-5-2024\Data validation study PROHEALTH\Overview data quality.xlsx"

# Read the CSV file using the dynamically determined column names
try:
    df = pd.read_csv(
        csv_file,
        header=0,                # Use the first row as the header
        na_values=['', 'NA'],    # Treat empty strings and 'NA' as NaN
        delimiter=';',           # Specify the delimiter
    )
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")

# Extract 'Record Id' and 'Falls_year' columns
record_ids = df['Record Id']
falls_year = df['Falls_year']

# Separate into fallers and non-fallers
fallers = df[df['Falls_year'] > 0]['Record Id'].values  # Record Ids for fallers
non_fallers = df[df['Falls_year'] == 0]['Record Id'].values  # Record Ids for non-fallers

# Read the quality file with the second row as header
df_quality = pd.read_excel(quality_file, sheet_name='Artefacts', header=1)

# Strip whitespace from the column names
df_quality.columns = df_quality.columns.str.strip()

# Now you should be able to extract the relevant columns
df_quality_filtered = df_quality[['Unnamed: 0', 'TCD-left', 'TCD-right']]
tcd_left_quality = df_quality['TCD-left'].values
tcd_right_quality = df_quality['TCD-right'].values

# Initialize lists to store the data
all_data = []
all_record_ids = []

# Loop over the range PHI011 to PHI043
for i in range(11, 43):
    filename = os.path.join(base_dir, f"PHI{i:03d}", file_pattern.format(i))
    
    if os.path.exists(filename):
        # Load the .mat file
        print(f"Importing {filename}.")
        data_time_200Hz, data_BP, data_time_10Hz, data_MAP, data_SBP, data_DBP, markers, HR, data_cbfv_r, data_cbfv_l, data_O2 = get_data(filename)
        sts1 = round(markers[0][0][0], 3); # 1st stand-up
        sts2 = round(markers[1][0][0], 3); # 2nd stand-up
        sts3 = round(markers[2][0][0], 3); # 3rd stand-up

        if data_type == "MAP":
            data = data_MAP
            time = data_time_10Hz
        
        if data_type == "SBP":
            data = data_SBP
            time = data_time_10Hz

        if data_type == "DBP":
            data = data_DBP
            time = data_time_10Hz

        if data_type == "HR":
            data_raw = HR
            time_raw = data_time_10Hz

            clean_HR_data_switch = 1  # Set to 1 to clean the HR data, 0 to skip cleaning
            if clean_HR_data_switch == 1:
                time, denoised_data, outliers = clean_heart_rate_data(time_raw, data_raw)
                #data_raw1, time_raw1 = clean_heart_rate(time, denoised_data)

                # Plot Results
                plt.figure(figsize=(10, 6), dpi=300)
                plt.plot(time_raw, data_raw, label='Raw Data', alpha=0.6)
                #plt.plot(cleaned_data, label='Cleaned Data', linewidth=2)
                plt.plot(time, denoised_data, label='Denoised Data', linewidth=2)
                #plt.scatter(np.where(outliers == -1)[0], data_raw[outliers == -1], color='red', label='Outliers')
                plt.legend()
                plt.legend(loc='lower left')
                plt.title("Cleaned Heart ($HR$) Rate Data")
                plt.xlabel("Time (s)")
                plt.ylabel("HR (bpm)")
                plt.grid()
                
                # Define the output file name (make sure the directory exists)
                output_file = os.path.join(this_dir, f"cleaned_heart_rate_plot_{i:03d}.png")
                
                # Save the plot as PNG (or any other format like EPS, PDF)
                plt.savefig(output_file, format='png', bbox_inches='tight')

                # Optional: Close the plot to avoid overlapping plots in the next iteration
                plt.show()
                plt.close()
                
                data = denoised_data
                
            else:
                data = data_raw # Use the raw HR data without cleaning

        if data_type == "NIRS":
            data = data_O2
            time = np.linspace(0, len(data)/100, len(data))  # Assuming 100 Hz sampling rate

        if data_type == "Doppler":
            time = data_time_200Hz
            patient_id = f"PHI{i:03d}"

            # Check if this patient is in the quality file
            if patient_id in df_quality_filtered['Unnamed: 0'].values:
                print(f"Found record {patient_id} in the quality file.")

                idx = list(df_quality_filtered['Unnamed: 0']).index(patient_id)
                left_quality = interpret_quality(tcd_left_quality[idx])
                right_quality = interpret_quality(tcd_right_quality[idx])
                quality.append([[patient_id], [left_quality], [right_quality]])
                
                data_l = data_cbfv_l
                data_r = data_cbfv_r

                fs = 1/(time[1] - time[0])  # Sampling frequency in Hz
                if smoothing == True:
                    data_l = butterworth_filter(data_l, cutoff=cutoff, fs=fs)
                    data_r = butterworth_filter(data_r, cutoff=cutoff, fs=fs)

                if one_sts == False:
                    # Now you can load the data_l for this patient and include the necessary sit-to-stands
                    if include_sts1 == True:
                        if 1 in left_quality and 1 in right_quality:
                            start_index, end_index = index_start_end(time, sts1)
                            all_data.append( (data_l[start_index:end_index] + data_r[start_index:end_index])/2 )
                            all_record_ids.append(patient_id)
                        elif 1 in left_quality:
                            start_index, end_index = index_start_end(time, sts1)
                            all_data.append(data_l[start_index:end_index])
                            all_record_ids.append(patient_id)
                        elif 1 in right_quality:
                            start_index, end_index = index_start_end(time, sts1)
                            all_data.append(data_r[start_index:end_index])
                            all_record_ids.append(patient_id)
                        
                    if include_sts2 == True:
                        if 2 in left_quality and 2 in right_quality:
                            start_index, end_index = index_start_end(time, sts2)
                            all_data.append( (data_l[start_index:end_index] + data_r[start_index:end_index])/2 )
                            all_record_ids.append(patient_id)
                        elif 2 in left_quality:
                            start_index, end_index = index_start_end(time, sts2)
                            all_data.append(data_l[start_index:end_index])
                            all_record_ids.append(patient_id)
                        elif 2 in right_quality:
                            start_index, end_index = index_start_end(time, sts2)
                            all_data.append(data_r[start_index:end_index])
                            all_record_ids.append(patient_id)
                    
                    if include_sts3 == True:
                        if 3 in left_quality and 3 in right_quality:
                            start_index, end_index = index_start_end(time, sts3)
                            all_data.append( (data_l[start_index:end_index] + data_r[start_index:end_index])/2 )
                            all_record_ids.append(patient_id)
                        elif 3 in left_quality:
                            start_index, end_index = index_start_end(time, sts3)
                            all_data.append(data_l[start_index:end_index])
                            all_record_ids.append(patient_id)
                        elif 3 in right_quality:
                            start_index, end_index = index_start_end(time, sts3)
                            all_data.append(data_r[start_index:end_index])
                            all_record_ids.append(patient_id)
                    
                if one_sts == True:
                    check = 1
                    # Now you can load the data_l for this patient and include the necessary sit-to-stands
                    if include_sts1 == True:
                        if 1 in left_quality and 1 in right_quality:
                            start_index, end_index = index_start_end(time, sts1)
                            all_data.append( (data_l[start_index:end_index] + data_r[start_index:end_index])/2 )
                            all_record_ids.append(patient_id)
                            check = 0
                        elif 1 in left_quality:
                            start_index, end_index = index_start_end(time, sts1)
                            all_data.append(data_l[start_index:end_index])
                            all_record_ids.append(patient_id)
                            check = 0
                        elif 1 in right_quality:
                            start_index, end_index = index_start_end(time, sts1)
                            all_data.append(data_r[start_index:end_index])
                            all_record_ids.append(patient_id)
                            check = 0
                        
                    if include_sts2 == True and check == 1:
                        if 2 in left_quality and 2 in right_quality:
                            start_index, end_index = index_start_end(time, sts2)
                            all_data.append( (data_l[start_index:end_index] + data_r[start_index:end_index])/2 )
                            all_record_ids.append(patient_id)
                            check = 0
                        elif 2 in left_quality:
                            start_index, end_index = index_start_end(time, sts2)
                            all_data.append(data_l[start_index:end_index])
                            all_record_ids.append(patient_id)
                            check = 0
                        elif 2 in right_quality:
                            start_index, end_index = index_start_end(time, sts2)
                            all_data.append(data_r[start_index:end_index])
                            all_record_ids.append(patient_id)
                            check = 0
                    
                    if include_sts3 == True and check == 1:
                        if 3 in left_quality and 3 in right_quality:
                            start_index, end_index = index_start_end(time, sts3)
                            all_data.append( (data_l[start_index:end_index] + data_r[start_index:end_index])/2 )
                            all_record_ids.append(patient_id)
                        elif 3 in left_quality:
                            start_index, end_index = index_start_end(time, sts3)
                            all_data.append(data_l[start_index:end_index])
                            all_record_ids.append(patient_id)
                        elif 3 in right_quality:
                            start_index, end_index = index_start_end(time, sts3)
                            all_data.append(data_r[start_index:end_index])
                            all_record_ids.append(patient_id)
            

        
        if data_type != "Doppler":
            if include_sts1 == True:
                start_index, end_index = index_start_end(time, sts1)
                all_data.append(data[start_index:end_index])
                all_record_ids.append(f"PHI{i:03d}")  # Store the Record Id corresponding to the loaded data
            if include_sts2 == True:
                start_index, end_index = index_start_end(time, sts2)
                all_data.append(data[start_index:end_index])
                all_record_ids.append(f"PHI{i:03d}")  # Store the Record Id corresponding to the loaded data
            if include_sts3 == True:
                start_index, end_index = index_start_end(time, sts3)
                all_data.append(data[start_index:end_index])
                all_record_ids.append(f"PHI{i:03d}")  # Store the Record Id corresponding to the loaded data
                
    else:
        print(f"File {filename} is missing, skipping.")

# Step 1: Separate the original MAP data into fallers and non-fallers
fallers_indices = [idx for idx, record_id in enumerate(all_record_ids) if record_id in fallers]
non_fallers_indices = [idx for idx, record_id in enumerate(all_record_ids) if record_id in non_fallers]

fallers_data = [all_data[idx] for idx in fallers_indices if idx < len(all_data)]
non_fallers_data = [all_data[idx] for idx in non_fallers_indices if idx < len(all_data)]
n_data = len(all_data)
n_fallers = len(fallers_data)
n_non_fallers = len(non_fallers_data)
print('Faller: ', len(fallers_data), 'Non-Fallers: ', len(non_fallers_data))

# Step 2: Pad the data
if fallers_data:
    max_length_fallers = max(len(data) for data in fallers_data)
    padded_fallers_data = [np.pad(data, (0, max_length_fallers - len(data)), mode='constant', constant_values=np.nan) for data in fallers_data]

    # Stack the padded arrays
    stacked_fallers = np.vstack(padded_fallers_data)

    # Calculate mean and standard deviation while ignoring NaN values
    mean_data_fallers = np.nanmean(stacked_fallers, axis=0)
    std_data_fallers = np.nanstd(stacked_fallers, axis=0)

else:
    print("No fallers data available.")

# Repeat similar steps for non-fallers if needed
if non_fallers_data:
    max_length_non_fallers = max(len(data) for data in non_fallers_data)
    padded_non_fallers_data = [np.pad(data, (0, max_length_non_fallers - len(data)), mode='constant', constant_values=np.nan) for data in non_fallers_data]

    # Stack the padded arrays
    stacked_non_fallers = np.vstack(padded_non_fallers_data)

    # Calculate mean and standard deviation while ignoring NaN values
    mean_data_non_fallers = np.nanmean(stacked_non_fallers, axis=0)
    std_data_non_fallers = np.nanstd(stacked_non_fallers, axis=0)

else:
    print("No non-fallers data available.")

# Step 3: Pad the data for all patients
max_length_data = max(len(data) for data in all_data)
padded_data = [np.pad(data, (0, max_length_data - len(data)), mode='constant', constant_values=np.nan) for data in all_data]

# Stack the padded arrays
stacked_data = np.vstack(padded_data)

# Calculate mean and standard deviation while ignoring NaN values
mean_data = np.nanmean(stacked_data, axis=0)
std_data = np.nanstd(stacked_data, axis=0)



#%%
### PLOT ###

# Example time array (you should replace this with your actual time data)
dt = time[1]-time[0]
time_array = np.linspace(-time_before_sts, time_after_sts-dt, len(mean_data))  # Assuming 200 Hz sampling rate

if data_type == 'Doppler':
    l_width = 2
    el_width = 0.01
    cap_size = 0.02
    alpha = 0.7 # Transparency level
if data_type == 'MAP' or data_type == 'NIRS' or data_type == 'SBP' or data_type == 'DBP':
    l_width = 3
    el_width = 0.1
    cap_size = 1
    alpha = 0.7 # Transparency level
if data_type == "HR":
    l_width = 2
    el_width = 0.1
    cap_size = 0.5
    alpha = 0.7 # Transparency level

plt.rcParams.update({'font.size': 14})  # Adjust 14 to your desired font size

# Plotting the mean with error bars representing the standard deviation
plt.figure(figsize=(10, 6), dpi=300)
if separate_fallers_non_fallers == True:
    plt.errorbar(time_array, mean_data_non_fallers, linewidth=l_width, color='navy', yerr=std_data_non_fallers, ecolor='cornflowerblue', elinewidth=el_width, capsize=cap_size, label=f'Non-Fallers (n={n_non_fallers})', alpha=alpha)
    plt.errorbar(time_array, mean_data_fallers, linewidth=l_width, color='darkred', yerr=std_data_fallers, ecolor='lightcoral', elinewidth=el_width, capsize=cap_size, label=f'Fallers (n={n_fallers})', alpha=alpha)
if separate_fallers_non_fallers == False:
    plt.errorbar(time_array, mean_data, linewidth=l_width, color='navy', yerr=std_data, ecolor='cornflowerblue', elinewidth=el_width, capsize=cap_size, label=f'Mean with SD (n={n_data})')
plt.axvline(x=0, color='grey', linestyle='--', label='STS')  # Add vertical line at UV

# Add labels and title
plt.xlabel('Time (s)')
if data_type == 'Doppler':
    plt.ylabel('CBFV (cm/s)')
    plt.title('Average Cerebral Blood Flow Velocity (CBFV) vs Time with Standard Deviation (SD)')
if data_type == 'MAP':
    plt.ylabel('MAP (mmHg)')
    plt.title('Average Mean Arterial Pressure (MAP) vs Time with Standard Deviation (SD)')
if data_type == 'SBP':
    plt.ylabel('SBP (mmHg)')
    plt.title('Average Systolic Blood Pressure (SBP) vs Time with Standard Deviation (SD)')
if data_type == 'DBP':
    plt.ylabel('DBP (mmHg)')
    plt.title('Average Diastolic Blood Pressure (DBP) vs Time with Standard Deviation (SD)')
if data_type == 'HR':
    plt.ylabel('HR (bpm)')
    plt.title('Average Heart Rate (HR) vs Time with Standard Deviation (SD)')
if data_type == 'NIRS':
    plt.ylabel('O2Hb (umol/L)')
    plt.title('Average O2Hb vs Time with Standard Deviation (SD)')
plt.legend(loc='upper left')
# Show the plot
plt.grid(True)
plt.show()

if individual_plots == True:
    plt.figure(figsize=(10, 6), dpi=300)
    if separate_fallers_non_fallers == True:
        for i in range(n_fallers):
            plt.plot(time_array, padded_fallers_data[i], linewidth=1, color='lightcoral', alpha=1)
        for i in range(n_non_fallers):
            plt.plot(time_array, padded_non_fallers_data[i], linewidth=1, color='cornflowerblue', alpha=1)
        # Add representative lines for legend
        plt.plot([], [], color='lightcoral', label='Fallers')
        plt.plot([], [], color='cornflowerblue', label='Non-Fallers')
        plt.legend(loc='upper left')
    if separate_fallers_non_fallers == False:
        for i in range(n_data):
            plt.plot(time_array, padded_data[i], linewidth=1, color='grey', alpha=0.5)
    plt.axvline(x=0, color='grey', linestyle='--', label='STS')  # Add vertical line at UV
    plt.xlabel('Time (s)')
    if data_type == 'Doppler':
        plt.ylabel('CBFV (cm/s)')
        plt.title('Cerebral Blood Flow Velocity (CBFV) vs Time for all Patients')
    if data_type == 'MAP':
        plt.ylabel('MAP (mmHg)')
        plt.title('Mean Arterial Pressure (MAP) vs Time for all Patients')
    if data_type == 'SBP':
        plt.ylabel('SBP (mmHg)')
        plt.title('Systolic Blood Pressure (SBP) vs Time for all Patients')
    if data_type == 'DBP':
        plt.ylabel('DBP (mmHg)')
        plt.title('Diastolic Blood Pressure (DBP) vs Time for all Patients')
    if data_type == 'HR':
        plt.ylabel('HR (bpm)')
        plt.title('Heart Rate (HR) vs Time for all Patients')
    if data_type == 'NIRS':
        plt.ylabel('O2Hb (umol/L)')
        plt.title('O2Hb vs Time for all Patients')
    plt.grid(True)
    plt.show()

#%%
# EXPORT average data

# Save the mean data to a .csv file
if separate_fallers_non_fallers == True:
    if data_type == 'Doppler':
        np.savetxt(f"Data_fallers_non_fallers_mean_{data_type}.csv", np.vstack((time_array, mean_data_fallers, std_data_fallers, mean_data_non_fallers, std_data_non_fallers)).T, delimiter=',', header='Time (s),Mean CBFV Fallers (cm/s),SD CBFV Fallers (cm/s),Mean CBFV Non-Fallers (cm/s),SD CBFV Non-Fallers (cm/s)', comments='')
    if data_type == 'MAP':
        np.savetxt(f"Data_fallers_non_fallers_mean_{data_type}.csv", np.vstack((time_array, mean_data_fallers, std_data_fallers, mean_data_non_fallers, std_data_non_fallers)).T, delimiter=',', header='Time (s),Mean MAP Fallers (mmHg),SD MAP Fallers (mmHg),Mean MAP Non-Fallers (mmHg),SD MAP Non-Fallers (mmHg)', comments='')
    if data_type == 'SBP':
        np.savetxt(f"Data_fallers_non_fallers_mean_{data_type}.csv", np.vstack((time_array, mean_data_fallers, std_data_fallers, mean_data_non_fallers, std_data_non_fallers)).T, delimiter=',', header='Time (s),Mean SBP Fallers (mmHg),SD SBP Fallers (mmHg),Mean SBP Non-Fallers (mmHg),SD SBP Non-Fallers (mmHg)', comments='')
    if data_type == 'DBP':
        np.savetxt(f"Data_fallers_non_fallers_mean_{data_type}.csv", np.vstack((time_array, mean_data_fallers, std_data_fallers, mean_data_non_fallers, std_data_non_fallers)).T, delimiter=',', header='Time (s),Mean DBP Fallers (mmHg),SD DBP Fallers (mmHg),Mean DBP Non-Fallers (mmHg),SD DBP Non-Fallers (mmHg)', comments='')
    if data_type == 'HR':
        np.savetxt(f"Data_fallers_non_fallers_mean_{data_type}.csv", np.vstack((time_array, mean_data_fallers, std_data_fallers, mean_data_non_fallers, std_data_non_fallers)).T, delimiter=',', header='Time (s),Mean HR Fallers (bpm),SD HR Fallers (bpm),Mean HR Non-Fallers (bpm),SD HR Non-Fallers (bpm)', comments='')
    if data_type == 'NIRS':
        np.savetxt(f"Data_fallers_non_fallers_mean_{data_type}.csv", np.vstack((time_array, mean_data_fallers, std_data_fallers, mean_data_non_fallers, std_data_non_fallers)).T, delimiter=',', header='Time (s),Mean O2Hb Fallers (umol/L),SD O2Hb Fallers (umol/L),Mean O2Hb Non-Fallers (umol/L),SD O2Hb Non-Fallers (umol/L)', comments='')
if separate_fallers_non_fallers == False:
    if data_type == 'Doppler':
        np.savetxt(f"Data_mean_{data_type}.csv", np.vstack((time_array, mean_data, std_data)).T, delimiter=',', header='Time (s),Mean CBFV (cm/s),SD CBFV (cm/s)', comments='')
    if data_type == 'MAP':
        np.savetxt(f"Data_mean_{data_type}.csv", np.vstack((time_array, mean_data, std_data)).T, delimiter=',', header='Time (s),Mean MAP (mmHg),SD MAP (mmHg)', comments='')
    if data_type == 'SBP':
        np.savetxt(f"Data_mean_{data_type}.csv", np.vstack((time_array, mean_data, std_data)).T, delimiter=',', header='Time (s),Mean SBP (mmHg),SD SBP (mmHg)', comments='')
    if data_type == 'DBP':
        np.savetxt(f"Data_mean_{data_type}.csv", np.vstack((time_array, mean_data, std_data)).T, delimiter=',', header='Time (s),Mean DBP (mmHg),SD DBP (mmHg)', comments='')
    if data_type == 'HR':
        np.savetxt(f"Data_mean_{data_type}.csv", np.vstack((time_array, mean_data, std_data)).T, delimiter=',', header='Time (s),Mean HR (bpm),SD HR (bpm)', comments='')
    if data_type == 'NIRS':
        np.savetxt(f"Data_mean_{data_type}.csv", np.vstack((time_array, mean_data, std_data)).T, delimiter=',', header='Time (s),Mean O2Hb (umol/L),SD O2Hb (umol/L)', comments='')
print("Data exported.")




#%%
# CA
"""
# Create the plot with the specified size and resolution
plt.figure(figsize=(5, 3), dpi=300)

# Set x-axis label and limit from 0 to 200, showing only 0, 100, and 200
plt.xlabel('Perfusion Pressure (mmHg)')
plt.xticks([0, 100, 200])
plt.xlim(0, 200)

# Set y-axis label but remove the tick numbers
plt.ylabel('Blood Flow')
plt.yticks([])  # Remove y-axis ticks

# Show the plot
plt.show()
"""

#%%
