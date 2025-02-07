Overview data quality.xls, with three data sheets:
- Timing markers: provides an overview of the marker times. Preferably, PortaSync markers (block pulses) were used, when these were present in both NIRS (oxy)
and blood pressure/TCD (Acqknowledge) files. However, these were sometimes missed/put too often, or not synchronized well. Therefore, this Excel
file provides all times (respective to the start of the Acqknowledge file) of the markers, and the start time of both the Acqknowledge and OxySoft 
file. 
- Artefacts: shows the exclusion of signals per postural change type and repetition. 
- NIRS artefacts: presents the classification of artefacts that were present in the NIRS signals throughout all repetitions of all posture change 
types.



DATA-SPECIFIC INFORMATION FOR: Preprocessed data (folder)
The postural changes are indicated with four markers (set by the PortaSync with block pulses, otherwise the times are indicated in the Excel file data quality.xls, each:
	- Start 5 minutes baseline (or in case of squat 1 minute).
	- Start standing up, participant starts moving in response to cue. 
	- End standing up, participant is fully standing. 
	- Stop, after 3 minutes of standing. 

The .mat files contain a struct with the following fields:
	- oxyvals: oxygenated haemoglobin, 12 channels: first 6 left, then 6 right. Relative values. In umol/L.
	- dxyvals: deoxygenated haemoglobin, 12 channels: first 6 left, then 6 right. Relative values. In umol/L.
	- absO2Hb: absolute values for oxygenated haemoglobin. In umol/L. 
	- absHHb: absolute values for deoxygenated haemoglobin. In umol/L.
	- TSI: tissue saturation index for left and right. In %.
	- TSI_FF: fit factor of tissue saturation index. 
	- nirs_time: time vector of NIRS signal, 0 is when device was started.  In seconds. 
	- fs_nirs: sampling frequency of NIRS device. Note: sampling frequency of other signals (BP, TCD and ECG) is 200 Hz. 
	- ADlabel: corresponding to the various extra parameters measured by the NIRS sensor: e.g. temperature, accelerometers, gyroscope.
	- ADvalues: values corresponding to the labels.
	- ECG: contains ecg signal, in mV. 
	- TCD_r: transcranial Doppler signal right, cerebral blood flow velocity in cm/s.
	- TCD_l: transcranial Doppler signal left, cerebral blood flow velocity in cm/s.
	- BP: blood pressure signal in mmHg. 
	- markers_acq: signal indicating when someone stood up: pulse for start measurement, start standing up, end standing up (when fully standing, indicating the time needed to complete the manoeuvre), end measurement. 
	- markerstijd: time in this struct when markers are present. So first value represents start measurement 1, 2nd value start standing up during measurement 1 etc. In seconds. 

Missing data or data of insufficient quality, which were later excluded, can be found in the excel file 'data quality.xls'. 

