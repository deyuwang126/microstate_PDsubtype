import numpy as np
import mne
from dyconnmap.fc.pli import pli
from scipy.io import savemat

# Read BrainVision format data
vhdr_file_path = 'vhdr_file_path'
raw = mne.io.read_raw_brainvision(vhdr_file_path, preload=True)
eeg_data_reshaped = raw._data

# Print data information 
print(raw.info)

# Read and parse events file
def read_seg_file(file_path):
    with open(file_path, 'r') as file:
        events = [int(line.strip().split('\t')[2]) for line in file if len(line.strip().split('\t')) > 2]
    return events

seg_file_path = '.seg_file_from_CARTOOL' #https://cartool.cibm.ch/
events = read_seg_file(seg_file_path)

# Calculate state duration
def calculate_state_duration(events, time_step=1):
    state_duration = []
    current_state = None
    current_state_start_time = None

    for time, state in enumerate(events):
        if current_state is None:
            current_state = state
            current_state_start_time = time * time_step
        elif current_state != state:
            state_duration.append((current_state, current_state_start_time, time * time_step))
            current_state = state
            current_state_start_time = time * time_step

    if current_state is not None:
        state_duration.append((current_state, current_state_start_time, len(events) * time_step))

    return state_duration

state_duration_result = calculate_state_duration(events)

# Extract and organize data by state
result_data_with_labels = [(state, eeg_data_reshaped[:, start:end+1]) for state, start, end in state_duration_result]

# Organize matrices by state
organized_data = {state: [] for state in range(5)}
for state, matrix in result_data_with_labels:
    organized_data[state].append(matrix)

# Calculate and organize average PLI by state
organized_avg_pli = {state: [] for state in range(5)}
for state, matrices in organized_data.items():
    for matrix in matrices:
        _, avg_pli = pli(matrix)
        organized_avg_pli[state].append(avg_pli)

# Process matrices and calculate the mean PLI
def process_matrices(matrices):
    matrices = np.array(matrices)
    array_lower = np.transpose(matrices, (0, 2, 1))
    array_combined = matrices + array_lower
    return np.mean(array_combined, axis=0)

average_data = [process_matrices(organized_avg_pli[state]) for state in range(1, 5)]

# Save the results
np.save('average_data.npy', average_data)
