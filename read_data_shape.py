import os
import numpy as np

folder_path = "/data1/wuruoyu/waveform-inversion/train_samples/CurveFault_A"

# List all .npy files
npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# Load each file and print its shape
for file_name in npy_files:
    file_path = os.path.join(folder_path, file_name)
    try:
        data = np.load(file_path)
        print(f"{file_name}: {data.shape}")
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
