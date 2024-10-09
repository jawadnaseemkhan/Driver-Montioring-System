import os
import sys
import subprocess

# Add the correct path where lanenet_model is located
sys.path.append('/lhome/jawakha/Desktop/Project/lanenet-lane-detection')

from lanenet_model import lanenet  # Now this should be resolved
from local_utils.config_utils.parse_config_utils import Config

# Path to your dataset (with resized frames)
dataset_directory = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA'

# Path to the pre-trained model
weights_path = '/lhome/jawakha/Desktop/Project/lanenet-lane-detection/checkpoints/tusimple_lanenet'

# Path to the config file
#config_path = '/lhome/jawakha/Desktop/Project/lanenet-lane-detection/config/tusimple_lanenet.yaml'
config_path = '/lhome/jawakha/Desktop/Project/lanenet-lane-detection/config/tusimple_lanenet.yaml'
print(f"Loading config from: {config_path}")

# Try reading the file directly
try:
    with open(config_path, 'r') as file:
        print("YAML file contents:")
        print(file.read())  # This will print the contents of the YAML file to confirm it is readable
except OSError as e:
    print(f"Error reading the YAML file: {e}")
# Directory to save the lane detection labels
base_save_dir = '/lhome/jawakha/Desktop/Project/Dataset/lanenet_labels'

# Path to the LaneNet testing script
test_script_path = '/lhome/jawakha/Desktop/Project/lanenet-lane-detection/tools/test_lanenet.py'

# Load the config
print(f"Loading config from: {config_path}")
lanenet_cfg = Config(config_path=config_path)

# Ensure the base save directory exists
if not os.path.exists(base_save_dir):
    os.makedirs(base_save_dir)

# Loop through all folders from 1 to 74
for folder_num in range(1, 75):
    folder_name = f"{folder_num:02d}"
    extracted_frames_dir = os.path.join(dataset_directory, folder_name, 'resized_frames')

    # Check if extracted frames exist
    if os.path.exists(extracted_frames_dir):
        # Create a save directory for each folder
        folder_save_dir = os.path.join(base_save_dir, folder_name)
        if not os.path.exists(folder_save_dir):
            os.makedirs(folder_save_dir)

        print(f"Processing folder: {folder_name}")

        # Loop through all frames in the folder
        for frame_file in os.listdir(extracted_frames_dir):
            frame_path = os.path.join(extracted_frames_dir, frame_file)
            
            # Ensure it's a valid image file
            if frame_file.endswith('.png') or frame_file.endswith('.jpg'):
                # Run inference for each frame and save the results as labels
                cmd = (
                    f"/bin/python3 {test_script_path} --weights_path {weights_path} "
                    f"--image_path {frame_path} --save_dir {folder_save_dir}"
                )
                print(f"Running lane detection on: {frame_path}")
                subprocess.run(cmd, shell=True)
    else:
        print(f"Extracted frames not found in folder: {folder_name}")
