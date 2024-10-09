import os
import cv2

def resize_frames(input_folder, output_folder, target_size=(512, 256)):
    """
    Resize images in a folder to the target size and save them in the output folder.
    
    :param input_folder: Path to the folder containing the original frames.
    :param output_folder: Path to the folder where resized frames will be saved.
    :param target_size: Target size for resizing (width, height).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):  # Assuming your images are in PNG format
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)
            resized_img = cv2.resize(img, target_size)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)
            print(f"Resized and saved: {output_path}")

def process_all_folders(dataset_directory, target_size=(512, 256)):
    """
    Loop through all 74 folders and resize the images in 'extracted_frames' subfolder.
    
    :param dataset_directory: Path to the root directory containing folders 01 to 74.
    :param target_size: Target size for resizing (width, height).
    """
    for folder_num in range(1, 75):
        folder_name = f"{folder_num:02d}"  # Format folder name with leading zeros
        input_folder = os.path.join(dataset_directory, folder_name, 'extracted_frames')
        output_folder = os.path.join(dataset_directory, folder_name, 'resized_frames')

        if os.path.exists(input_folder):
            print(f"Processing folder: {input_folder}")
            resize_frames(input_folder, output_folder, target_size)
        else:
            print(f"Folder {input_folder} does not exist. Skipping.")

# Path to the root dataset directory
dataset_directory = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA'

# Process all folders and resize frames to 512x256
process_all_folders(dataset_directory, target_size=(512, 256))
