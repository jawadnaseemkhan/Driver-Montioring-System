import os
import cv2

def extract_frames(video_path, output_folder, interval=5):
    """
    Extract frames from a video at a given interval.
    
    :param video_path: Path to the input video file.
    :param output_folder: Folder where the extracted frames will be saved.
    :param interval: Interval between frames to save (in frames).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if success and frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.png")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted frames saved to {output_folder}")

# Loop through your DR(eye)VE dataset folders and extract frames
dataset_directory = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA'
for folder_num in range(1, 75):
    folder_name = f"{folder_num:02d}"
    video_path = os.path.join(dataset_directory, folder_name, 'video_garmin.avi')
    output_folder = os.path.join(dataset_directory, folder_name, 'extracted_frames')
    extract_frames(video_path, output_folder, interval=5)
