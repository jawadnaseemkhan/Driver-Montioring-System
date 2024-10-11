import os
import cv2
import multiprocessing
import subprocess
from lane_assist import process_road_videos
from facial_expressions import detect_drowsiness_and_emotions

# Function to process road videos for lane detection in a separate process
def process_road_videos_worker(directory, control_queue):
    process_road_videos(directory)
    # Send message to indicate completion of road video processing
    control_queue.put('done')

# Function to handle driver monitoring via webcam in a separate process
def run_webcam_worker(control_queue):
    frame_counter = 0
    cam_cap = cv2.VideoCapture(0)

    if not cam_cap.isOpened():
        print("Error: Could not open webcam")
        return

    while True:
        ret_cam, cam_frame = cam_cap.read()

        if not ret_cam:
            print("Error: Unable to capture webcam feed.")
            break

        # Process camera frame for drowsiness and emotions
        cam_frame_processed, frame_counter = detect_drowsiness_and_emotions(cam_frame, frame_counter)
        
        # Display the processed camera frame
        cv2.imshow('Facial Expression Detection', cam_frame_processed)

        # Check for control signal from road video process
        if not control_queue.empty():
            msg = control_queue.get()
            if msg == 'done':  # Stop when road video is done
                break

        # Exit if ESC is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press ESC to exit both processes
            control_queue.put('exit')
            break

    cam_cap.release()
    cv2.destroyAllWindows()

    # Kill the camera process
    subprocess.run(['sudo', 'fuser', '-k', '/dev/video0'])

def main():
    # Directory containing the road video subfolders
    road_video_directory = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA'

    # Control queue to communicate between processes
    control_queue = multiprocessing.Queue()

    # Create separate processes for road video processing and webcam monitoring
    road_process = multiprocessing.Process(target=process_road_videos_worker, args=(road_video_directory, control_queue))
    webcam_process = multiprocessing.Process(target=run_webcam_worker, args=(control_queue,))

    # Start both processes
    road_process.start()
    webcam_process.start()

    # Wait for both processes to complete
    road_process.join()
    webcam_process.join()

if __name__ == "__main__":
    main()
