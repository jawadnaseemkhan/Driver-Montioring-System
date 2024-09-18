import os
from lane_assist import process_road_videos
from facial_expressions import detect_drowsiness_from_webcam

def main():
    # Directory containing the road video subfolders
    road_video_directory = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA'
    
    # Process road videos for lane detection
    process_road_videos(road_video_directory)

    # Start the webcam for driver monitoring
    detect_drowsiness_from_webcam()

if __name__ == "__main__":
    main()
