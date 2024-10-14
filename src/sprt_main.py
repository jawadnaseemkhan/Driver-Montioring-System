import os
import cv2
from lane_assist import process_road_videos
#from facial_expressions import detect_drowsiness_and_emotions
from faceExpression import detect_drowsiness_and_emotions
def main():
    # Directory containing the road video subfolders
    road_video_directory = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA'
    
    # Process road videos for lane detection
    process_road_videos(road_video_directory)

    # Start the webcam for driver monitoring
    cap = cv2.VideoCapture(0)  # Open the webcam
    frame_counter = 0

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Pass the frame and frame counter to detect_drowsiness_and_emotions
        detect_drowsiness_and_emotions(frame, frame_counter)

        # Display the frame
        cv2.imshow("Driver Monitoring", frame)

        frame_counter += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
