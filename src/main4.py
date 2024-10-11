import os
import cv2
import numpy as np
from lane_assist import detect_lanes, bonnet_end_y, visualize_roi, adjust_brightness_contrast
from facial_expressions import detect_drowsiness_and_emotions, frame_counter

# Mouse callback function to capture the Y-coordinate on click
def mouse_callback(event, x, y, flags, param):
    global bonnet_end_y
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Bonnet end detected at Y-coordinate: {y}")
        bonnet_end_y = y

def main():
    global bonnet_end_y, frame_counter
    # Directory containing the road video subfolders
    road_video_directory = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA'
    
    folder_num = 1  # Start with the first folder for example
    video_path = os.path.join(road_video_directory, f"{folder_num:02d}", 'video_garmin.avi')
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Open the road video capture and webcam
    road_cap = cv2.VideoCapture(video_path)
    cam_cap = cv2.VideoCapture(0)  # Webcam feed
    
    bonnet_end_y = None  # Reset for each video
    
    cv2.namedWindow('Lane Detection with Camera', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Lane Detection with Camera', mouse_callback)  # Set up mouse callback to capture clicks
    
    while road_cap.isOpened() and cam_cap.isOpened():
        ret_road, road_frame = road_cap.read()
        ret_cam, cam_frame = cam_cap.read()
        
        if not ret_road:
            print("Road video ended or cannot be read.")
            break
        
        if not ret_cam:
            print("Cannot read from the camera feed.")
            break
        
        # Wait until bonnet_end_y is set
        if bonnet_end_y is None:
            combined_frame = road_frame.copy()
            cv2.putText(combined_frame, "Click on the bonnet", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Lane Detection with Camera', combined_frame)
            cv2.waitKey(1)
            continue  # Skip processing until bonnet_end_y is set
        
        # Process road frame for lane detection
        road_frame_with_lanes = detect_lanes(road_frame)
        
        # Debugging: Check if lane detection is working
        print("Lane detection completed")
        
        # **Display ROI Overlay**: Ensure the ROI is drawn on the road frame
        height, width = road_frame.shape[:2]
        fixed_roi_height_ratio = 0.2
        roi_top_y = max(0, bonnet_end_y - int(height * fixed_roi_height_ratio))
        roi_vertices = np.array([[
            (width * 0.2, bonnet_end_y),
            (width * 0.2, roi_top_y),
            (width * 0.8, roi_top_y),
            (width * 0.8, bonnet_end_y)
        ]], dtype=np.int32)
        road_frame_with_lanes = visualize_roi(road_frame_with_lanes, roi_vertices)

        # Check if lane center is being calculated correctly (add this to `detect_lanes` function)
        print("Lane Center Detection Status: OK")

        # Process camera frame for drowsiness and emotions
        cam_frame_processed, frame_counter = detect_drowsiness_and_emotions(cam_frame, frame_counter)
        
        # Resize the camera frame to be smaller (e.g., 320x180) for the corner display
        cam_frame_small = cv2.resize(cam_frame_processed, (320, 180))
        
        # Ensure that the camera feed is correctly placed within the boundaries of the road frame
        if height >= 180 and width >= 320:
            road_frame_with_lanes[height-180:height, 0:320] = cam_frame_small
        
        # Display the combined frame in a maximized window
        cv2.imshow('Lane Detection with Camera', road_frame_with_lanes)
        cv2.setWindowProperty('Lane Detection with Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Maximize the window
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Move to the next video or reset
            folder_num += 1
            video_path = os.path.join(road_video_directory, f"{folder_num:02d}", 'video_garmin.avi')
            road_cap.release()
            road_cap = cv2.VideoCapture(video_path)  # Reload new video
        elif key == 27:  # ESC key pressed
            print("Terminating...")
            road_cap.release()
            cam_cap.release()
            cv2.destroyAllWindows()
            return
        
    road_cap.release()
    cam_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
