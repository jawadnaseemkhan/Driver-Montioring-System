import os
import cv2
import numpy as np

def adjust_brightness_contrast(frame):
    # Convert to YUV color space to enhance brightness and contrast
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # Equalize the histogram of the Y channel
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def detect_lanes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using Canny with adjusted thresholds
    edges = cv2.Canny(blurred, 50, 150)  # Adjusted lower threshold
    
    # Define a higher region of interest (ROI) to avoid bonnet and focus on the road
    height, width = frame.shape[:2]
    
    # Raise the bottom portion of the ROI to avoid detecting the car's bonnet
    roi_vertices = np.array([[
        (width * 0.1, height * 0.7),  # Bottom-left corner (moved higher)
        (width * 0.4, height * 0.5),  # Upper-left corner
        (width * 0.6, height * 0.5),  # Upper-right corner
        (width * 0.9, height * 0.7)   # Bottom-right corner (moved higher)
    ]], dtype=np.int32)

    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Check if there are any detected edges
    if masked_edges is None or np.sum(masked_edges) == 0:
        print("No edges detected.")
        return frame  # Return the original frame if no edges were detected
    
    # Apply morphological operations (closing to fill gaps)
    kernel = np.ones((5, 5), np.uint8)  
    masked_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel)
    
    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120, 
        minLineLength=100, 
        maxLineGap=50
    )
    
    # Draw detected lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame

def process_road_videos(directory):
    # Loop through all numbered folders (01 to 74)
    for folder_num in range(1, 75):
        folder_name = f"{folder_num:02d}"  # Format folder name with leading zeros
        video_path = os.path.join(directory, folder_name, 'video_garmin.avi')

        # Check if the video file exists before processing
        if not os.path.exists(video_path):
            print(f"Video not found in folder: {folder_name}")
            continue
        
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect lanes in the frame
            frame_with_lanes = detect_lanes(frame)
            
            # Display the frame with detected lanes
            cv2.imshow('Lane Detection', frame_with_lanes)
            
            # Check for 'q' or 'ESC' key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # 'q' key to stop the current video and move to the next
                print(f"Skipping video: {video_path}")
                break
            elif key == 27:  # ESC key to stop all videos
                print("Terminating all videos.")
                cap.release()
                cv2.destroyAllWindows()
                return  # Exit from the process_videos function to stop all videos
        
        # Release the video capture object after processing each video
        cap.release()
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()

# # Directory containing the video subfolders
# lane_assist_directory = '/lhome/jawakha/Desktop/University/Thesis/Dataset/DREYEVE_DATA'
# process_road_videos(lane_assist_directory)
