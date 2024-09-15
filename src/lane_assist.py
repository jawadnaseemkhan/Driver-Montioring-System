import os
import cv2
import numpy as np

def detect_lanes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform edge detection using Canny with adjusted thresholds
    edges = cv2.Canny(blurred, 100, 200)  # Increased lower threshold to reduce noise
    
    # Define a tighter region of interest (ROI) to avoid cars and other objects
    height, width = frame.shape[:2]
    
    # Narrower ROI to focus only on the lane marking areas ahead of the car
    roi_vertices = np.array([[
        (width * 0.3, height * 0.7),  # Bottom-left corner of the ROI
        (width * 0.4, height * 0.5),  # Upper-left corner of the ROI
        (width * 0.6, height * 0.5),  # Upper-right corner of the ROI
        (width * 0.7, height * 0.7)   # Bottom-right corner of the ROI
    ]], dtype=np.int32)
    
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough Transform to detect lines (tuned parameters)
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,  # Increased to focus on stronger lines
        minLineLength=180,  # Increased to focus on longer lane lines
        maxLineGap=50  # Reduced to focus on more continuous lines
    )
    
    # Draw detected lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    return frame

def process_videos(directory):
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

# Directory containing the video subfolders
directory = '/lhome/jawakha/Desktop/University/Thesis/Dataset/DREYEVE_DATA'
process_videos(directory)
