import os
import cv2
import numpy as np

# Function to adjust brightness and contrast using histogram equalization
def adjust_brightness_contrast(frame):
    # Convert the frame to YUV color space
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    # Apply histogram equalization to the Y channel (luminance) to enhance brightness
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    
    # Convert back to BGR color space for further processing
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# Function to detect lanes in the given frame
def detect_lanes(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detector with specific thresholds to find edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Get frame dimensions to define a region of interest (ROI)
    height, width = frame.shape[:2]
    
    # Define the vertices of the polygonal ROI, focusing on the road area
    roi_vertices = np.array([[
        (int(width * 0.1), int(height * 0.7)),  # Bottom-left
        (int(width * 0.4), int(height * 0.5)),  # Upper-left
        (int(width * 0.6), int(height * 0.5)),  # Upper-right
        (int(width * 0.9), int(height * 0.7))   # Bottom-right
    ]], dtype=np.int32)
    
    # Highlight ROI vertices by drawing small red dots on the frame
    for vertex in roi_vertices[0]:
        cv2.circle(frame, tuple(vertex), 5, (0, 0, 255), -1)  # Red dot with radius 5 pixels
    
    # Create a mask for the ROI and apply it to the edges detected by Canny
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # If no edges are detected in the ROI, return the original frame
    if np.sum(masked_edges) == 0:
        print("No edges detected in this frame.")
        return frame
    
    # Use morphological closing to fill small gaps in the detected edges
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel)
    
    # Apply the Hough Transform to detect lines in the closed edge image
    lines = cv2.HoughLinesP(
        closed_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=100,
        maxLineGap=50
    )
    
    # If lines are detected, draw them on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green line with thickness 2 pixels
    
    return frame

# Function to process all videos in the given directory
def process_road_videos(directory):
    # Loop through folders named 01 to 74
    for folder_num in range(1, 75):
        folder_name = f"{folder_num:02d}"  # Format folder name with leading zeros
        video_path = os.path.join(directory, folder_name, 'video_garmin.avi')
        
        # Check if the video file exists before processing
        if not os.path.exists(video_path):
            print(f"Video not found in folder: {folder_name}")
            continue
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Break when no more frames are available
            
            # Apply lane detection on the current frame
            frame_with_lanes = detect_lanes(frame)
            
            # Overlay the video path as text on the frame (top left corner)
            cv2.putText(
                frame_with_lanes,
                f"Video: {video_path}",
                (10, 30),  # Position: 10 pixels from the left, 30 pixels from the top
                cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                0.6,  # Font scale (size)
                (0, 0, 255),  # White text
                2,  # Thickness of the text
                cv2.LINE_AA  # Anti-aliased text for smoother edges
            )
            
            # Display the frame with the detected lanes and video path
            cv2.imshow('Lane Detection', frame_with_lanes)
            
            # Check for user input to either skip the video or quit processing
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to skip the current video
                print(f"Skipping video: {video_path}")
                break
            elif key == 27:  # Press 'ESC' to stop processing all videos
                print("Terminating all video processing.")
                cap.release()
                cv2.destroyAllWindows()
                return
        
        # Release the video capture object after processing each video
        cap.release()
    
    # Close all OpenCV windows when done
    cv2.destroyAllWindows()

# Uncomment and set the correct directory to run the script
lane_assist_directory = 'D:/Downloads/DREYEVE_DATA/DREYEVE_DATA'
process_road_videos(lane_assist_directory)
