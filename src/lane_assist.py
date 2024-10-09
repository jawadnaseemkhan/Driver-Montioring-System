import os
import cv2
import numpy as np

<<<<<<< HEAD

# Global variable to store the bonnet Y-coordinate
bonnet_end_y = None

# Mouse callback function to capture the Y-coordinate on click
def mouse_callback(event, x, y, flags, param):
    global bonnet_end_y
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Bonnet end detected at Y-coordinate: {y}")
        bonnet_end_y = y

def adjust_brightness_contrast(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def visualize_roi(frame, roi_vertices):
    overlay = frame.copy()
    cv2.fillPoly(overlay, roi_vertices, (0, 255, 255))
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

=======
# Function to adjust brightness and contrast using histogram equalization
def adjust_brightness_contrast(frame):
    # Convert the frame to YUV color space
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    # Apply histogram equalization to the Y channel (luminance) to enhance brightness
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    
    # Convert back to BGR color space for further processing
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# Function to detect lanes in the given frame
>>>>>>> 2338cfdb0f3b0dfe91d21abea8c911bb6b20be46
def detect_lanes(frame):
    global bonnet_end_y
    
    # Wait until bonnet_end_y is set by the user
    if bonnet_end_y is None:
        return frame
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
<<<<<<< HEAD
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    height, width = frame.shape[:2]

    # Fixed height for the ROI (e.g., 20% of the frame height)
    fixed_roi_height_ratio = 0.2
    roi_top_y = max(0, bonnet_end_y - int(height * fixed_roi_height_ratio))

    # Define the ROI vertices with fixed height
    roi_vertices = np.array([[
        (width * 0.3, bonnet_end_y),
        (width * 0.3, roi_top_y),
        (width * 0.8, roi_top_y),
        (width * 0.8, bonnet_end_y)
    ]], dtype=np.int32)

    frame = visualize_roi(frame, roi_vertices)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    kernel = np.ones((5, 5), np.uint8)
    masked_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel)

=======
    
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
>>>>>>> 2338cfdb0f3b0dfe91d21abea8c911bb6b20be46
    lines = cv2.HoughLinesP(
        closed_edges,
        rho=1,
        theta=np.pi / 180,
<<<<<<< HEAD
        threshold=100,
        minLineLength=50,
        maxLineGap=30
    )

    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 999.0
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    lane_center = calculate_lane_center(left_lines, right_lines, width)
    vehicle_center = width / 2

    feedback = "Centered"
    if lane_center is not None:
        deviation = vehicle_center - lane_center
        if deviation > 20:
            feedback = "Steer Right"
        elif deviation < -20:
            feedback = "Steer Left"
=======
        threshold=120,
        minLineLength=100,
        maxLineGap=50
    )
    
    # If lines are detected, draw them on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green line with thickness 2 pixels
>>>>>>> 2338cfdb0f3b0dfe91d21abea8c911bb6b20be46
    
    # Display the feedback on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Steering Feedback: {feedback}", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return frame

<<<<<<< HEAD
def calculate_lane_center(left_lines, right_lines, frame_width):
    if not left_lines or not right_lines:
        return None

    left_xs = [x1 for x1, y1, x2, y2 in left_lines] + [x2 for x1, y1, x2, y2 in left_lines]
    right_xs = [x1 for x1, y1, x2, y2 in right_lines] + [x2 for x1, y1, x2, y2 in right_lines]

    left_mean = np.mean(left_xs)
    right_mean = np.mean(right_xs)

    lane_center = (left_mean + right_mean) / 2
    return lane_center

def process_road_videos(directory):
    global bonnet_end_y
    
=======
# Function to process all videos in the given directory
def process_road_videos(directory):
    # Loop through folders named 01 to 74
>>>>>>> 2338cfdb0f3b0dfe91d21abea8c911bb6b20be46
    for folder_num in range(1, 75):
        folder_name = f"{folder_num:02d}"
        video_path = os.path.join(directory, folder_name, 'video_garmin.avi')
<<<<<<< HEAD

=======
        
        # Check if the video file exists before processing
>>>>>>> 2338cfdb0f3b0dfe91d21abea8c911bb6b20be46
        if not os.path.exists(video_path):
            print(f"Video not found in folder: {folder_name}")
            continue
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        cv2.namedWindow('Lane Detection')
        cv2.setMouseCallback('Lane Detection', mouse_callback)  # Set up mouse callback to capture clicks

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Break when no more frames are available
            
<<<<<<< HEAD
            frame_with_lanes = detect_lanes(frame)
            
            cv2.imshow('Lane Detection', frame_with_lanes)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"Skipping video: {video_path}")
                break
            elif key == 27:
                print("Terminating all videos.")
=======
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
>>>>>>> 2338cfdb0f3b0dfe91d21abea8c911bb6b20be46
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.release()
    
<<<<<<< HEAD
    cv2.destroyAllWindows()

# Directory containing the video subfolders
#lane_assist_directory = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA'
#process_road_videos(lane_assist_directory)
=======
    # Close all OpenCV windows when done
    cv2.destroyAllWindows()

# Uncomment and set the correct directory to run the script
lane_assist_directory = 'D:/Downloads/DREYEVE_DATA/DREYEVE_DATA'
process_road_videos(lane_assist_directory)
>>>>>>> 2338cfdb0f3b0dfe91d21abea8c911bb6b20be46
