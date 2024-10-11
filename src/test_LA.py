import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Global variable to store the bonnet Y-coordinate
bonnet_end_y = None

# Mouse callback function to capture the Y-coordinate on click
def mouse_callback(event, x, y, flags, param):
    global bonnet_end_y
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Bonnet end detected at Y-coordinate: {y}")
        bonnet_end_y = y

# Function to adjust brightness and contrast using histogram equalization
def adjust_brightness_contrast(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

# Function to visualize the region of interest (ROI)
def visualize_roi(frame, roi_vertices):
    overlay = frame.copy()
    cv2.fillPoly(overlay, roi_vertices, (0, 255, 255))  # Yellow polygon for the ROI
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame

# Function to detect lanes in the given frame
def detect_lanes(frame):
    global bonnet_end_y
    
    # Wait until bonnet_end_y is set by the user
    if bonnet_end_y is None:
        return frame, "Waiting for bonnet end click"
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    height, width = frame.shape[:2]

    # Fixed height for the ROI (e.g., 20% of the frame height)
    fixed_roi_height_ratio = 0.1
    roi_top_y = max(0, bonnet_end_y - int(height * fixed_roi_height_ratio))

    # Define the ROI vertices with a fixed height
    roi_vertices = np.array([[
        (width * 0.225, bonnet_end_y),
        (width * 0.225, roi_top_y),
        (width * 0.75, roi_top_y),
        (width * 0.75, bonnet_end_y)
    ]], dtype=np.int32)

    frame = visualize_roi(frame, roi_vertices)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    kernel = np.ones((5, 5), np.uint8)
    masked_edges = cv2.morphologyEx(masked_edges, cv2.MORPH_CLOSE, kernel)

    # Apply the Hough Transform to detect lines in the closed edge image
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
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

    # Display the feedback and center point on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Steering Feedback: {feedback}", (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw the lane center and vehicle center for visualization
    if lane_center is not None:
        cv2.circle(frame, (int(lane_center), int(roi_top_y + (bonnet_end_y - roi_top_y) / 2)), 5, (255, 0, 0), -1)  # Blue for lane center
        cv2.circle(frame, (int(vehicle_center), int(roi_top_y + (bonnet_end_y - roi_top_y) / 2)), 5, (0, 0, 255), -1)  # Red for vehicle center
    
    return frame, feedback

# Function to calculate the lane center
def calculate_lane_center(left_lines, right_lines, frame_width):
    if not left_lines or not right_lines:
        return None

    left_xs = [x1 for x1, y1, x2, y2 in left_lines] + [x2 for x1, y1, x2, y2 in left_lines]
    right_xs = [x1 for x1, y1, x2, y2 in right_lines] + [x2 for x1, y1, x2, y2 in right_lines]

    left_mean = np.mean(left_xs)
    right_mean = np.mean(right_xs)

    lane_center = (left_mean + right_mean) / 2
    return lane_center

# Function to process all videos in the given directory and plot feedback
def process_road_videos(directory):
    global bonnet_end_y

    # Variables to track steering states
    feedback_counts = {"Centered": 0, "Steer Left": 0, "Steer Right": 0, "Waiting for bonnet end click": 0}
    
    for folder_num in range(1, 5):
        folder_name = f"{folder_num:02d}"
        video_path = os.path.join(directory, folder_name, 'video_garmin.avi')

        if not os.path.exists(video_path):
            print(f"Video not found in folder: {folder_name}")
            continue
        
        cap = cv2.VideoCapture(video_path)
        
        cv2.namedWindow('Lane Detection')
        cv2.setMouseCallback('Lane Detection', mouse_callback)  # Set up mouse callback to capture clicks

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Break when no more frames are available
            
            frame_with_lanes, feedback = detect_lanes(frame)
            
            feedback_counts[feedback] += 1  # Track feedback for each frame

            cv2.imshow('Lane Detection', frame_with_lanes)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"Skipping video: {video_path}")
                break
            elif key == 27:
                print("Terminating all videos.")
                cap.release()
                cv2.destroyAllWindows()
                return
        
        cap.release()
    
    cv2.destroyAllWindows()

    # Plot the steering feedback as a bar chart
    plot_feedback(feedback_counts)

# Function to plot steering feedback
def plot_feedback(feedback_counts):
    labels = list(feedback_counts.keys())
    values = list(feedback_counts.values())

    plt.bar(labels, values, color=['blue', 'green', 'red', 'gray'])
    plt.xlabel("Steering Feedback")
    plt.ylabel("Count (Number of Frames)")
    plt.title("Steering Feedback Distribution")
    plt.show()

# Example usage:
# Directory containing the video subfolders
lane_assist_directory = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA'
process_road_videos(lane_assist_directory)
