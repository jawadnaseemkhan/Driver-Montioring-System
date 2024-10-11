import json
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# Function to draw lanes on image
def draw_lanes(image_path, lanes, h_samples):
    # Read the image
    image = cv2.imread(image_path)
    
    # Iterate over each lane
    for lane in lanes:
        for x, y in zip(lane, h_samples):
            if x > 0:  # Ignore points marked with -2 (no lane point)
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

    # Display the image with lanes
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Lane Detection')
    plt.show()

# Function to generate lane masks
def create_lane_mask(image_shape, lanes, h_samples):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Draw lanes on the mask
    for lane in lanes:
        for x, y in zip(lane, h_samples):
            if x > 0:  # Ignore invalid points
                cv2.circle(mask, (x, y), 5, 255, -1)  # White circle on black background
    return mask

# Load the JSON file line by line
with open('/lhome/jawakha/Desktop/Project/data/test_label_new.json', 'r') as f:
    data = [json.loads(line) for line in f]  # Load each line separately

# Define base directory where the images are stored
image_base_dir = '/lhome/jawakha/Desktop/Project/data/TUSimple/test_set/clips/'

# Loop through dataset and visualize lanes, and save masks
output_mask_dir = "/lhome/jawakha/Desktop/Project/masks/"
os.makedirs(output_mask_dir, exist_ok=True)  # Ensure mask output directory exists

for entry in data:
    # Extract the relative image path from the JSON entry
    raw_file = entry['raw_file']  # e.g., 'clips/0530/1492626047222176976_0/1.jpg'
    
    # Construct the full image path
    img_path = os.path.join(image_base_dir, raw_file)
    
    # Get the lane points and h_samples
    lanes = entry['lanes']        # Lane x-coordinates for each lane
    h_samples = entry['h_samples']  # y-coordinates for lane points
    
    # Read image to get shape
    image = cv2.imread(img_path)
    
    if image is not None:
        # Draw lanes on the image for visualization
        draw_lanes(img_path, lanes, h_samples)

        # Create lane mask for the image
        mask = create_lane_mask(image.shape, lanes, h_samples)
        
        # Save mask
        mask_name = os.path.basename(raw_file).replace('.jpg', '_mask.png')
        mask_path = os.path.join(output_mask_dir, mask_name)
        cv2.imwrite(mask_path, mask)
        print(f"Mask saved at {mask_path}")

    # Uncomment to process one image for testing purposes
    # break
