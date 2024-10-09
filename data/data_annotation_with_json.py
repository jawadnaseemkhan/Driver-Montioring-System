import json
import cv2
import matplotlib.pyplot as plt
import os

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

# Load the JSON file line by line
with open('/lhome/jawakha/Desktop/Project/data/test_label_new.json', 'r') as f:
    data = [json.loads(line) for line in f]  # Load each line separately

# Loop through dataset and visualize lanes
for entry in data:
    # Image path (adjust the path if necessary)
    img_path = os.path.join('/lhome/jawakha/Desktop/Project/data/TUSimple', entry['raw_file'])
    
    # Get the lane points and h_samples
    lanes = entry['lanes']        # Lane x-coordinates for each lane
    h_samples = entry['h_samples']  # y-coordinates for lane points

    # Draw lanes on the image
    draw_lanes(img_path, lanes, h_samples)

    # Uncomment the following line to stop after one image for testing purposes
    # break
