import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load a pretrained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Define the transformation for preprocessing the images
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define paths
base_folder = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA/'  # Your base folder containing the 74 folders
output_folder = '/lhome/jawakha/Desktop/Project/Dataset/outputs/'  # Folder to save the outputs

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define a function to overlay the segmentation mask on the original image
def overlay_segmentation(original_image, segmentation_mask, alpha=0.6):
    """
    Overlay the segmentation mask on the original image with transparency.
    """
    original_image_np = np.array(original_image)
    
    # Assuming the lanes are labeled with a specific class (e.g., class 1)
    colored_mask = np.zeros_like(original_image_np)
    colored_mask[segmentation_mask == 1] = [0, 255, 0]  # Green color for detected lanes

    # Overlay the segmentation mask on the original image
    overlay_image = cv2.addWeighted(original_image_np, 1 - alpha, colored_mask, alpha, 0)
    
    # Display the overlay
    plt.imshow(overlay_image)
    plt.axis('off')
    plt.show()

# Loop through all 74 folders
for folder_num in range(1, 75):
    folder_name = f"{folder_num:02d}"
    folder_path = os.path.join(base_folder, folder_name, 'resized_frames')
    output_subfolder = os.path.join(output_folder, folder_name)
    
    # Create output subfolder if it doesn't exist
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # Loop through all images in the resized_frames folder
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        
        # Ensure it's a valid image file (check extensions)
        if image_file.endswith('.png') or image_file.endswith('.jpg'):
            try:
                # Load and preprocess the input image
                input_image = Image.open(image_path).convert('RGB')
                input_tensor = preprocess(input_image).unsqueeze(0)
                
                # Perform inference with the model
                with torch.no_grad():
                    output = model(input_tensor)['out'][0]
                
                # Process the output segmentation mask
                output_image = torch.argmax(output, 0).byte().cpu().numpy()  # Convert to numpy array
                
                # Save the segmentation mask
                output_image_pil = Image.fromarray(output_image)  # Convert to PIL Image
                output_image_path = os.path.join(output_subfolder, f"output_{image_file}")
                output_image_pil.save(output_image_path)

                print(f"Processed: {image_file}")
                
                # Visual inspection: Overlay the segmentation mask on the original image
                overlay_segmentation(input_image, output_image)

            except Exception as e:
                print(f"Failed to process {image_file}: {str(e)}")

print("Processing complete!")
