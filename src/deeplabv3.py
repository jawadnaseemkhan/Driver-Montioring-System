import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load a pretrained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Define the transformation
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define paths
base_folder = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA/'  # Your base folder containing the 74 folders
output_folder = '/lhome/jawakha/Desktop/Project/Dataset/outputs/'  # Folder to save the outputs
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all 74 folders
for folder_num in range(1, 75):
    folder_name = f"{folder_num:02d}"
    folder_path = os.path.join(base_folder, folder_name, 'resized_frames')
    output_subfolder = os.path.join(output_folder, folder_name)
    
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # Loop through all images in the resized_frames folder
    for image_file in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_file)
        
        # Ensure it's a valid image file
        if image_file.endswith('.png') or image_file.endswith('.jpg'):
            try:
                input_image = Image.open(image_path).convert('RGB')  # Convert to RGB in case of grayscale images
                input_tensor = preprocess(input_image).unsqueeze(0)
                
                # Perform inference
                with torch.no_grad():
                    output = model(input_tensor)['out'][0]
                
                # Save output segmentation mask (as an example, here saving as PNG)
                output_image = torch.argmax(output, 0).byte().cpu().numpy()  # Convert to numpy array
                output_image_pil = Image.fromarray(output_image)  # Convert to PIL Image
                output_image_path = os.path.join(output_subfolder, f"output_{image_file}")
                output_image_pil.save(output_image_path)

                print(f"Processed: {image_file}")
            
            except Exception as e:
                print(f"Failed to process {image_file}: {str(e)}")

print("Processing complete!")
