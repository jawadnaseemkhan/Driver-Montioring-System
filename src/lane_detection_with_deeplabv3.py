import torch
from torchvision import models, transforms
import cv2
import os
import json
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Step 1: Define TuSimple Dataset
class TuSimpleDataset(Dataset):
    def __init__(self, image_paths, lane_annotations, transform=None):
        self.image_paths = image_paths
        self.lane_annotations = lane_annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])

        if image is None:
            print(f"Warning: Failed to load image {self.image_paths[idx]}")
            return None, None  # Return None if image fails to load

        lanes = self.lane_annotations[idx]  # Lane annotations

        # Create a mask from lane points (assuming lanes are annotated as x,y points)
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for lane in lanes:
            for x, y in zip(lane, range(len(lane))):  # y-coordinates are often fixed heights in TuSimple
                if x > 0:  # Ignore points without valid x-coordinates
                    cv2.circle(mask, (x, y), 5, 255, -1)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(mask, dtype=torch.long)

# Step 2: Define Image Transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Step 3: Load JSON and Extract Image Paths and Annotations
clips_path = '/lhome/jawakha/Desktop/Project/data/TUSimple/train_set/clips/'
json_files = [
    '/lhome/jawakha/Desktop/Project/data/TUSimple/train_set/label_data_0313.json',
    '/lhome/jawakha/Desktop/Project/data/TUSimple/train_set/label_data_0531.json',
    '/lhome/jawakha/Desktop/Project/data/TUSimple/train_set/label_data_0601.json'
]

# Load JSON annotations line by line
image_paths = []
lane_annotations = []

for json_file in json_files:
    with open(json_file, 'r') as f:
        for line in f:
            entry = json.loads(line)  # Parse each line as JSON
            img_path = os.path.join(clips_path, entry['raw_file'].lstrip('/'))  # Avoid extra slashes
            print(f"Image path: {img_path}")  # Debugging: print image paths
            lanes = entry['lanes']  # Lane annotations
            image_paths.append(img_path)
            lane_annotations.append(lanes)

# Step 4: Create DataLoader
train_dataset = TuSimpleDataset(image_paths, lane_annotations, transform=transform)

def custom_collate(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if len(batch) == 0:  # If batch is empty, return empty tensors
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)

# Step 5: Load Pre-trained DeepLabV3 Model
model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)  # Adjust output layer for binary lane segmentation
model = model.cpu()  # Use CPU since there's no CUDA support
model.train()

# Step 6: Training Setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Step 7: Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    epoch_loss = 0
    for images, labels in train_loader:
        if len(images) == 0 or len(labels) == 0:  # Skip empty batches
            continue

        images, labels = images.cpu(), labels.cpu()
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, labels.unsqueeze(1).float())  # Convert to match model output shape
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}')

# Now you can run this code to train the model.
