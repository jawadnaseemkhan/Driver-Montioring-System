# **Integrated Driver Monitoring and Lane Detection System**

## **Overview**
This project combines **driver facial expression analysis** and **road maneuvering assessment** to enhance driving performance and safety. By integrating real-time **facial expression recognition** with advanced **lane detection techniques**, the system provides insights into driver behavior and road conditions, enabling a more secure driving experience.

## **Key Features**
1. **Driver Monitoring**:
   - Real-time detection of drowsiness and emotional states using facial landmarks and emotion recognition.
   - Eye Aspect Ratio (EAR) calculation for drowsiness detection.
   - Webcam integration for live monitoring.

2. **Lane Detection**:
   - Advanced lane detection using semantic segmentation and computer vision techniques.
   - Resilience to varied weather and lighting conditions, including night, rain, and low visibility.
   - Region of Interest (ROI) adjustment for dynamic environments.

3. **Integrated System**:
   - Simultaneous road video processing and driver monitoring.
   - Real-time overlay of the webcam feed on road video for comprehensive visualization.

---


### **Components**
- **Driver Monitoring**:
  - Facial expression analysis using **FER** and **dlib**.
  - Detection of fatigue and emotions like excitement, sadness, or drunkenness.

- **Lane Detection**:
  - Road video analysis with **DeepLabV3** semantic segmentation.
  - Edge detection and Hough Transform for lane identification.

- **Integration**:
  - Combined feed from the road video and webcam for real-time analysis.

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries & Tools**:
  - OpenCV
  - TensorFlow / PyTorch (for DeepLabV3)
  - Dlib
  - FER (Facial Expression Recognition)
  - CVAT (for annotation)
- **Dataset**:
  - [DREYEVE Dataset](https://paperswithcode.com/dataset/dr-eye-ve) for road videos and frames.

---

## **Setup & Installation**
### **Requirements**
- Python 3.8+
- Virtual Environment (optional but recommended)
- Required Libraries:
  ```bash
  pip install -r requirements.txt


### **Project Structure**
        ├── data/
    │   ├── videos/             # Input road videos
    │   ├── frames/             # Extracted frames for annotation
    ├── models/
    │   ├── checkpoints/        # Pretrained model weights
    ├── scripts/
    │   ├── lane_assist.py      # Lane detection script
    │   ├── facial_expressions.py  # Facial expression recognition
    │   ├── main.py             # Integrated system script
    ├── output/
    │   ├── segmentation/       # Lane segmentation results
    │   ├── logs/               # Log files
    ├── README.md               # Project documentation
