import cv2
import dlib
from scipy.spatial import distance
from fer import FER  
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Initialize the dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/lhome/jawakha/Desktop/Project/Dataset/Shape_Predictor/shape_predictor_68_face_landmarks.dat')

# Initialize the FER emotion detector
emotion_detector = FER(mtcnn=True)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # Vertical distance
    B = distance.euclidean(eye[2], eye[4])  # Vertical distance
    C = distance.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmarks for left and right eye based on 68-point facial landmarks
(L_EYE_START, L_EYE_END) = (42, 48)  # Indices for left eye
(R_EYE_START, R_EYE_END) = (36, 42)  # Indices for right eye

# Thresholds for drowsiness detection
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20  # Number of consecutive frames indicating drowsiness

# Initialize frame counter outside the function
frame_counter = 0

# Store actual and predicted labels for evaluation
y_true = []  # Ground truth labels
y_pred = []  # Predicted labels

# Define emotion labels for evaluation
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

def detect_drowsiness_and_emotions(frame, frame_counter, true_emotion=None):
    global y_true, y_pred  # Use global variables to store true/predicted labels
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        
        # Get the coordinates for the left and right eye landmarks
        left_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(L_EYE_START, L_EYE_END)])
        right_eye = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(R_EYE_START, R_EYE_END)])
        
        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0  # Average EAR
        
        # Check for drowsiness
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES:
                cv2.putText(frame, "Drowsiness Detected!", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            frame_counter = 0  # Reset frame counter if not drowsy
        
        # Draw eye landmarks
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        
        # Emotion detection on the face region
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        face_region = frame[y1:y2, x1:x2]
        emotion_results = emotion_detector.detect_emotions(face_region)
        
        if emotion_results:
            emotion = emotion_results[0]['emotions']
            # Determine the dominant emotion
            emotion_label = max(emotion, key=emotion.get)
            cv2.putText(frame, emotion_label.capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Append the actual and predicted emotions for evaluation
            if true_emotion:
                y_true.append(true_emotion)  # Add ground truth emotion
                y_pred.append(emotion_label.capitalize())  # Add predicted emotion
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame, frame_counter

# Open video capture (from webcam or file)
video_capture = cv2.VideoCapture(0)  # Use '0' for webcam or replace with video file path

frame_counter = 0
true_emotion = "Happiness"  # Replace this with the actual label if available

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    
    frame, frame_counter = detect_drowsiness_and_emotions(frame, frame_counter, true_emotion)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# After running the detection, generate evaluation metrics

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=emotion_labels)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix for Facial Expression Recognition')
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.show()

# Classification Report (Precision, Recall, F1-Score)
print(classification_report(y_true, y_pred, target_names=emotion_labels))
