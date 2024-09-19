import cv2
import dlib
from scipy.spatial import distance
from fer import FER  # Facial Expression Recognition
import numpy as np

# Initialize the dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/lhome/jawakha/Desktop/Project/Dataset/Shape_Predictor/shape_predictor_68_face_landmarks.dat')

# Initialize FER with GPU (if available)
# detector = FER(mtcnn=True)

# Initialize the FER emotion detector
emotion_detector = FER(mtcnn=True)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])  # Vertical distance
    B = distance.euclidean(eye[2], eye[4])  # Vertical distance
    C = distance.euclidean(eye[0], eye[3])  # Horizontal distance
    
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# Eye landmarks for left and right eye based on 68-point facial landmarks
(L_EYE_START, L_EYE_END) = (42, 48)  # Indices for left eye
(R_EYE_START, R_EYE_END) = (36, 42)  # Indices for right eye

# Thresholds for drowsiness detection
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20  # Number of consecutive frames indicating drowsiness
frame_counter = 0

def detect_drowsiness_from_webcam():
    cap = cv2.VideoCapture(0)  # Using webcam feed
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
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
                if emotion['angry'] > 0.5:
                    cv2.putText(frame, "Aggression", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif emotion['sad'] > 0.5:
                    cv2.putText(frame, "Sadness", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                elif emotion['happy'] > 0.5:
                    cv2.putText(frame, "Excitement", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif emotion['neutral'] > 0.5:
                    cv2.putText(frame, "Neutral", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Driver Monitoring - Drowsiness & Emotion Detection", frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Call the function
detect_drowsiness_from_webcam()
