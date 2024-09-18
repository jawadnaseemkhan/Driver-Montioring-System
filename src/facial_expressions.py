import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Constants for drowsiness detection
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 48

# Load Dlib's pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/lhome/jawakha/Desktop/Project/Dataset/Shape_Predictor/shape_predictor_68_face_landmarks.dat')

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_drowsiness_from_webcam():
    eye_closed_frames = 0
    cap_webcam = cv2.VideoCapture(0)  # Use the webcam

    while cap_webcam.isOpened():
        ret, frame = cap_webcam.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            landmarks = predictor(gray, face)
            
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Draw the eyes on the webcam feed
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
            
            if ear < EYE_AR_THRESH:
                eye_closed_frames += 1
                if eye_closed_frames >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                eye_closed_frames = 0  # Reset counter if eyes are open
            
        # Display the webcam feed
        cv2.imshow('Driver Monitoring (Webcam)', frame)
        
        # Exit the webcam if 'q' or 'ESC' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap_webcam.release()
    cv2.destroyAllWindows()
