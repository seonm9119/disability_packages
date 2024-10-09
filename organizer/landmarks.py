import os
import cv2
import numpy as np
import mediapipe as mp

POSE_LANDMARKS = [0,7,8,11,12,13,14,15,16,23,24]

FACE_LANDMARKS= [0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 
                 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 
                 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 
                 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 
                 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 
                 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 
                 176, 178, 181, 185, 191, 234, 246, 249, 251, 263, 
                 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 
                 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 
                 317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 
                 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 
                 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 
                 398, 400, 402, 405, 409, 415, 454, 466]


LANDMARKS = {'face': FACE_LANDMARKS, 'pose': POSE_LANDMARKS}
def modify_landmarks(results, landmark='face'):

    if landmark == 'face':
         landmarks = results.face_landmarks

    elif landmark == 'pose':
         landmarks = results.pose_landmarks
         

    if landmarks is not None:
        modified_landmarks = [landmarks.landmark[i] for i in LANDMARKS[landmark]]
        return np.array([[ln.x, ln.y, ln.z] for ln in modified_landmarks])
    
    return None


def extract_landmarks(video_path, file_path, landmark='face'):
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if there are sufficient frames in the video
    if frame_count < 10:
        print(f"Extraction failed because the number of frames is less than 10 : {video_path}")
        cap.release()
        return


    # Initialize array to store landmarks
    modified_landmarks = np.zeros((1, len(LANDMARKS[landmark]), 3))

    with mp_holistic.Holistic(
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as holistic:
    
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            keypoints = modify_landmarks(results, landmark)

            # Append valid keypoints
            if keypoints is not None:
                modified_landmarks = np.vstack([modified_landmarks, keypoints[np.newaxis, :]])

    cap.release()

    if len(modified_landmarks) > 10 and not os.path.exists(file_path):
        np.save(file_path, modified_landmarks[1:])  # Remove initial placeholder

    

    
