import cv2
import numpy as np

def extract_optical_flow(video_path, file_path, img_size=224):

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Cannot read the video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, (img_size, img_size)) 
    flow_list = []

    while True:
        ret, next_frame = cap.read()
        if not ret:
            break

        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.resize(next_gray, (img_size, img_size)) 
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        

        # Flow 벡터의 크기 및 각도 계산
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

         # Create optical flow map (Hue = angle, Value = magnitude)
        hsv = np.zeros((img_size, img_size, 3), dtype=np.uint8)  # Create a 3-channel HSV image
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2  # Convert angle to hue
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Normalize magnitude
        flow_map = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


        flow_list.append(flow_map)

        prev_gray = next_gray

    flow_array = np.array(flow_list)

    # NumPy 배열로 저장
    np.save(file_path, flow_array)

    cap.release()
