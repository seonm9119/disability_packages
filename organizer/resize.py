import cv2
import os

def _resize_video(video_path, file_path):
    cap = cv2.VideoCapture(video_path)

    # 영상 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = 224
    frame_height = 224

    # 저장할 영상 설정 (코덱, FPS, 사이즈)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (frame_width, frame_height))

    # 프레임 리사이즈 후 저장
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        # 프레임 리사이즈
        resized_frame = cv2.resize(frame, (frame_width, frame_height))
    
        # 리사이즈된 프레임 저장
        out.write(resized_frame)

    # 리소스 해제
    cap.release()
    out.release()

def resize_video(source_folder):
    # Create a subdirectory 'audio' in the source folder if it doesn't already exist
    path = os.path.join(source_folder, 'resize')
    if not os.path.exists(path):
        os.makedirs(path)

    mp4_files = os.listdir(f"{source_folder}/video")

    for _path in mp4_files:
        video_path = os.path.join(source_folder, 'video', _path)
        file_path = os.path.join(path, f"{_path.split('.')[0]}.mp4")

        if not os.path.exists(file_path):
            _resize_video(video_path, file_path)

    print(f"Video resize completed for the folder: {source_folder}")


