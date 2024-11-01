import os
from moviepy.editor import VideoFileClip
from .landmarks import extract_landmarks
from .optical_flow import extract_optical_flow


def flow_extractor(source_folder):
    path = os.path.join(source_folder, 'flow')
    if not os.path.exists(path):
        os.makedirs(path)

    mp4_files = os.listdir(f"{source_folder}/video")

    for _path in mp4_files:
        video_path = os.path.join(source_folder, 'video', _path)
        file_path = os.path.join(path, f"{_path.split('.')[0]}.npy")

        if not os.path.exists(file_path):
            extract_optical_flow(video_path, file_path)
    
    print(f"Optical Flow extraction completed for the folder: {source_folder}")



def landmarks_extractor(source_folder, landmark='face'):
    # Create a subdirectory 'audio' in the source folder if it doesn't already exist
    path = os.path.join(source_folder, landmark)
    if not os.path.exists(path):
        os.makedirs(path)

    mp4_files = os.listdir(f"{source_folder}/video")

    for _path in mp4_files:
        video_path = os.path.join(source_folder, 'video', _path)
        file_path = os.path.join(path, f"{_path.split('.')[0]}.npy")

        if not os.path.exists(file_path):
            extract_landmarks(video_path, file_path, landmark)


def face_extractor(source_folder):
    landmarks_extractor(source_folder, landmark='face')
    print(f"Face extraction completed for the folder: {source_folder}")
    

def audio_extractor(source_folder):
    """
    Extracts audio from all .mp4 video files in the source folder.

    Parameters:
        source_folder (str): Path to the folder containing a 'video' subfolder with .mp4 files.
    """
    
    # Create a subdirectory 'audio' in the source folder if it doesn't already exist
    audio_path = os.path.join(source_folder, 'audio')
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)

    # List all files in the 'video' subfolder of the source folder
    mp4_files = os.listdir(f"{source_folder}/video")
    
    # Extract and save audio from each unique .mp4 file
    for file_path in mp4_files:
        file_name = file_path.split('.')[0]
        

        save_file = os.path.join(audio_path, f"{file_name}.wav")

        # If the .wav file does not exist, extract audio from the corresponding .mp4 file
        if not os.path.exists(save_file):
            video_path = os.path.join(source_folder, 'video', file_path)
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio

            # Save the extracted audio as a .wav file if the audio exists
            if audio_clip is not None:
                audio_clip.write_audiofile(save_file, verbose=False, logger=None)

    print(f"Audio extraction completed for the folder: {source_folder}")

