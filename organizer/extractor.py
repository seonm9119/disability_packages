import os
from moviepy.editor import VideoFileClip
from organizer.utils import filter_unique_audio


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
    
    # Filter unique .mp4 files to avoid redundant audio extraction
    unique_files = filter_unique_audio(mp4_files)

    # Extract and save audio from each unique .mp4 file
    for file_path in unique_files:
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
