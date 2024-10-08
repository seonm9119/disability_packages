import os
import glob
from moviepy.editor import VideoFileClip

def audio_extractor(source_folder, save_path=None):
    """
    Extracts audio from all .mp4 video files in the specified source folder.

    Parameters:
        source_folder (str): The path to the folder containing .mp4 files.
        save_path (str, optional): The path to the folder where extracted audio files will be saved. 
                                    If None, audio files will be saved in the same location as the videos.
    """

    # Use glob to find all .mp4 files in the specified source folder
    mp4_files = glob.glob(os.path.join(source_folder, '*.mp4'))
    for file_path in mp4_files:

        # Determine the save file path based on the provided save_path
        save_file = f"{file_path.split('.')[0]}.wav" if save_path is None else os.path.join(save_path, 'sample.wav')

        # Extract audio if the file doesn't already exist
        if not os.path.exists(save_file):
            video_clip = VideoFileClip(file_path)
            audio_clip = video_clip.audio

            if audio_clip is not None:
                audio_clip.write_audiofile(save_file, verbose=False, logger=None)

    print(f"Audio extraction completed for the folder: {source_folder}")