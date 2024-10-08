import os
import glob
from moviepy.editor import VideoFileClip

def audio_extractor(source_folder, **kwargs):

    mp4_files = glob.glob(os.path.join(source_folder, '*.mp4'))
    for file_path in mp4_files:
        audio_path = f"{file_path.split('.')[0]}.wav"
        
        if not os.path.exists(audio_path):
            video_clip = VideoFileClip(file_path)
            audio_clip = video_clip.audio

            if audio_clip is not None:
                audio_clip.write_audiofile(audio_path, verbose=False, logger=None)

    print(f"Audio extraction completed for the folder: {source_folder}")