import os

def iterate_label_folder(func, user_path='D:/disorder/004m'):
    """
    Iterates through label folders in the specified user path and applies the given function.

    Parameters:
        func (function): A function to apply to each label folder. 
                         It should accept a folder path as its argument.
        user_path (str, optional): The path to the user directory containing label folders. 
                                   Defaults to 'D:/disorder/004m'.
    """
    
    folder_list = os.listdir(user_path)
    for label in folder_list:
        # Check if the item is a folder (ignores items with a '.' in the name)
        if '.' not in label:
            folder = os.path.join(user_path, label)
            func(folder)



def filter_unique_audio(mp4_files):
    """
    Filters and returns unique audio files from the given list based on camera number.
    
    Parameters:
        mp4_files (list): List of .mp4 file paths.
    """
    patterns = {}
    
    for file_path in mp4_files:
        segments = file_path.split('_') 
        cam = int(segments[4]) 

        segments[4] = '*'
        pattern = '_'.join(segments).split('*')[0]  

        # Only cameras with numbers greater than 6 have usable audio
        # Store the file path for the unique pattern
        if pattern not in patterns and cam > 6:  
            patterns[pattern] = file_path  

    # Return a list of unique file paths
    return list(patterns.values())  
