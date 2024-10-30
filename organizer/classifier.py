import os
import shutil


def classify_files(folder_list, save_path='D:/disorder'):
    """
    Classifies files from multiple source folders provided in folder_list, 
    saving them into a structured directory under the specified save_path.
    
    Parameters:
        folder_list (list): List of folders containing files to classify.
        save_path (str): Base path where classified files will be saved. 
                         Default is 'D:/disorder'.
    """
    
    # Classify files in each source folder
    for folder in folder_list:
        for source_folder in folder:
            _classify_files(source_folder, save_path)
 



def _classify_files(source_folder, save_path='D:/disorder'):
    """
    Classifies files in the source_folder based on user and label extracted 
    from the file names, then saves them into a structured directory 
    under the specified save_path.

    Parameters:
        source_folder (str): Path to the folder containing files to classify.
        save_path (str): Base path where classified files will be saved. 
                         Default is 'D:/disorder'.
    """
    
    for file_name in os.listdir(source_folder):
        file_path = os.path.join(source_folder, file_name)  
        patterns = file_name.split("_")  
        file_extension = file_name.split(".")[1]  
        
        # Skip files that are not .mp4
        if file_extension != 'mp4':
            continue

        user = patterns[0]  
        label = patterns[6]  

        # Construct the destination path for saving the classified file
        destination_path = os.path.join(save_path, user, label, 'video')
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)  
            
        # Copy file to the designated folder
        save_file = os.path.join(destination_path, file_name)
        if not os.path.exists(save_file): 
            shutil.copy(file_path, destination_path)  
    
    print(f"File classification completed for the folder: {source_folder}") 
