import os
import shutil


def classify_files(folder_list, save_path='D:/disorder'):
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

        user = patterns[0]  
        label = patterns[6]  


        save_path = os.path.join(save_path, user, label, 'video')
        if not os.path.exists(save_path):  
            os.makedirs(save_path)  
            
        # Copy file to the designated folder
        save_file = os.path.join(save_path, file_name)  
        if not os.path.exists(save_file): 
            shutil.copy(file_path, save_path)  
    
    print(f"File classification completed for the folder: {source_folder}") 
