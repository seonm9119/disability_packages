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
