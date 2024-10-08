import os
import random
import numpy as np
import torch
import logging

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(file_path):
    """Set up logging configuration."""
    logger = logging.getLogger('PyTorchLogger')
    logger.setLevel(logging.INFO)  # Set the logging level

    # Create a file handler for writing log messages to a file
    file_handler = logging.FileHandler(f'{file_path}/training.log')
    file_handler.setLevel(logging.INFO)

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define the format of log messages
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def remove_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)