import os
import random
import fnmatch
import numpy as np
import pandas as pd
from disability.utils import set_seed

def make_multidatasets(config, mode='face'):

    classes_to_idx = {}
    classes = []

    csv_filename = os.path.join(config.data_dir, config.annotation)
    file = pd.read_csv(csv_filename, header=None, encoding='utf-8')

    # 딕셔너리 및 리스트 초기화
    classes_to_idx = pd.Series(file[0].values, index=file[1]).to_dict()
    classes = file[1].tolist()

    dataframes = []
    for class_folder in classes:
        audio_path = os.path.join(config.data_dir, class_folder, 'audio')
        motion_path = os.path.join(config.data_dir, class_folder, mode)
        motion_list = os.listdir(motion_path)

        
        df = pd.DataFrame(columns=[f'{mode}_path', 'audio_path'])
        for file in os.listdir(audio_path):
            file_name = file.split('.')[0]

            segments = file_name.split('_')
            
            if segments[4].isdigit():
                segments[4] = '*'

            pattern = '_'.join(segments)


            matched_face = [face_file for face_file in motion_list if fnmatch.fnmatch(face_file, f"{pattern}*")]

            if matched_face:
                random_face = random.choice(matched_face)
                df.loc[len(df)] = {f'{mode}_path': random_face, 'audio_path': file}  
        
        df[f'{mode}_path'] = df[f'{mode}_path'].apply(lambda x: os.path.join(motion_path, x))
        df['audio_path'] = df['audio_path'].apply(lambda x: os.path.join(audio_path, x))
        df['label'] = classes_to_idx[class_folder]

        dataframes.append(df)


    df = pd.concat(dataframes, ignore_index=True)
    print(f"Saving merged training data to CSV file at: {config.csv_file}")
    df.to_csv(f'{config.data_dir}/{config.csv_file}', index=False)

    idx_to_classes = {value: key for key, value in classes_to_idx.items()}
    class_counts = df.groupby('label').size().reset_index(name='count')
    class_counts = class_counts[['count', 'label']]
    class_counts['label'] = class_counts['label'].apply(lambda x: str(idx_to_classes[x]).ljust(10))
    print(f"{class_counts}\n")


def load_datasets(config, file_path=None, ratio=0.1, mode='face'):

    if file_path is None:
        make_multidatasets(config, mode)
        file_path = os.path.join(config.data_dir, config.csv_file)
        return load_datasets(config, file_path=file_path, ratio=ratio, mode=mode)
    


    df = pd.read_csv(file_path)
    shuffled_df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    val_size = int(len(shuffled_df) * ratio)
    val_df = shuffled_df[:val_size].reset_index(drop=True)
    train_df = shuffled_df[val_size:].reset_index(drop=True)

    print(f"Total number of samples: {len(shuffled_df)}")
    print(f"Number of training samples: {len(train_df)}")
    print(f"Number of validation samples: {len(val_df)}")
    
    return {'train': train_df, 'val': val_df}


import torch
from torch.utils.data import Dataset, DataLoader
import librosa

class MultiDataset(Dataset):
    def __init__(self, df, mode='face'):
        self.audio = df['audio_path']
        self.face = df[f'{mode}_path']
        self.labels = df['label']
        assert(len(self.audio) == len(self.labels))
    
    def __len__(self):
        return len(self.audio)
    
    def get_data(self, file, target_sr=16000):
        data, sr = librosa.load(file)
        down_d = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        fix_len_d = librosa.util.fix_length(down_d, size=12000)
        return fix_len_d, target_sr
    
    def mfcc_data(self, file):
        data,sr = self.get_data(file)
        data = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
        return data
    
    def __getitem__(self, idx):
        audio_seq = self.mfcc_data(self.audio[idx])
        landmark = torch.from_numpy(np.load(self.face[idx])).to(torch.float32)
        label = int(self.labels[idx])

        return audio_seq, landmark, label

import torch
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    audio, face, labels = zip(*batch)  # Unzip the batch into data and labels

    # Pad the data sequences
    padded_data = pad_sequence([d.clone().detach() for d in face], batch_first=True, padding_value=100)
    labels = torch.tensor(labels, dtype=torch.int64)
    audio = torch.tensor(audio).to(dtype=torch.float32)

    return {'input' :{'audio' : audio, 'face': padded_data}, 'label':labels}

def build_loader(config, file_path=None, ratio=0.1, mode='face'):  
    set_seed(config.seed)
    df = load_datasets(config, file_path, ratio, mode)

    loaders = {}
    for key, _df in df.items():
        datasets = MultiDataset(_df, mode)
        loaders[key] = DataLoader(datasets, 
                                  batch_size=config.batch_size, 
                                  shuffle=True,
                                  collate_fn=collate_fn)
        
    return loaders