import torch
import librosa
from .utils import load_datasets
from disability.utils import set_seed
from torch.utils.data import Dataset, DataLoader



class AudioMNISTDataset(Dataset):
    def __init__(self, df):
        self.audio = df['file_path']
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
        label = torch.tensor(int(self.labels[idx]))
        
        audio_seq = torch.tensor(audio_seq).to(dtype=torch.float32)
        return {'input': audio_seq, 'label': label}


def build_loader(config, file_path=None, ratio=0.1):  

    set_seed(config.seed)
    df = load_datasets(config, file_path, ratio, 'audio')

    loaders = {}
    for key, _df in df.items():
        datasets = AudioMNISTDataset(_df)
        loaders[key] = DataLoader(datasets, 
                                  batch_size=config.batch_size, 
                                  shuffle=True)
        
    return loaders



    
    