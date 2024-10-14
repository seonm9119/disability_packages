import torch
import numpy as np
from .utils import load_datasets
from disability.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from pytorchvideo.transforms import UniformTemporalSubsample

class MotionDataset(Dataset):
    def __init__(self, df, num_samples):
        self.face = df['file_path']
        self.labels = df['label']
        self.transform = UniformTemporalSubsample(num_samples=num_samples)


        assert(len(self.face) == len(self.labels))
    
    def __len__(self):
        return len(self.face)
    
    def __getitem__(self, idx):

        landmark = torch.from_numpy(np.load(self.face[idx])).to(torch.float32)
        landmark = landmark.unsqueeze(0)
        
        landmark = self.transform(landmark).squeeze(0)
        label = int(self.labels[idx])

        return {'input' :landmark, 'label':label}

def build_loader(config, file_path=None, ratio=0.1, num_samples=50, mode='face'):  
    set_seed(config.seed)
    df = load_datasets(config, file_path, ratio, mode)

    loaders = {}
    for key, _df in df.items():
        datasets = MotionDataset(_df, num_samples)
        loaders[key] = DataLoader(datasets, 
                                  batch_size=config.batch_size, 
                                  shuffle=True)
        
    return loaders



    
    