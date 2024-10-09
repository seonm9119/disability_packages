import torch
import numpy as np
from .utils import load_datasets
from disability.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class MotionDataset(Dataset):
    def __init__(self, df):
        self.face = df['file_path']
        self.labels = df['label']
        assert(len(self.face) == len(self.labels))
    
    def __len__(self):
        return len(self.face)
    
    def __getitem__(self, idx):

        landmark = torch.from_numpy(np.load(self.face[idx])).to(torch.float32)
        label = int(self.labels[idx])

        return landmark, label


def collate_fn(batch):
    data, labels = zip(*batch)  # Unzip the batch into data and labels

    # Pad the data sequences
    padded_data = pad_sequence([d.clone().detach() for d in data], batch_first=True, padding_value=100)
    labels = torch.tensor(labels, dtype=torch.int64)

    return {'input' :padded_data, 'label':labels}


def build_loader(config, file_path=None, ratio=0.1, mode='face'):  
    set_seed(config.seed)
    df = load_datasets(config, file_path, ratio, mode)

    loaders = {}
    for key, _df in df.items():
        datasets = MotionDataset(_df)
        loaders[key] = DataLoader(datasets, 
                                  batch_size=config.batch_size, 
                                  shuffle=True,
                                  collate_fn=collate_fn)
        
    return loaders



    
    