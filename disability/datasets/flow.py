import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .utils import load_datasets
from disability.utils import set_seed

from pytorchvideo.transforms import (
    Normalize,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda
)


class OpticalFlowDataset(Dataset):
    def __init__(self, df, num_samples):
        self.flow = df['file_path']
        self.labels = df['label']

        self.transform=Compose(
                [
                    UniformTemporalSubsample(num_samples=num_samples),
                    Lambda(lambda x: x / 255.0),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )


        assert(len(self.flow) == len(self.labels))
    
    def __len__(self):
        return len(self.flow)
    
    def __getitem__(self, idx):

        flow = torch.from_numpy(np.load(self.flow[idx])).to(torch.float32)
        flow = flow.permute(3,0,1,2)
        flow = self.transform(flow)
        label = int(self.labels[idx])
        
        flow = flow.permute(1, 0, 2, 3)
        return {'input' :flow, 'label':label}


def build_loader(config, file_path=None, ratio=0.1, num_samples=50, mode='flow'):  
    set_seed(config.seed)
    df = load_datasets(config, file_path, ratio, mode)

    loaders = {}
    for key, _df in df.items():
        datasets = OpticalFlowDataset(_df, num_samples)
        loaders[key] = DataLoader(datasets, 
                                  batch_size=config.batch_size, 
                                  shuffle=True)
        
    return loaders



    
    