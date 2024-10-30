import torch
import torch.nn as nn
from tqdm import tqdm
from .fcn import FullyConnectedNN
from .videoLSTM import VideoLSTM
from .audio import AudioExtractor
from disability.trainer import Trainer, to_device


class AudioFlowClassifier(nn.Module):
    def __init__(self, 
                 num_classes=2,
                 hidden_size=256,
                 num_layers=2):
        super().__init__()

        self.flow = VideoLSTM(num_classes=num_classes, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers)
        
        self.audio = AudioExtractor()

        self.fusion = nn.Sequential(nn.Linear(2*hidden_size, hidden_size),
                                    nn.ReLU(),
                                    FullyConnectedNN(input_dim=256, hidden_dim=128, num_classes=num_classes))

    def forward(self, data):

        audio_feat = self.audio(data['audio'])
        face_feat = self.flow(data['flow'])

        combined_features = torch.cat((audio_feat, face_feat), dim=1)
        output = self.fusion(combined_features)

        return output
    


class AudioFlowClassifierTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def train(self, model, loader, optimizer, criterion, device):
        losses = []
        model.train()
        
        for data in tqdm(loader):
            data = to_device(data, device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data['label'])
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        
        return losses
    

    def eval(self, model, loader, criterion, device):
        
        losses = []
        correct = 0
        total = 0
        model.eval()
    
        with torch.no_grad():
            for data in tqdm(loader):
                data = to_device(data, device)

                output = model(data)
                loss = criterion(output, data['label'])
                losses.append(loss.item())

                predicted = torch.argmax(output.data, dim=1)
                total += data['label'].size(0)
                correct += (predicted == data['label']).sum()
        
        return losses, total, correct