import torch
import torch.nn as nn
from tqdm import tqdm
from .fcn import FullyConnectedNN
from .motion import MotionExtractor
from .audio import AudioExtractor
from disability.trainer import Trainer, to_device


class MultiClassifier(nn.Module):
    def __init__(self, 
                 num_axis=3,
                 num_landmarks=128, 
                 num_classes=2,
                 hidden_size=256,
                 num_layers=2):
        super(MultiClassifier, self).__init__()

        self.motion = MotionExtractor(num_axis=num_axis,
                                      num_landmarks=num_landmarks, 
                                      hidden_size=hidden_size,
                                      num_layers=num_layers)
        
        self.audio = AudioExtractor()

        self.fusion = nn.Sequential(nn.Linear(2*hidden_size, hidden_size),
                                    nn.ReLU(),
                                    FullyConnectedNN(input_dim=256, hidden_dim=128, num_classes=num_classes))

    def forward(self, data):

        audio_feat = self.audio(data['audio'])
        face_feat = self.motion(data['face'])

        combined_features = torch.cat((audio_feat, face_feat), dim=1)
        output = self.fusion(combined_features)

        return output
    


class MultiClassifierTrainer(Trainer):
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