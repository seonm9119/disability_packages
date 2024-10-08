import torch.nn as nn
import torch.nn.functional as F
from .fcn import FullyConnectedNN


class AudioExtractor(nn.Module):
    def __init__(self):
        super(AudioExtractor, self).__init__()
        self.conv1 = nn.Conv1d(40, 32, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
        return x



class AudioClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(AudioClassifier, self).__init__()
        
        self.audio = AudioExtractor()
        self.fcn = FullyConnectedNN(input_dim=256, hidden_dim=128, num_classes=num_classes)

    def forward(self, x):

        x = self.audio(x)
        x = self.fcn(x)

        return F.softmax(x, dim=1)
    
    