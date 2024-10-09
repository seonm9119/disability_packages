import torch.nn as nn
from .fcn import FullyConnectedNN


class MotionExtractor(nn.Module):
    def __init__(self, 
                 num_axis=3,
                 num_landmarks=128, 
                 hidden_size=256,
                 num_layers=2,
                 face=True):
        super(MotionExtractor, self).__init__()
        
        self.face = face
        # LSTM for processing sequences
        self.lstm = nn.LSTM(input_size=num_axis * num_landmarks, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True)

    def forward(self, x):
        # x shape: (B, frame, landmark, xyz)
        
        if self.face:
            B, F, _, _ = x.shape
            x = x.view(B, F, -1)  # Reshape to (B, frame, 3 * landmark)

        # Forward pass through LSTM
        _, (hn, _) = self.lstm(x)

        # Use the last hidden state for classification
        last_hidden = hn[-1]  # Get the last hidden state from the last layer

        return last_hidden


class MotionClassifier(nn.Module):
    def __init__(self, 
                 num_axis=3,
                 num_landmarks=128, 
                 num_classes=2,
                 hidden_size=256,
                 num_layers=2,
                 face=True):
        super(MotionClassifier, self).__init__()
        
        self.motion = MotionExtractor(num_axis=num_axis,
                                      num_landmarks=num_landmarks, 
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      face=face)

        # Fully connected layers for classification
        self.fcn = FullyConnectedNN(input_dim=hidden_size, hidden_dim=128, num_classes=num_classes)

    def forward(self, x):
        
        x = self.motion(x)
        x = self.fcn(x)

        return x