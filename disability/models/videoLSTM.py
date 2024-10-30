import torch
import torch.nn as nn
import torchvision.models as models

class VideoLSTM(nn.Module):
    def __init__(self, 
                 num_classes=2, 
                 hidden_size=128, 
                 num_layers=2):
        super(VideoLSTM, self).__init__()
        
        self.cnn = models.resnet18(pretrained=True) 
        self.cnn_fc_size = self.cnn.fc.in_features
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        
        
        # LSTM 정의
        self.lstm = nn.LSTM(input_size=self.cnn_fc_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        
        cnn_out = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :]
            frame = self.cnn(frame)
            cnn_out.append(frame.view(batch_size, -1))

        lstm_input = torch.stack(cnn_out, dim=1)
        _, (hn, _) = self.lstm(lstm_input)
        
        return hn[-1]
    

class VideoClassifier(nn.Module):
    def __init__(self, 
                 num_classes=2, 
                 hidden_size=128, 
                 num_layers=2):
        super(VideoClassifier, self).__init__()
        
        self.extractor = VideoLSTM(num_classes=num_classes, 
                                   hidden_size=hidden_size, 
                                   num_layers=num_layers)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.extractor(x)
        output = self.fc(x)
        return output