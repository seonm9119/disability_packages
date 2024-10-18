import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

class VideoLSTM(nn.Module):
    def __init__(self, 
                 num_classes=2, 
                 hidden_size=128, 
                 num_layers=2,
                 pretrain=True):
        super(VideoLSTM, self).__init__()
        
        # CNN Feature Extractor (2D CNN)
        self.cnn = models.resnet18(pretrained=True)  # ResNet18 사용
        self.cnn_fc_size = self.cnn.fc.in_features  # CNN의 출력 크기
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # 마지막 레이어 제거
        
        
        # LSTM 정의
        self.lstm = nn.LSTM(input_size=self.cnn_fc_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        
        # 최종 출력 레이어
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 입력: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, C, H, W = x.size()
        
        # CNN을 통해 각 프레임의 특징 추출
        cnn_out = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :]  # (batch_size, channels, height, width)
            frame = self.cnn(frame)  # CNN을 통한 특징 추출
            cnn_out.append(frame.view(batch_size, -1))  # (batch_size, cnn_fc_size)

        # LSTM 입력으로 변환
        lstm_input = torch.stack(cnn_out, dim=1)  # (batch_size, num_frames, cnn_fc_size)
        
        # LSTM 통과
        lstm_out, (hn, cn) = self.lstm(lstm_input)  # lstm_out: (batch_size, num_frames, hidden_size)
        
        # 마지막 시퀀스의 출력을 사용하여 분류
        output = self.fc(hn[-1])  # (batch_size, num_classes)
        
        return output