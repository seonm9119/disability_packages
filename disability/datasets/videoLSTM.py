import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import load_datasets
from disability.utils import set_seed

from pytorchvideo.transforms import (
    Normalize,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    Resize,
)


class VideoDataset(Dataset):
    def __init__(self, df, num_samples):
        self.video = df['file_path']
        self.labels = df['label']

        self.transform=Compose(
                [
                    UniformTemporalSubsample(num_samples=num_samples),
                    Lambda(lambda x: x / 255.0),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    Resize(224),
                ]
            )


        assert(len(self.video) == len(self.labels))
    
    def __len__(self):
        return len(self.video)
    
    def __getitem__(self, idx):
        video = self.load_video(self.video[idx])  # 비디오 로드
        video = self.transform(video).permute(1, 0, 2, 3)
        

        label = self.labels[idx]
        return {'input' :video, 'label':label}

    def load_video(self, path, img_size=224):
        # OpenCV를 사용하여 비디오 파일 열기
        cap = cv2.VideoCapture(path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # 비디오의 끝에 도달하면 종료
            
            # OpenCV는 기본적으로 BGR 형식으로 프레임을 반환하므로 RGB로 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 프레임을 텐서로 변환하고 크기 조정 (예: 224x224)
            frame = cv2.resize(frame, (img_size, img_size))  # 원하는 크기로 조정
            frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  # (channels, height, width)
            
            frames.append(frame)

        cap.release()  # 비디오 캡처 객체 해제
        
        # 모든 프레임을 텐서로 변환
        video_tensor = torch.stack(frames).permute(1, 0, 2, 3)  # (num_frames, channels, height, width)
        return video_tensor


def build_loader(config, file_path=None, ratio=0.1, num_samples=50, mode='video'):  
    set_seed(config.seed)
    df = load_datasets(config, file_path, ratio, mode)

    loaders = {}
    for key, _df in df.items():
        datasets = VideoDataset(_df, num_samples)
        loaders[key] = DataLoader(datasets, 
                                  batch_size=config.batch_size, 
                                  shuffle=True)
        
    return loaders



    
    