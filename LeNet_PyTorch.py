import torch
import torch.nn as nn

# LeNet 구현

class LeNet(nn.Module):
  def __init__(self, n_output, n_hidden1, n_hidden2):
    super(LeNet, self).__init__()
    # 3개의 입력 이미지 채널, 6개의 출력 특징 맵, 5x5 합성곱 커널 [합성곱 처리 행렬의 사이즈-1 만큼 출력 데이터의 화소 수가 줄어듦.]
    self.conv1 = nn.Conv2d(3, 6, 5)
    # 6개의 입력 이미지 채널, 16개의 출력 특징 맵, 5x5 합성곱 커널
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d((2,2))
    self.flatten = nn.Flatten()
    # 크기 120, 84, 10의 완전 연결 계층
    self.l1 = nn.Linear(2304, n_hidden1) # 5*5는 이 계층의 공간 차원임
    self.l2 = nn.Linear(n_hidden1, n_hidden2)
    self.l3 = nn.Linear(n_hidden2, n_output)

    self.features = nn.Sequential(
        self.conv1,
        self.relu,
        self.conv2,
        self.relu,
        self.maxpool)
    
    self.classifier = nn.Sequential(
        self.l1,
        self.relu,
        self.l2,
        self.relu,
        self.l3)


  def forward(self, x):
    x1 = self.features(x)
    x2 = self.flatten(x1)
    x3 = self.classifier(x2)
    return x3
