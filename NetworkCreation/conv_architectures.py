import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as T
import numpy as np

class ConvBlock_v0(nn.Module):
    def __init__(self):
        super(ConvBlock_v0, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        return out6
    
    
class ConvBlock_v1(nn.Module):
    def __init__(self):
        super(ConvBlock_v1, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool2 = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out2 = self.pool1(self.conv2(self.relu2(self.bn2(out1))))
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out4 = self.pool2(self.conv4(self.relu4(self.bn4(out3))))
        return out4
    
    
class ConvBlock_v2(nn.Module):
    def __init__(self):
        super(ConvBlock_v2, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool2 = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out3 = self.pool1(self.conv3(self.relu3(self.bn3(out2))))
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out6 = self.pool2(self.conv6(self.relu6(self.bn6(out5))))
        return out6
    
class ConvBlock_v3(nn.Module):
    def __init__(self):
        super(ConvBlock_v3, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool1 = nn.MaxPool2d(2,2)
        
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.pool2 = nn.MaxPool2d(2,2)
    
    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out2 = self.pool1(self.conv2(self.relu2(self.bn2(out1))))
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out4 = self.pool2(self.conv4(self.relu4(self.bn4(out3))))
        return out4
    
