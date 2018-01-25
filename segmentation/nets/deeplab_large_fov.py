import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

class deeplab_vgg_lfov(nn.Module):

    def __init__(self):
        super(deeplab_vgg_lfov, self).__init__()
        self.conv1_1 = nn.Conv2d(3,64,3,padding = 1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64,64,3,padding = 1)
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv2_1 = nn.Conv2d(64,128,3,padding = 1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128,128,3,padding = 1)
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv3_1 = nn.Conv2d(128,256,3,padding = 1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256,256,3,padding = 1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256,256,3,padding = 1)
        self.relu3_3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = 3, stride = 2,padding=1)
        self.conv4_1 = nn.Conv2d(256,512,3,padding = 1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512,512,3,padding = 1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512,512,3,padding = 1)
        self.relu4_3 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size = 3, stride = 1,padding=1)
        self.conv5_1 = nn.Conv2d(512,512,3,padding = 2,dilation = 2)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512,512,3,padding = 2,dilation = 2)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512,512,3,padding = 2,dilation = 2)
        self.relu5_3 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size = 3, stride = 1,padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size = 3, stride = 1,padding=1)
        self.fc6 = nn.Conv2d(512,1024,3,padding = 12,dilation = 12)
        self.relu6 = nn.ReLU()
        #self.drop6 = nn.Dropout2d(p=0.5)
        self.fc7 = nn.Conv2d(1024,1024,1)
        # As this is an equivalent of FC, we do not add this term to the optimization.
        self.reelu7 = nn.ReLU()
        #self.drop7 = nn.Dropout2d(p=0.5)
        self.fc8_voc12 = nn.Conv2d(1024,21,1)
        self.log_softmax = nn.LogSoftmax()
        self.fc8_interp_test = nn.UpsamplingBilinear2d(size=(513,513))
        
    
    def forward(self, x):
        y = x.size()
        x = self.relu1_1(self.conv1_1(x))
        x = self.pool1(self.relu1_2(self.conv1_2(x)))
        x = self.relu2_1(self.conv2_1(x))
        x = self.pool2(self.relu2_2(self.conv2_2(x)))
        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.pool3(self.relu3_3(self.conv3_3(x)))
        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.pool4(self.relu4_3(self.conv4_3(x)))
        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.pool5a(self.pool5(self.relu5_3(self.conv5_3(x))))
        x = self.relu6(self.fc6(x))
        x = self.reelu7(self.fc7(x))
        x = self.fc8_voc12(x)
        x = self.log_softmax(x)
        x = self.fc8_interp_test(x)
        
        return x
