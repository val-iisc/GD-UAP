import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.legacy.nn import SpatialCrossMapLRN as SpatialCrossMapLRNOld
from torch.autograd import Function, Variable
from torch.nn import Module

# function interface, internal, do not use this one!!!
class SpatialCrossMapLRNFunc(Function):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        self.save_for_backward(input)
        self.lrn = SpatialCrossMapLRNOld(self.size, self.alpha, self.beta, self.k)
        self.lrn.type(input.type())
        return self.lrn.forward(input)

    def backward(self, grad_output):
        input, = self.saved_tensors
        return self.lrn.backward(input, grad_output)

# use this one instead
class SpatialCrossMapLRNa(Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(SpatialCrossMapLRNa, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input):
        return SpatialCrossMapLRNFunc(self.size, self.alpha, self.beta, self.k)(input)

__all__ = ['AlexNet', 'alexnet']

class LRN(nn.Module):
    def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
        super(LRN, self).__init__()
        self.lrn = SpatialCrossMapLRNa(size, alpha, beta, k)

    def forward(self, x):
        #self.lrn.clearState()
        return self.lrn.forward(x)
   
    
    
class fcn_alexnet(nn.Module):

    def __init__(self):
        super(fcn_alexnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=100)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride = 2)
        self.norm1 = LRN(size=5, alpha=0.0001, beta=0.75)
        #nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride = 2)
        self.norm2 = LRN(size=5, alpha=0.0001, beta=0.75)
        #nn.MaxPool2d(kernel_size=3, stride=2),
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc6 = nn.Conv2d(256, 4096, kernel_size=6, padding=0, groups=1)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout()
        self.fc7= nn.Conv2d(4096, 4096, kernel_size=1, padding=0, groups=1)
        self.reelu7 = nn.ReLU()
        self.drop7 = nn.Dropout()
        self.score_fr = nn.Conv2d(4096, 21, kernel_size=1, padding=0, groups=1)
        self.upscore_final = nn.ConvTranspose2d(21, 21, 63, stride=32, groups=1, bias=False)
        
        ## Handle crop
        
        
    def forward(self, y):
        x = y
        #self.interp = nn.UpsamplingBilinear2d(size = (  y[2],y[3]   ))\
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        #print(x.size())
        x = self.norm1(x)
        #print(x.size())
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        #print(x.size())
        x = self.norm2(x)
        #print(x.size())
        x = self.relu3(self.conv3(x))
        #print(x.size())
        x = self.relu4(self.conv4(x))
        #print(x.size())
        x = self.relu5(self.conv5(x))
        x = self.pool5(x)
        #print(x.size())
        x = self.relu6(self.fc6(x))
        #print(x.size())
        x = self.drop6(x)
        x = self.reelu7(self.fc7(x))
        #print(x.size())
        x = self.drop7(x)
        x = self.score_fr(x)
        #print(x.size())
        x = self.upscore_final(x)
        #print('upscore',x.size())

        # The output sizes of Caffe and Pytorch did not match here, hence the crop is different.
        # The performance we get is 46.21 mIU, compared to 48.0 mIU from:
        # https://github.com/shelhamer/fcn.berkeleyvision.org

        x = x[:, :, 11:11 + y.size()[2], 11:11 + y.size()[3]]
        #x = self.interp(x)
        return x
