'''
file containing all models 
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F 

#---------------------------------------------------------------------
class AlexNet(nn.Module): 
    '''
    Implementation of the AlexNet model 
    found in paper: https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf 
    '''
    def __init__(self, num_classes=10): 
        super(AlexNet, self).__init__()
        
        self.l1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.LocalResponseNorm(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.l2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.LocalResponseNorm(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.l3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1), 
            nn.LocalResponseNorm(384),
            nn.ReLU())
        
        self.l4 = nn.Sequential(
            nn.Conv2D(384, 384, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(384),
            nn.ReLU())
        
        self.l5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.LocalResponseNorm(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=9216, out_features=4096), 
            nn.ReLU(), 
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=4096, out_features=num_classes))
        
    def forward(self, x): 
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out = self.l5(out)
        out = out.reshape(out.size(0), -1)
        out = self.classifier(out) 
        return out


#---------------------------------------------------------------------
# Inception model 

class ConvBlock(nn.Module): 
    '''
    All convolutional uses in network follow same method so create model 
    to reduce repitition 
    '''
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs): 
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x): 
        x = self.conv(x) 
        x = self.bn(x) 
        out = self.relu(x) 

        return out 

class Inception(nn.Module):
    '''
    Inception block for the GoogLeNet model 
    '''
    def __init__(self, in_channels, out_1, red_3, out_3, red_5, out_5, out_pool):
        super(Inception, self).__init__()
        
        self.conv1 = ConvBlock(in_channels, out_1, kernel_size=1) 
        
        self.conv3 = nn.Sequential( 
            ConvBlock(in_channels, red_3, kernel_size=1, padding=0),
            ConvBlock(red_3, out_3, kernel_size=3, padding=1))
        
        self.conv5 = nn.Sequential(
            ConvBlock(in_channels, red_5, kernel_size=1), 
            ConvBlock(red_5, out_5, kernel_size=5, padding=2)) 
        
        self.max = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1), 
            ConvBlock(in_channels, out_pool, kernel_size=1))
        
    def forward(self, x): 
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        x4 = self.max(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return out

class Auxiliary(nn.Module): 
    '''
    Auxiliary module that stems from one of the intermediate layers and takes 
    predictions into account to help network propogate gradients 
    '''
    def __init__(self, in_channels, num_classes): 
        super(Auxiliary, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1x1 = ConvBlock(in_channels, 128, kernel_size=1)
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        
        self.dropout = nn.Dropout(0.7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1x1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GoogLeNet(nn.Module): 
    '''
    GoogLeNet design based on paper: https://arxiv.org/pdf/1409.4842.pdf
    '''
    def __init__(self, out_channels=10, use_auxiliary=True):
        super(GoogLeNet, self).__init__()
        
        self.l1 = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3), 
            nn.MaxPool2d(kernel_size=3, stride=2))
             
        self.l2 = nn.Sequential(
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
        self.use_auxiliary = use_auxiliary
        if use_auxiliary: 
            self.auxiliary4a = Auxiliary(512, out_channels)
            self.auxiliary4d = Auxiliary(528, out_channels)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64) 
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64) 
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64) 
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128) 
        self.avg = nn.AvgPool2d(kernel_size=7, stride=1)
        
        self.drop = nn.Dropout(p=0.4)
            
        self.linear = nn.Linear(1024,out_channels)
    
    def forward(self, x): 
        x = self.l1(x) 
        x = self.l2(x)

        x = self.inception3a(x) 
        x = self.inception3b(x) 
        x = self.max3(x) 

        x = self.inception4a()
        if self.training and self.use_auxiliary: 
            x = self.auxiliary4a(x) 

        x = self.inception4b(x)
        x = self.inception4c(x) 
        x = self.inception4d(x) 

        if self.training and self.use_auxiliary: 
            x = self.auxiliary4d(x) 
        x = self.inception4e(x)

        x = self.max_pool(x)

        x = self.inception5a(x) 
        x = self.inception5b(x) 
        x = self.avg(x) 
        x = self.drop(x) 
        x = x.reshape(x.shape[0], -1)

        x = self.linear(x) 
        
        return x 

if __name__ == "__main__":
    pass



