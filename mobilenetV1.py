import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class DSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, depth_multiplier=1.0, kernel_size=3, stride=1, padding=1, groups=1):
        super(DSConv2d, self).__init__()
        self.dsconv = nn.Sequential(
                nn.Conv2d(in_channels=int(in_channels*depth_multiplier), out_channels=int(in_channels*depth_multiplier), 
                kernel_size=kernel_size, stride=stride, padding=padding, groups=int(in_channels*depth_multiplier)), 
                nn.BatchNorm2d(num_features=int(in_channels*depth_multiplier)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(in_channels*depth_multiplier), out_channels=int(out_channels*depth_multiplier), 
                kernel_size=1, groups=groups),
                nn.BatchNorm2d(num_features=int(out_channels*depth_multiplier)),
                nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.dsconv(x)

        return x 



class MobilenetV1(nn.Module):
    def __init__(self, depth_multiplier=1.0):
        super(MobilenetV1, self).__init__()
        self.conv1   = nn.Conv2d(in_channels=3, out_channels=int(32*depth_multiplier), kernel_size=3, stride=2, padding=1)
        self.dsconv1 = DSConv2d(in_channels=32, out_channels=64, depth_multiplier=depth_multiplier)
        self.dsconv2 = DSConv2d(in_channels=64, out_channels=128, depth_multiplier=depth_multiplier, stride=2)
        self.dsconv3 = DSConv2d(in_channels=128, out_channels=128, depth_multiplier=depth_multiplier)
        self.dsconv4 = DSConv2d(in_channels=128, out_channels=256, depth_multiplier=depth_multiplier, stride=2)
        self.dsconv5 = DSConv2d(in_channels=256, out_channels=256, depth_multiplier=depth_multiplier)
        self.dsconv6 = DSConv2d(in_channels=256, out_channels=512, depth_multiplier=depth_multiplier, stride=2)
        self.dsconv7 = nn.Sequential(
            DSConv2d(in_channels=512, out_channels=512, depth_multiplier=depth_multiplier)
        )
        self.dsconv8 = DSConv2d(in_channels=512, out_channels=1024, depth_multiplier=depth_multiplier, stride=2)
        self.dsconv9 = DSConv2d(in_channels=1024, out_channels=1024, depth_multiplier=depth_multiplier)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc1 = nn.Linear(in_features=1024, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        x = self.dsconv4(x)
        x = self.dsconv5(x)
        x = self.dsconv6(x)
        x = self.dsconv7(x)
        x = self.dsconv8(x)
        x = self.dsconv9(x)
        x = self.avg_pool(x)
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.sigmoid(x)

        return x


if __name__=="__main__":
    mnv1 = MobilenetV1(depth_multiplier=0.5)
    mnv1.cuda()
    summary(mnv1, (3, 224, 224))
