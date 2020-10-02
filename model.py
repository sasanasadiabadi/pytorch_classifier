import torch 
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=16*64*64, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(self.conv1(x))
        x = self.bn2(x)
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        x = x.view(-1, 16*64*64)
        x = self.relu(self.fc1(x))
        x = self.bn3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x 

if __name__=="__main__":
    xnet = Classifier()
    xnet.eval()
    x = torch.randn(1, 3, 256, 256)
    # run forward pass 
    out = xnet(x)
    print(out.shape)

    x = torch.rand(1, 3, 256, 256)
    torch.onnx.export(xnet, x, "classifier.onnx")