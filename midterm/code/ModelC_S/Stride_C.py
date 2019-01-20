import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AllConvNet(nn.Module):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1,stride = 2)
        self.conv3 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv4 = nn.Conv2d(192, 192, 3, padding=1,stride = 2)

        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 1, padding=1)

        self.conv7 = nn.Conv2d(192, n_classes, 1, padding=1)


    def forward(self, x):
        #x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x))

        conv2_out = F.relu(self.conv2(conv1_out))

        conv3_out = F.relu(self.conv3(conv2_out))

        conv4_out = F.relu(self.conv4(conv3_out))

        conv5_out = F.relu(self.conv5(conv4_out))

        conv6_out = F.relu(self.conv6(conv5_out))

        conv7_out = F.relu(self.conv7(conv6_out))

        pool_out = F.adaptive_avg_pool2d(conv7_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        F.softmax(pool_out,dim=0)
        return pool_out