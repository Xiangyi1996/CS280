import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class AllConvNet(nn.Module):
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(AllConvNet, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 96, 3, padding=1, stride=2)

        self.conv4 = nn.Conv2d(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2d(192, 192, 3, padding=1, stride=2)

        self.conv7 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2d(192, 192, 1, padding=1)

        self.conv9 = nn.Conv2d(192, n_classes, 1, padding=1)


    def forward(self, x):
        #x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x))

        conv2_out = F.relu(self.conv2(conv1_out))

        conv3_out = F.relu(self.conv3(conv2_out))

        conv4_out = F.relu(self.conv4(conv3_out))

        conv5_out = F.relu(self.conv5(conv4_out))

        conv6_out = F.relu(self.conv6(conv5_out))

        conv7_out = F.relu(self.conv7(conv6_out))

        conv8_out = F.relu(self.conv8(conv7_out))

        conv9_out = F.relu(self.conv9(conv8_out))

        pool_out = F.adaptive_avg_pool2d(conv9_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        F.softmax(pool_out,dim=0)
        return pool_out

    def load(self, path):
        self.load_state_dict(torch.load(path))
        
    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + 'AllCNN' + '_'
            name = time.strftime(prefix + '.pth')
        torch.save(self.state_dict(), name)
        return name
    
    def load_weight(self,path):
        #model_dict = self.load_state_dict(torch.load(path))
        # print("Model's state_dict:")
        # for param_tensor in self.state_dict():
        #     print(param_tensor, "\t", self.state_dict()[param_tensor].size())
        pretrained_dict = torch.load(path)
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # 3. load the new state dict
        self.load_state_dict(model_dict)

        # 4. create final layer and initialize it
        self.linear = nn.Linear(192, 10)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0.01)
        self.cuda()  # put model on GPU once again