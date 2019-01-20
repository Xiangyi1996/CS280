from allconv import *
import torch
model = AllConvNet(3)
best = torch.load('best.pt')
print(best)