import torch
import torch.nn as nn
# x1 = torch.randn(128, 512, 28, 28)
# x2 = nn.Conv2d(512, 2048, kernel_size=4, padding=1, stride=4)
# x = x2(x1)
# print(x.shape)
lay1 = torch.randn(128, 256, 56, 56)
c1 = nn.Conv2d(256, 2048, kernel_size=8, padding=1, stride=8)
l1 = c1(lay1)
print(l1.shape)
lay3 = torch.randn(128, 1024, 14, 14)
c3 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1, stride=2)
l3 = c3(lay3)
print(l3.shape)









































