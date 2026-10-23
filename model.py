"""Tiny U-Net for 'where is the sheet of paper' (about 90k parameters, depthwise-separable convs).

Everything it uses is easy to run in plain JavaScript: conv (3x3 / depthwise / 1x1) + folded BatchNorm + ReLU6,
nearest x2 upsampling, concat, a global-context add, a 1x1 head and a bilinear x2 on the logits.
Input: RGB scaled to [-1, 1], 128x128. Output: logits of the sheet mask, 128x128.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(cin, cout, k, s, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))


class DS(nn.Module):  # depthwise 3x3 + pointwise 1x1
    def __init__(self, cin, cout, stride=1):
        super().__init__(); self.dw = cbr(cin, cin, 3, stride, cin); self.pw = cbr(cin, cout, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.s0 = cbr(3, 16, 3, 2)                                  # 64
        self.s1 = DS(16, 24, 1)                                     # 64
        self.s2 = nn.Sequential(DS(24, 32, 2), DS(32, 32, 1))       # 32
        self.s3 = nn.Sequential(DS(32, 64, 2), DS(64, 64, 1))       # 16
        self.s4 = nn.Sequential(DS(64, 96, 2), DS(96, 96, 1))       # 8
        self.ctx = cbr(96, 96, 1, 1)                                # whole-frame context (paper vs surface)
        self.d3 = DS(96 + 64, 64, 1); self.d2 = DS(64 + 32, 32, 1); self.d1 = DS(32 + 24, 24, 1)
        self.head = nn.Conv2d(24, 1, 1)

    def forward(self, x):
        x = (x - .5) * 2
        a = self.s0(x); b = self.s1(a); c = self.s2(b); d = self.s3(c); e = self.s4(d)
        e = e + self.ctx(e.mean((2, 3), keepdim=True))
        u = self.d3(torch.cat([F.interpolate(e, scale_factor=2, mode='nearest'), d], 1))
        u = self.d2(torch.cat([F.interpolate(u, scale_factor=2, mode='nearest'), c], 1))
        u = self.d1(torch.cat([F.interpolate(u, scale_factor=2, mode='nearest'), b], 1))
        return F.interpolate(self.head(u), scale_factor=2, mode='bilinear', align_corners=False)


if __name__ == '__main__':
    n = Net(); print(sum(p.numel() for p in n.parameters()), 'parameters'); print(n(torch.zeros(2, 3, 128, 128)).shape)
