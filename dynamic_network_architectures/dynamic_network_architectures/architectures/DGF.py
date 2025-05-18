import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_sigmoid(in_channels, out_channels, kernel_size=1, stride=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
        nn.Sigmoid()
    )

class DGF(nn.Module):
    def __init__(self, in_channel):
        super(DGF, self).__init__()

        self.gate1 = conv_sigmoid(in_channel, in_channel)
        self.gate2 = conv_sigmoid(in_channel, in_channel)
        self.gate3 = conv_sigmoid(in_channel, in_channel)
        self.gate4 = conv_sigmoid(in_channel, in_channel)
        self.norm = nn.InstanceNorm3d(in_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, f ,x, y, z): 
        g_x = self.gate1(x)
        g_y = self.gate2(y)
        g_z = self.gate3(z)
        g_f = self.gate4(f)
        x1 = g_x * x + (1 - g_x) * (y * g_y + z * g_z )
        y1 = g_y * y + (1 - g_y) * (x * g_x + z * g_z )
        z1 = g_z * z + (1 - g_z) * (y * g_y + x * g_x )
        f = f * g_f + (1 - g_f) * (x1 + y1 + z1)
        return f

if __name__ == '__main__':
    x = torch.randn(2, 64, 32, 64, 128)
    y = torch.randn(2, 64, 32, 64, 128)
    z = torch.randn(2, 64, 32, 64, 128)
    f = torch.randn(2, 64, 32, 64, 128)
    model = DGF(64)
    f = model(f,x,y,z)
    print(f.shape)

    # 参数量统计
    print(sum(p.numel() for p in model.parameters()))