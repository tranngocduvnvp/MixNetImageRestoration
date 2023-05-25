import torch
import torch.nn as nn
import torch.nn.functional as F

class EFFat(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) + self.max_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + x
    

class convMixerLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channel, 
            out_channels=in_channel, 
            groups=in_channel, 
            kernel_size=kernel_size,
            stride=1,
            padding="same"
        )
        self.eff = EFFat(out_channel)
        self.pointwise = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.activation = nn.SiLU()
        self.batchnorm1 = nn.BatchNorm2d(in_channel)
        self.batchnorm2 = nn.BatchNorm2d(out_channel)
    
    def forward(self, x):
        ori = x
        x = self.depthwise(x)
        x = self.activation(x)
        x = self.eff(self.batchnorm1(x)) + ori
        x = self.pointwise(x)
        x = self.activation(x)
        x = self.batchnorm2(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0 ,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0 ,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class Channel_attention(nn.Module):
    def __init__(self, c, reduction=16) -> None:
        super().__init__()
        self.fcap1 = nn.Conv2d(c, c//reduction,1)
        self.fcmp1 = nn.Conv2d(c, c//reduction, 1)
        self.fcap2 = nn.Conv2d(c//reduction, c, 1)
        self.fcmp2 = nn.Conv2d(c//reduction, c, 1)
        self.acg = nn.Conv2d(c//reduction, c, 1)
    def forward(self, x):
        f1 = F.relu(self.fcap1(x))
        f2 = F.relu(self.fcmp1(x))
        f = self.acg(f1 + f2)
        out = F.sigmoid(f + self.fcap2(f1) + self.fcmp2(f2))*x
        return out

class CTblock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3) -> None:
        super().__init__()
        self.mixspatial = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, padding="same", groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            Channel_attention(in_channel),
            nn.Conv2d(in_channel, out_channel, 3, padding="same"),
            nn.BatchNorm2d(out_channel),
        )
        self.mixchannel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
        )
    
    def forward(self, x):
        y = F.relu(self.mixspatial(x) + self.mixchannel(x))
        return y
class UpDownstream(nn.Module):
    def __init__(self, scale, in_channel, out_channel) -> None:
        super().__init__()
        self.scale = scale
        self.conv1x1 = nn.Conv2d(in_channel, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.ac = nn.GELU()
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.ac(x)
        bn, c, h, w = x.shape
        x = F.interpolate(x, size=(int(h*self.scale), int(w*self.scale)), mode="bilinear")
        return x

class NormMode(nn.Module):
    def __init__(self, scale, in_channel, out_channel) -> None:
        super().__init__()
        """nhận đầu vào là một tensor cxhxw

        Returns:
            - vector key: d
            - Tensor value: c1xh1xw1
        """
        self.norm = UpDownstream(scale, in_channel, out_channel)
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.mg = nn.AdaptiveMaxPool2d((1,1))
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.GELU(),
            nn.Conv2d(in_channel, out_channel, 1)
        )

    def forward(self, x):
        v = self.norm(x).unsqueeze(1) # (bs, 1, c, h, w)
        k = self.mlp(self.avg(x)+self.mg(x)).view(x.shape[0], 1, -1) #(bs,1, c)
        return v, k
        

class AttentionDC(nn.Module):
    def __init__(self, scale, in_channel, out_channel) -> None:
        super().__init__()
        self.normfm1 = NormMode(scale[0], in_channel[0], out_channel)
        self.normfm2 = NormMode(scale[1], in_channel[1], out_channel)
        self.normfm3 = NormMode(scale[2], in_channel[2], out_channel)
        self.normfm4 = NormMode(scale[3], in_channel[3], out_channel)
        self.normfmDecode = NormMode(1, out_channel, out_channel)
        self.mlp = nn.Linear(out_channel*2, out_channel)

    
    def forward(self, feature_maps):
        fm1, fm2, fm3, fm4, fmdecode = feature_maps
        v1, k1 = self.normfm1(fm1)
        v2, k2 = self.normfm2(fm2)
        v3, k3 = self.normfm3(fm3)
        v4, k4 = self.normfm4(fm4)
        vd, qd = self.normfmDecode(fmdecode) #(bs, 1, c)
        K = torch.cat([k1, k2, k3, k4], dim=1) #(bs, 4, c)
        K = torch.cat([K, qd.expand_as(K)], dim=2) #(bs, 4, 2c)
        atten = F.softmax(self.mlp(K), dim=1).unsqueeze(-1).unsqueeze(-1)   #(bs, 4, c, 1, 1)
        V = torch.cat([v1,v2,v3,v4], dim=1) #(bs, 4, c, h, w)
        V = V*atten #(bs, 4, c, h, w)
        V = torch.sum(V, dim=1) #(bs, c, h, w)
        V = torch.cat([V, vd.squeeze(1)], dim=1)
        return V

class RB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=2, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.out_layers = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=2, padding="same"),
            nn.BatchNorm2d(out_channels, out_channels),
            nn.GELU(),
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)