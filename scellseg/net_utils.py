import torch
import torch.nn as nn
import torch.nn.functional as F

def cbrBlock(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
        nn.BatchNorm2d(out_channels, eps=1e-5),
        nn.ReLU(inplace=True),
    )

def brcBlock(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )

def bcBlock(in_channels, out_channels, sz):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels, eps=1e-5),
        nn.Conv2d(in_channels, out_channels, sz, padding=sz//2),
    )


def cbBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
        nn.BatchNorm2d(out_channels, eps=1e-5)
    )


class makeStyle(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pool_all = nn.AvgPool2d(28)
        self.flatten = nn.Flatten()

    def forward(self, x0):
        # style = self.pool_all(x0)
        style = F.avg_pool2d(x0, kernel_size=(x0.shape[-2], x0.shape[-1]))
        style = self.flatten(style)
        style = style / torch.sum(style ** 2, axis=1, keepdim=True) ** .5
        # style = style*0
        return style


class brcStyle(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False):
        super().__init__()
        self.conv = brcBlock(in_channels, out_channels, sz)

        self.concatenation = concatenation
        if concatenation:
            self.full = nn.Linear(style_channels, out_channels * 2)
        else:
            self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False):
        feat = self.full(style)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat.unsqueeze(-1).unsqueeze(-1)).to_mkldnn()
        else:
            y = x + feat.unsqueeze(-1).unsqueeze(-1)

        y = self.conv(y)
        return y

class attnBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int,):
        super(attnBlock, self).__init__()
        self.W_g = cbBlock(F_g, F_int)
        self.W_x = cbBlock(F_l, F_int)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1, eps=1e-5),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x, mkldnn=False):
        g1 = self.W_g(g)  # gating signal conv for feature of downsampling pass
        x1 = self.W_x(x)  # gating signal conv for feature of upsampling pass
        psi = self.relu(g1 + x1)  # concat + relu
        psi = self.psi(psi)
        if mkldnn:
            out = (x.to_dense() * psi.to_dense()).to_mkldnn()
        else:
            out = x * psi
        return out


class convdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            if t == 0:
                self.conv.add_module('conv_%d' % t, brcBlock(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, brcBlock(out_channels, out_channels, sz))

    def forward(self, x):
        x = self.conv[0](x)
        x = self.conv[1](x)
        return x


class resdown(nn.Module):
    def __init__(self, in_channels, out_channels, sz):
        super().__init__()
        self.conv = nn.Sequential()
        self.proj = bcBlock(in_channels, out_channels, 1)
        for t in range(4):
            if t == 0:
                self.conv.add_module('conv_%d' % t, brcBlock(in_channels, out_channels, sz))
            else:
                self.conv.add_module('conv_%d' % t, brcBlock(out_channels, out_channels, sz))

    def forward(self, x):
        x = self.proj(x) + self.conv[1](self.conv[0](x))
        x = x + self.conv[3](self.conv[2](x))
        return x


class downsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase) - 1):
            if residual_on:
                self.down.add_module('res_down_%d' % n, resdown(nbase[n], nbase[n + 1], sz))
            else:
                self.down.add_module('conv_down_%d' % n, convdown(nbase[n], nbase[n + 1], sz))

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            if n > 0:
                y = self.maxpool(xd[n - 1])
            else:
                y = x
            xd.append(self.down[n](y))
        return xd


class resup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False, dense_on=False):
        super().__init__()
        self.conv = nn.Sequential()
        self.dense_on = dense_on
        self.conv.add_module('conv_0', brcBlock(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', brcStyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation))
        self.conv.add_module('conv_2', brcStyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module('conv_3', brcStyle(out_channels, out_channels, style_channels, sz))
        self.proj = bcBlock(in_channels, out_channels, 1)
        if dense_on:
            self.up_conv = cbrBlock(in_channels+out_channels, out_channels, 1)

    def forward(self, x, y, style, mkldnn=False):
        out = self.proj(x) + self.conv[1](style, self.conv[0](x) + y, mkldnn=mkldnn)
        out = out + self.conv[3](style, self.conv[2](style, out, mkldnn=mkldnn), mkldnn=mkldnn)
        if self.dense_on:
            if mkldnn:
                x = x.to_dense()
                out = out.to_dense()
                out = self.up_conv(torch.cat([x, out], dim=1).to_mkldnn())
            else:
                out = self.up_conv(torch.cat([x, out], dim=1))
        return out


class convup(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, sz, concatenation=False, dense_on=False):
        super().__init__()
        self.dense_on = dense_on
        self.conv = nn.Sequential()
        if dense_on:
            self.up_conv = cbrBlock(in_channels+out_channels, out_channels, 1)
        self.conv.add_module('conv_0', brcBlock(in_channels, out_channels, sz))
        self.conv.add_module('conv_1', brcStyle(out_channels, out_channels, style_channels, sz, concatenation=concatenation))

    def forward(self, x, y, style, mkldnn=False):
        out = self.conv[1](style, self.conv[0](x) + y)
        if self.dense_on:
            out = self.up_conv(torch.cat([x, out], dim=1))
        return out


class upsample(nn.Module):
    def __init__(self, nbase, sz, residual_on=True, concatenation=False, attn_on=False, dense_on=False, style_channels=[256, 256, 256, 256]):
        super().__init__()
        self.attn_on = attn_on
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')
        self.up = nn.Sequential()
        if attn_on:
            self.up_conv = nn.Sequential()
            self.attn = nn.Sequential()
        nblock = 1
        for n in range(1, len(nbase)-1):
            if residual_on:
                self.up.add_module('res_up_%d' % (n - 1), resup(nbase[n], nbase[n-1], style_channels[n-1], sz, concatenation, dense_on=dense_on))
            else:
                self.up.add_module('conv_up_%d' % (n - 1), convup(nbase[n], nbase[n-1], style_channels[n-1], sz, concatenation, dense_on=dense_on))

            if attn_on:
                self.up_conv.add_module('up_conv_%d' % (n - 1), cbrBlock(nbase[n], nbase[n-1], 1))
                self.attn.add_module('attn_%d' % (n - 1), attnBlock(nbase[n-1], nbase[n-1], nbase[n-1] // 2))
            nblock+=1

        if residual_on:
            self.up.add_module('res_up_%d' % (nblock-1), resup(nbase[nblock], nbase[nblock - 1], style_channels[-1], sz, concatenation, dense_on=dense_on))
        else:
            self.up.add_module('conv_up_%d' % (nblock-1), convup(nbase[nblock], nbase[nblock - 1], style_channels[-1], sz, concatenation, dense_on=dense_on))

    def forward(self, style, xd, mkldnn=False):
        x = self.up[-1](xd[-1], xd[-1], style[-1], mkldnn=mkldnn)
        for n in range(len(self.up) - 2, -1, -1):
            if mkldnn:
                x = self.upsampling(x.to_dense()).to_mkldnn()
            else:
                x = self.upsampling(x)
            y_w=1  # control whether to use xd[n]

            if self.attn_on:
                x = self.up_conv[n](x)
                xdi = self.attn[n](x, xd[n], mkldnn=mkldnn)
                if mkldnn:
                    xdi = xdi.to_dense()
                    x = x.to_dense()
                    x = torch.cat((xdi, x), dim=1).to_mkldnn()
                else:
                    x = torch.cat((xdi, x), dim=1)
                y_w=0

            x = self.up[n](x, xd[n]*y_w, style[n], mkldnn=mkldnn)
        return x
