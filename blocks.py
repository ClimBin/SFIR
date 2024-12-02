import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False, freq=False):
        super(ResBlock, self).__init__()
        self.f = filter
        self.f2 = freq
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=False)
        self.mfa = MFA(in_channel,out_channel) if filter else nn.Identity()
        self.fre = EnhancedSpectralPhaseEnhancementModule(in_channel) if freq else nn.Identity()
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=True)

    def forward(self, x):
        x1 = self.conv1(x)
        if self.f is True:
            x1_1 = self.mfa(x1)
            if self.f2 is True:
                x1_1 = self.fre(x1_1)
            return self.conv2(x1_1) + x
        else:
            return self.conv2(x1) + x


class win_atten(nn.Module):
    def __init__(self, channel, window_size=7):
        super(win_atten, self).__init__()
        self.window_size = window_size
        self.attention = NonLocalBlock(channel) 

    def forward(self, x):

        N, C, H, W = x.shape
        window_size = self.window_size


        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size

        x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), "constant", 0)


        H_padded, W_padded = H + pad_h, W + pad_w


        x = x.view(N, C, H_padded // window_size, window_size, W_padded // window_size, window_size)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)

        x = self.attention(x) 

        x = x.view(N, H_padded // window_size, W_padded // window_size, C, window_size, window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(N, C, H_padded, W_padded)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, pad_h // 2: H + pad_h // 2, pad_w // 2: W + pad_w // 2]

        return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        self.query_conv = nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 2, 1, bias=False)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        batch_size, C, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key)
        attention = self.softmax(attention)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)

        return out


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialSelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return out


class MFA(nn.Module):
    def __init__(self,in_channel, out_channel, win_k=7):
        super(MFA, self).__init__()
        self.scale_sizes = [4,2]
        pools, convs, watts, sas = [],[],[],[]
        for i in self.scale_sizes:
            pools.append(nn.AvgPool2d(kernel_size=i, stride=i))
            convs.append(nn.Conv2d(in_channel,in_channel,1,bias=False))
            watts.append(win_atten(in_channel,win_k))
        self.pools = nn.ModuleList(pools)
        self.convs = nn.ModuleList(convs)
        self.watts = nn.ModuleList(watts)
        self.relu = nn.GELU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)
        self.fil_out = LowPassFilterModule(in_channels=in_channel,kernel_size=3)
        self.attention = SpatialSelfAttention(in_channel)

    def forward(self, x):
        x_size = x.size()
        resl = x
        for i in range(len(self.scale_sizes)):
            if i == 0:
                y = self.attention(self.watts[i](self.convs[i](self.pools[i](x))))
            else:
                y = self.watts[i](self.convs[i](self.pools[i](x)+y_up))
            resl = torch.add(resl, F.interpolate(y, x_size[2:], mode='bilinear', align_corners=True))
            if i != len(self.scale_sizes)-1:
                y_up = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        resl = self.relu(resl)
        resl = self.out_conv(self.fil_out(resl))
        return resl


class LowPassFilterModule(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(LowPassFilterModule, self).__init__()
        self.in_channels = in_channels


        self.raw_weights = nn.Parameter(
            torch.randn(in_channels, 1, kernel_size, kernel_size)
        )
        self.ks = kernel_size

        self.conv3x3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(self, x):

        raw_weights = self.raw_weights.view(self.in_channels, -1) 
        normalized_weights = F.softmax(raw_weights, dim=1)
        normalized_weights = normalized_weights.view(self.in_channels, 1, self.ks, self.ks)

        x_low = F.conv2d(x, normalized_weights, padding=self.ks//2, groups=self.in_channels)


        high_freq = x - x_low


        high_freq_conv = self.conv3x3(high_freq)

        out = x + high_freq_conv
        return out


class MPCA(nn.Module):
    def __init__(self, n_feat):
        super(MPCA, self).__init__()
        self.main = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        self.mag = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0),
        )

    def forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        mag_out = self.mag(mag)
        mag_res = mag_out - mag
        pooling = torch.nn.functional.adaptive_avg_pool2d(mag_res, (1, 1))
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')

        return self.main(x) + y


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class EnhancedSpectralPhaseEnhancementModule(nn.Module):
    def __init__(self, channel):
        super(EnhancedSpectralPhaseEnhancementModule, self).__init__()
        self.spectral_net = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )

        self.phase_net = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False)
        self.ca = ChannelAttentionModule(channel)

    def forward(self, x):

        fft_x = fft.rfft2(x, dim=(-2, -1))
        

        magnitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        

        magnitude_processed = self.spectral_net(magnitude)

        magnitude_processed = magnitude_processed * self.ca(magnitude_processed)
        

        batch_size, channels, height, width = phase.size()
        phase = phase.view(batch_size, channels, height, width)
        phase_processed = self.phase_net(phase)
        

        fft_processed = magnitude_processed * torch.exp(1j * phase_processed)

        x_reconstructed = fft.irfft2(fft_processed, s=x.size()[-2:], dim=(-2, -1))
        
        return x_reconstructed
    



class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True,freq=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True, freq=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()

        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x

class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()

        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))