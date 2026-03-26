import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

class DDAM(nn.Module):
    def __init__(self, channels, reduction=1, bias=False):
        super(DDAM, self).__init__()
        self.channels = channels

        self.sobel_kernel_x = torch.tensor([[[[1., 0., -1.],
                                              [2., 0., -2.],
                                              [1., 0., -1.]]]], dtype=torch.float32) / 4.0
        self.sobel_kernel_y = torch.tensor([[[[1., 2., 1.],
                                              [0., 0., 0.],
                                              [-1., -2., -1.]]]], dtype=torch.float32) / 4.0

        self.register_buffer('kernel_x', self.sobel_kernel_x)
        self.register_buffer('kernel_y', self.sobel_kernel_y)

        self.conv= conv(channels, channels, kernel_size=3, bias=bias)

    def forward(self, x_spatial):
        batch, c, h, w = x_spatial.shape

        # Gradient map
        grad_x = F.conv2d(x_spatial, self.kernel_x.repeat(c, 1, 1, 1), padding=1, groups=c)
        grad_y = F.conv2d(x_spatial, self.kernel_y.repeat(c, 1, 1, 1), padding=1, groups=c)

        K = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # K: [B, C, H, W]

        # Original spatial feature
        Q = x_spatial  # Q: [B, C, H, W]

        # Frequency feature
        x_freq = torch.fft.rfft2(x_spatial, norm='backward')

        amplitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        V = amplitude  # V: [B, C, H, W//2+1]

        # Dual-domain Attention
        K_flat = K.view(batch, c, -1)  # [B, C, H*W]
        Q_flat = Q.view(batch, c, -1)  # [B, C, H*W]
        V_flat = V.view(batch, c, -1)  # [B, C, (H)*(W//2+1)]

        # attn = (softmax((K^T * Q) / sqrt(d_k))) * V
        d_k = c
        attention_scores = torch.bmm(K_flat, Q_flat.transpose(1, 2)) / (d_k ** 0.5)  # [B, H*W, H*W]
        attention_weights = F.softmax(attention_scores, dim=-1)  # F_attn: [B, H*W, H*W]
        attended_freq_flat = torch.bmm(attention_weights, V_flat)  # [B, C, (H)*(W//2+1)]

        attended_freq = attended_freq_flat.view(batch, c, h, w // 2 + 1)

        attended_freq_conv = self.conv(attended_freq)

        x_freq_attended = torch.polar(attended_freq_conv, phase)

        x_spatial_attended = torch.fft.irfft2(x_freq_attended, s=(h, w), norm='backward')

        return x_spatial_attended