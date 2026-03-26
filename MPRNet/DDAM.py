import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                    padding=(kernel_size//2), bias=bias, stride=stride)

class DDAM(nn.Module):
    def __init__(self, channels, reduction=1, bias=False):
        super().__init__()
        self.channels = channels

        # Sobel kernels for gradient
        self.sobel_kernel_x = torch.tensor([[[[1., 0., -1.],
                                              [2., 0., -2.],
                                              [1., 0., -1.]]]], dtype=torch.float32) / 4.0
        self.sobel_kernel_y = torch.tensor([[[[1., 2., 1.],
                                              [0., 0., 0.],
                                              [-1., -2., -1.]]]], dtype=torch.float32) / 4.0
        self.register_buffer('kernel_x', self.sobel_kernel_x)
        self.register_buffer('kernel_y', self.sobel_kernel_y)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # CAB
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

        self.conv_fuse = conv(channels, channels, kernel_size=3, bias=bias)

    def forward(self, x_spatial):
        batch, c, h, w = x_spatial.shape

        # gradient
        grad_x = F.conv2d(x_spatial, self.kernel_x.repeat(c, 1, 1, 1), padding=1, groups=c)
        grad_y = F.conv2d(x_spatial, self.kernel_y.repeat(c, 1, 1, 1), padding=1, groups=c)
        F_grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)  # [B, C, H, W]

        # frequency
        x_freq = torch.fft.rfft2(x_spatial, norm='ortho')
        amplitude = torch.abs(x_freq)  # [B, C, H, W//2+1]
        phase = torch.angle(x_freq)    # [B, C, H, W//2+1]

        grad_desc = self.avg_pool(F_grad).view(batch, c)          # [B, C]

        freq_desc = torch.mean(amplitude, dim=[2,3])               # [B, C]

        dual_desc = torch.cat([grad_desc, freq_desc], dim=1)      # [B, 2*C]
        ca_weights = self.fc(dual_desc).view(batch, c, 1, 1)      # [B, C, 1, 1]

        modulated_spatial = x_spatial * ca_weights  # [B, C, H, W]

        modulated_freq = torch.fft.rfft2(modulated_spatial, norm='ortho')
        mod_amplitude = torch.abs(modulated_freq)

        fused_freq = torch.polar(amplitude + 0.5 * mod_amplitude, phase)

        x_spatial_attended = torch.fft.irfft2(fused_freq, s=(h, w), norm='ortho')

        output = self.conv_fuse(x_spatial_attended)

        return output