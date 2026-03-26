import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff ** 2) + (self.epsilon ** 2)))
        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)
        down = filtered[:, :, ::2, ::2]
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down*4
        filtered = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


class FrequencyLoss(nn.Module):
    def __init__(self, epsilon=1e-3, amp_weight=1.0, phase_weight=0.5, consistency_weight=0.05):
        super(FrequencyLoss, self).__init__()
        self.epsilon = epsilon
        self.amp_weight = amp_weight
        self.phase_weight = phase_weight
        self.consistency_weight = consistency_weight

    def fft_shift(self, x, mode='forward'):
        b, c, h, w = x.shape
        shift_size_h = h // 2
        shift_size_w = w // 2
        if mode == 'forward':
            return torch.roll(x, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))
        elif mode == 'backward':
            return torch.roll(x, shifts=(shift_size_h, shift_size_w), dims=(2, 3))
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def fft_transform(self, x):
        x_shifted = self.fft_shift(x, 'forward')
        fft_complex = torch.fft.fft2(x_shifted, norm='ortho')
        amplitude = torch.abs(fft_complex)
        phase = torch.angle(fft_complex)
        return amplitude, phase

    def amplitude_loss(self, pred_amp, target_amp):
        diff = pred_amp - target_amp
        loss = torch.mean(torch.sqrt((diff ** 2) + (self.epsilon ** 2)))
        return loss

    def phase_loss(self, pred_phase, target_phase):
        phase_diff = torch.atan2(torch.sin(pred_phase - target_phase),
                                 torch.cos(pred_phase - target_phase))
        loss = torch.mean(torch.abs(phase_diff))
        return loss

    def spectrum_consistency_loss(self, pred, target):
        pred_amp, _ = self.fft_transform(pred)
        target_amp, _ = self.fft_transform(target)

        pred_hist = torch.histc(pred_amp.flatten(), bins=100, min=0, max=pred_amp.max().item())
        target_hist = torch.histc(target_amp.flatten(), bins=100, min=0, max=target_amp.max().item())

        pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
        target_hist = target_hist / (target_hist.sum() + 1e-8)

        consistency_loss = F.mse_loss(pred_hist, target_hist)
        return consistency_loss

    def forward(self, pred_img, target_img):
        pred_amp, pred_phase = self.fft_transform(pred_img)
        target_amp, target_phase = self.fft_transform(target_img)

        amp_loss = self.amplitude_loss(pred_amp, target_amp)
        phase_loss = self.phase_loss(pred_phase, target_phase)
        consistency_loss = self.spectrum_consistency_loss(pred_img, target_img)

        total_loss = (self.amp_weight * amp_loss +
                      self.phase_weight * phase_loss +
                      self.consistency_weight * consistency_loss)

        return total_loss, {
            'amp_loss': amp_loss.item() if torch.is_tensor(amp_loss) else amp_loss,
            'phase_loss': phase_loss.item() if torch.is_tensor(phase_loss) else phase_loss,
            'consistency_loss': consistency_loss.item() if torch.is_tensor(consistency_loss) else consistency_loss,
            'total_freq_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss
        }


class TotalLoss(nn.Module):
    def __init__(self, char_weight=1.0, edge_weight=0.05, freq_weight=0.05):
        super(TotalLoss, self).__init__()
        self.char_weight = char_weight
        self.edge_weight = edge_weight
        self.freq_weight = freq_weight

        self.char_loss = CharbonnierLoss()
        self.edge_loss = EdgeLoss()
        self.freq_loss = FrequencyLoss()

    def forward(self, pred_imgs, target_img):
        if not isinstance(pred_imgs, list):
            pred_imgs = [pred_imgs]

        total_char_loss = 0
        total_edge_loss = 0
        total_freq_loss = 0

        freq_loss_details = []

        for i, pred_img in enumerate(pred_imgs):
            if target_img.shape[-2:] != pred_img.shape[-2:]:
                target_resized = F.interpolate(target_img, size=pred_img.shape[-2:],
                                               mode='bilinear', align_corners=False)
            else:
                target_resized = target_img

            char_loss = self.char_loss(pred_img, target_resized)
            edge_loss = self.edge_loss(pred_img, target_resized)

            freq_loss, freq_details = self.freq_loss(pred_img, target_resized)
            freq_loss_details.append(freq_details)

            total_char_loss += char_loss
            total_edge_loss += edge_loss
            total_freq_loss += freq_loss

        total_loss = (self.char_weight * total_char_loss +
                      self.edge_weight * total_edge_loss +
                      self.freq_weight * total_freq_loss)

        loss_details = {
            'char_loss': total_char_loss.item() if torch.is_tensor(total_char_loss) else total_char_loss,
            'edge_loss': total_edge_loss.item() if torch.is_tensor(total_edge_loss) else total_edge_loss,
            'freq_loss': total_freq_loss.item() if torch.is_tensor(total_freq_loss) else total_freq_loss,
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'freq_details': freq_loss_details
        }

        return total_loss, loss_details