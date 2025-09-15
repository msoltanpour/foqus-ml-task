from collections.abc import Callable
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


TransformType = Callable[[torch.Tensor, ], torch.Tensor]


class Normalize(nn.Module):
    """
    Per-sample normalization for multi-coil complex data.

    Input  : (C, H, W, 2) where the last dim is [real, imag]
    Strategy:
      - compute magnitude = sqrt(real^2 + imag^2)
      - compute mean/std over the whole sample (all coils & pixels)
      - apply (x - mean)/std to BOTH real & imag using magnitude stats
    """

    def __init__(self, eps: float = 1e-8, clip: float | None = None):
        super().__init__()
        self.eps = eps
        self.clip = clip

    def forward(self, coil_images: torch.Tensor) -> torch.Tensor:
        if not isinstance(coil_images, torch.Tensor):
            coil_images = torch.as_tensor(coil_images)

        if coil_images.ndim != 4 or coil_images.shape[-1] != 2:
            raise ValueError(f"Normalize expects (C,H,W,2), got {tuple(coil_images.shape)}")

        real = coil_images[..., 0]
        imag = coil_images[..., 1]
        mag = torch.sqrt(real ** 2 + imag ** 2)

        mean = mag.mean()
        std = mag.std()
        if std < self.eps:
            std = torch.tensor(1.0, dtype=coil_images.dtype, device=coil_images.device)

        real_n = (real - mean) / std
        imag_n = (imag - mean) / std
        out = torch.stack((real_n, imag_n), dim=-1)

        if self.clip is not None and self.clip > 0:
            out = torch.clamp(out, -self.clip, self.clip)
        return out




class EquispacedUndersample(nn.Module):
    """
    Equispaced undersampling in k-space with a centered low-frequency band.

    Args:
        acceleration (int): keep every `acceleration`-th line along width
        center_fraction (float): fraction of width fully sampled at center, (0,1]
    """
    def __init__(self, acceleration: int, center_fraction: float):
        super().__init__()
        if acceleration < 1:
            raise ValueError("acceleration must be >= 1")
        if not (0.0 < center_fraction <= 1.0):
            raise ValueError("center_fraction must be in (0, 1]")
        self.acceleration = int(acceleration)
        self.center_fraction = float(center_fraction)

    def forward(self, coil_images: torch.Tensor) -> torch.Tensor:
        # Expect (C, H, W, 2). Uses fft/ifft helpers already in this file.
        if coil_images.ndim != 4 or coil_images.shape[-1] != 2:
            raise ValueError(f"EquispacedUndersample expects (C,H,W,2), got {tuple(coil_images.shape)}")

        k = fft(coil_images)                 # -> (C, H, W, 2)
        _, _, W, _ = k.shape

        # Build 1D mask over width
        mask = torch.zeros(W, dtype=torch.bool, device=k.device)
        mask[:: self.acceleration] = True

        num_low = max(1, int(round(W * self.center_fraction)))
        mid = W // 2
        start = max(0, mid - num_low // 2)
        end = min(W, start + num_low)
        mask[start:end] = True

        # Apply mask: broadcast to (1,1,W,1) and zero unsampled lines
        k = k * mask.view(1, 1, W, 1)

        return ifft(k)



class Augmentation(nn.Module):
    """
    Light image-domain augmentations for complex data:
      - small rotation (±max_rot_deg)
      - small translation (±max_shift_px)
    Applied identically to real & imag channels.
    """

    def __init__(self, max_rot_deg: float = 10.0, max_shift_px: float = 2.0):
        super().__init__()
        self.max_rot = float(max_rot_deg)
        self.max_shift = float(max_shift_px)

    def forward(self, coil_images: torch.Tensor) -> torch.Tensor:
        # Expect (C, H, W, 2)
        if coil_images.ndim != 4 or coil_images.shape[-1] != 2:
            raise ValueError(f"Augmentation expects (C,H,W,2), got {tuple(coil_images.shape)}")

        C, H, W, _ = coil_images.shape
        # pack to (1, C*2, H, W) so a single grid_sample hits real+imag together
        x = coil_images.permute(0, 3, 1, 2).reshape(1, C * 2, H, W)

        # random small rotation (radians) and translation (normalized coords)
        theta_deg = (torch.empty(1).uniform_(-self.max_rot, self.max_rot).item())
        theta = math.radians(theta_deg)
        tx = torch.empty(1).uniform_(-self.max_shift, self.max_shift).item() / (W / 2)
        ty = torch.empty(1).uniform_(-self.max_shift, self.max_shift).item() / (H / 2)

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        A = torch.tensor([[cos_t, -sin_t, tx],
                          [sin_t,  cos_t, ty]], dtype=x.dtype, device=x.device).unsqueeze(0)

        grid = F.affine_grid(A, size=x.size(), align_corners=False)
        x_aug = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

        # unpack back to (C, H, W, 2)
        x_aug = x_aug.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous()
        return x_aug




class ToCompatibleTensor(nn.Module):
    """
    Convert (C, H, W, 2) -> (channels, H, W) where channels = C*2 (real & imag stacked).

    Rationale:
      - Keeps phase info explicit.
      - Simple, model-agnostic; works with standard Conv2d.
      - Trade-off: doubles channels vs magnitude-only or coil-combined inputs.
    """
    def forward(self, coil_images: torch.Tensor) -> torch.Tensor:
        if coil_images.ndim != 4 or coil_images.shape[-1] != 2:
            raise ValueError(f"ToCompatibleTensor expects (C,H,W,2), got {tuple(coil_images.shape)}")
        C, H, W, _ = coil_images.shape
        # (C, H, W, 2) -> (C*2, H, W)
        out = coil_images.permute(0, 3, 1, 2).reshape(C * 2, H, W).contiguous()
        return out


# ========== HELPER/UTILITY FUNCTIONS ==========

def fft(
    data: torch.Tensor,
    dim: tuple[int, int] = (1, 2),
) -> torch.Tensor:
    """Applies the Fast Fourier Transform to a tensor.

    Args:
        data: Input tensor. For complex-valued tensors, any shape is
            accepted. For real-valued tensors, the last dimension must have
            length 2, representing the real/imaginary parts.
        dim: Dimensions along which to apply the FFT. Default is (1, 2),
            yielding a 2D FFT.

    Returns:
        The FFT of the data. With the same shape and type as the input tensor.
    """
    return _fft_base(data, dim, False)


def ifft(
    data: torch.Tensor,
    dim: tuple[int, int] = (1, 2),
) -> torch.Tensor:
    """Applies the inverse Fast Fourier Transform to a tensor.

    Args:
        data: Input tensor. For complex-valued tensors, any shape is
            accepted. For real-valued tensors, the last dimension must have
            length 2, representing the real/imaginary parts.
        dim: Dimensions along which to apply the IFFT. Default is (1, 2),
            yielding a 2D IFFT.

    Returns:
        The IFFT of the data with the same shape and type as the input tensor.
    """
    return _fft_base(data, dim, True)


def _fft_base(
    data: torch.Tensor,
    dim: tuple[int, int],
    inverse: bool,
) -> torch.Tensor:
    """Apply centered Fast Fourier Transform or its inverse."""
    complex_input = torch.is_complex(data)
    if not complex_input:
        data = torch.view_as_complex(data)

    fft_func = torch.fft.ifftn if inverse else torch.fft.fftn
    data = torch.fft.ifftshift(data, dim=dim)
    data = fft_func(data, dim=dim, norm="ortho")
    data = torch.fft.fftshift(data, dim=dim)

    if not complex_input:
        data = torch.view_as_real(data)

    return data
