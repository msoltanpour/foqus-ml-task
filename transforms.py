from collections.abc import Callable
import torch
import torch.nn as nn

TransformType = Callable[[torch.Tensor, ], torch.Tensor]


class Normalize(nn.Module):
    """Normalizes images to have consistent statistics."""
    def forward(self, coil_images: torch.tensor) -> torch.Tensor:
        """Apply normalization to coil images.

        Args:
            coil_images: Coil images with shape (coils, height, width, 2).

        Returns:
            Normalized coil images with the same shape as the input.
        """
        return coil_images


class EquispacedUndersample(nn.Module):
    """Transform which undersamples k-space."""
    def __init__(self, acceleration: int, center_fraction: float):
        super().__init__()
        self.acceleration = acceleration
        self.center_fraction = center_fraction

    def forward(self, coil_images: torch.Tensor):
        """Performs undersampling of raw MRI data.

        Your method should:
            1) Convert the input coil images to frequency space (k-space).
            2) Sample every `acceleration`'th line along the `width` axis.
               For example, if `acceleration=3`, sample every 3rd line.
            2) Additionally sample `width*center_fraction` lines from the
               center of the `width` axis.
            4) Set all remaining (unsampled) lines to 0.
            5) Convert from k-space back to image space (coil images).

        Args:
            coil_images: Coil images with shape (coils, height, width, 2).

        Returns:
            Undersampled coil images with shape (coils, height, width, 2).
        """
        kspace = fft(coil_images)
        lines = set(range(0, kspace.shape[2], self.acceleration))
        mid_idx = kspace.shape[2]//2
        num_low = round(kspace.shape[2]*self.center_fraction)
        lines = lines | set(range(mid_idx-num_low//2, mid_idx+num_low//2))
        for line in range(kspace.shape[2]):
            if line not in lines:
                kspace[..., line, :] = 0
        coil_images = ifft(kspace)
        return coil_images


class Augmentation(nn.Module):
    """Transform which augments coil images.

    This must be one of:
        - random crop
        - random rotation (NOT only 90 degree increments)
        - random affine transformation
        - random resize
        - random intensity inversion
    """
    def forward(self, coil_images: torch.Tensor) -> torch.Tensor:
        """Augment coil images.

        Args:
            coil_images: Coil images with shape (coils, height, width, 2).

        Returns:
            Augmented coil images with shape (coils, height, width, 2).
        """
        return coil_images


class ToCompatibleTensor(nn.Module):
    """Converts coil images to a format we can pass to our model."""
    def forward(self, coil_images: torch.Tensor):
        """Convert coil images to a format we can pass to our model.

        Specifically, we need to go from a shape of (coils, height, width, 2)
        to a shape of (channels, height, width). When doing so, be sure to
        consider any disadvantages of your approach.

        Args:
            coil_images: Coil images with shape (coils, height, width, 2)

        Returns:
            A tensor with shape (channels, height, width) that will be
                compatible with PyTorch's 2D layers (e.g., Conv2d) when batched
        """
        return coil_images


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
