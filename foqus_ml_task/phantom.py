import numpy as np


def ellipse_mask(shape: tuple[int, int], x: int, y: int, a: int, b: int) -> np.ndarray:
    """Creates a boolean mask in the shape of an ellipse.

    Args:
        shape: Shape of the mask array.
        x: Horizontal location of the ellipse.
        y: Vertical location of the ellipse.
        a: Horizontal semi-axis length.
        b: Vertical semi-axis length.

    Returns:
        Boolean mask of the ellipse.
    """
    # Make ellipse mask
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    x2 = xx.astype(np.float32) - x
    x2 *= x2
    y2 = yy.astype(np.float32) - y
    y2 *= y2
    mask = x2 / (a * a) + y2 / (b * b) < 1
    return mask


def generate_birdcage_sensitivities(
    n_coils: int, matrix_size: int, relative_radius: float = 1.5, normalize: bool = True
) -> np.ndarray:
    """Generates birdcage coil sensitivity maps.

    This implementation follows that of the ismrmrd-python-tools repo
    (https://github.com/ismrmrd/ismrmrd-python-tools), which in turn is based
    on Jeff Fessler's IRT package (http://web.eecs.umich.edu/~fessler/code/).

    Args:
        n_coils: Number of coils to create.
        matrix_size: Square size of the sensitivity maps.
        relative_radius: Radius of the circle along which coil "centers" are
            placed. Default is 1.5, meaning 1.5 * matrix_size.
        normalize: When True, sensitivity maps are normalized by the RSS of the
            sensitivity maps. Default is True.

    Returns:

    """
    coil_indices = np.arange(n_coils, dtype=np.float32)
    coil_x = relative_radius * np.cos(coil_indices * (2 * np.pi / n_coils))
    coil_y = relative_radius * np.sin(coil_indices * (2 * np.pi / n_coils))
    coil_phase = -coil_indices * (2 * np.pi / n_coils)

    yy, xx = np.mgrid[:matrix_size, :matrix_size]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    xx_coil = 2 * xx[None, ...] / matrix_size - 1 - coil_x[:, None, None]
    yy_coil = 2 * yy[None, ...] / matrix_size - 1 - coil_y[:, None, None]
    rr = np.sqrt(xx_coil**2 + yy_coil**2)
    phi = np.arctan2(xx_coil, -yy_coil) + coil_phase[:, None, None]

    sensitivity_maps = (1 / rr) * np.exp(1j * phi)

    if normalize:
        rss = root_sum_of_squares(sensitivity_maps, keepdims=True)
        sensitivity_maps /= rss

    return sensitivity_maps


class RandomPhantomGenerator:
    INTENSITY_RANGES = {
        "skull": (0.9, 1),
        "background": (0, 0.1),
        "brain": (0.4, 0.6),
        "dark_ellipse": (0.1, 0.3),
        "light_ellipse": (0.7, 0.9),
    }
    HEAD_MAJOR_AX_RANGE = (0.35, 0.5)
    HEAD_ASPECT_RATIO_RANGE = (1.2, 1.6)
    NUM_ELLIPSE_RANGE = (4, 8)
    MAX_ELLIPSE_ASPECT_RATIO = 2.5
    ELLIPSE_SIZE_RANGE = (0.1, 0.3)
    NOISE_STD = 0.025

    def __init__(self, n_coils: int = 8, size: int = 256):
        """Initializes a random phantom generator.

        Args:
            n_coils: Number of coils in each phantom. Default is 8.
            size: Square image size of the phantoms. Default is 256.
        """
        self.n_coils = n_coils
        self.size = size

    def generate_phantom(self, seed: int | None = None) -> np.ndarray:
        """Generates a random brain phantom.

        Args:
            seed: Integer seed to use for reproducibility. Default is None.

        Returns:
            A phantom MRI image with shape (coils, height, width).
        """
        # Create seeded rng for reproducibility
        rng = np.random.default_rng(seed)

        # Fill background
        image = np.ones((self.size, self.size), dtype=np.float32)
        image *= rng.uniform(*self.INTENSITY_RANGES["background"])

        # Draw skull and brain
        brain_mask, x_brain, y_brain, a_brain, b_brain = self._draw_head(image, rng)

        # Draw random light/dark ellipses in brain
        for _ in range(rng.integers(*self.NUM_ELLIPSE_RANGE, endpoint=True)):
            a = b_brain * rng.uniform(*self.ELLIPSE_SIZE_RANGE)
            b = a * rng.uniform(1 / self.MAX_ELLIPSE_ASPECT_RATIO, self.MAX_ELLIPSE_ASPECT_RATIO)
            x = rng.uniform(x_brain - a_brain + a, x_brain + a_brain - a)
            y = rng.uniform(y_brain - b_brain + b, y_brain + b_brain - b)
            mask = ellipse_mask(image.shape, x, y, a, b) & brain_mask

            intensity_type = "light" if rng.uniform() < 0.5 else "dark"
            image[mask] = rng.uniform(*self.INTENSITY_RANGES[intensity_type + "_ellipse"])

        # Add random noise
        image += rng.normal(loc=0, scale=self.NOISE_STD, size=image.shape)

        # Generate sensitivity maps and apply them to produce coil images
        sens_maps = generate_birdcage_sensitivities(self.n_coils, self.size)
        coil_images = sens_maps * image[None]

        return coil_images

    def _draw_head(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Draws the skull and brain background."""
        b = rng.uniform(*self.HEAD_MAJOR_AX_RANGE) * image.shape[1]
        a = b / rng.uniform(*self.HEAD_ASPECT_RATIO_RANGE)
        max_dx = (image.shape[1] - 2 * a) / 8
        max_dy = (image.shape[0] - 2 * b) / 8
        x = image.shape[1] / 2 + rng.uniform(-max_dx, max_dx)
        y = image.shape[0] / 2 + rng.uniform(-max_dy, max_dy)
        outer_mask = ellipse_mask(image.shape, x, y, a, b)
        a_brain, b_brain = 0.95 * a, 0.95 * b
        brain_mask = ellipse_mask(image.shape, x, y, a_brain, b_brain)
        image[outer_mask] = rng.uniform(*self.INTENSITY_RANGES["skull"])
        image[brain_mask] = rng.uniform(*self.INTENSITY_RANGES["brain"])
        return brain_mask, x, y, a_brain, b_brain


def root_sum_of_squares(
    coil_images: np.ndarray, coil_axis: int = 0, keepdims: bool = False
) -> np.ndarray:
    """Applies root-sum-of-squares to a complex coil image array.

    Args:
        coil_images: Set of coil images.
        coil_axis: Coil axis of the `coil_images` array. Default is 0.
        keepdims: Flag to keep the coil axis after summation. Default is False.

    Returns:
        RSS image(s). If `keepdims=True`, the coil axis will become a singleton
            axis, otherwise the coil axis is removed.
    """
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=coil_axis, keepdims=keepdims))
