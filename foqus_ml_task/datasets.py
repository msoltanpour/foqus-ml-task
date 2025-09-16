import random

import torch
from torch.utils.data import Dataset

from .transforms import TransformType
from phantom import RandomPhantomGenerator


class RandomPhantomDataset(Dataset):
    """Simple dataset which generates deterministic phantom head MRI images."""
    def __init__(
        self,
        length: int = 1000,
        n_coils: int = 8,
        image_size: int = 256,
        offset: int = 0,
        transforms: list[TransformType] | None = None,
    ):
        """Initialize a random phantom dataset.

        Args:
            length: Length of the dataset. Default is 1000.
            n_coils: Number of coils in each phantom. Default is 8.
            image_size: In-plane image size. Default is 256.
            offset: An optional offset to be added to requested indices in
                __getitem__. This allows for the creation of datasets that do
                not have any of the same samples. Default is 0.
            transforms: List of callable transformations to apply.
                Default is None (no transforms).
        """
        self.length = length
        self.n_coils = n_coils
        self.image_size = image_size
        self.offset = offset
        self.transforms = transforms or []
        self.generate = RandomPhantomGenerator(
            n_coils=n_coils, size=image_size
        ).generate_phantom

    def validate_index(self, idx: int) -> int:
        """Ensures an index is valid and non-negative.

        Args:
            idx: Input index.

        Returns:
            The input index if it is non-negative, otherwise its positive
                equivalent.
        """
        new_idx = idx
        if idx < 0:
            new_idx += self.length
        if not (0 <= new_idx < self.length):
            raise IndexError(
                f"Index {idx} is out of range for "
                f"dataset with length {self.length}"
            )
        return new_idx

    def __getitem__(self, item: int) -> torch.Tensor:
        """Get an item from the dataset.

        Args:
            item: Index of the item in the dataset.

        Returns:
            Complex coil images with shape (coils, image_size, image_size, 2).
        """
        # Generate base random phantom
        # This is a real-valued complex tensor with shape
        # (coils, image_size, image_size, 2) - the last axis is the
        # real/imaginary parts
        coil_images = self.generate(self.validate_index(item) + self.offset)
        coil_images = torch.view_as_real(torch.from_numpy(coil_images))

        # Apply transforms
        for transform in self.transforms:
            coil_images = transform(coil_images)

        return coil_images

    def __len__(self):
        """Length of the dataset."""
        return self.length


class RandomPhantomTripletDataset(Dataset):
    """Dataset providing triplets of images for training embedding models."""
    def __init__(
        self,
        length: int = 1000,
        n_coils: int = 8,
        image_size: int = 256,
        offset: int = 0,
        deterministic: bool = False,
        transforms1: list[TransformType] | None = None,
        transforms2: list[TransformType] | None = None,
    ):
        """Initialize image triplet dataset.

        Args:
            length: Length of the dataset. Default is 1000.
            n_coils: Number of coils in each phantom. Default is 8.
            image_size: In-plane image size. Default is 256.
            offset: An optional offset to be added to requested indices in
                __getitem__. This allows for the creation of datasets that do
                not have any of the same samples. Default is 0.
            deterministic: When True, the third image in the triplet is sampled
                deterministically, otherwise it is sampled randomly. This is
                useful for validation. Default is False.
            transforms1: List of callable transformations to apply to the 1st
                sample. Default is None (no transforms).
            transforms2: List of callable transformations to apply to the 2nd
                sample. Default is None (no transforms).
        """
        self.dataset = RandomPhantomDataset(
            length=length,
            n_coils=n_coils,
            image_size=image_size,
            offset=offset,
            transforms=None,
        )
        self.deterministic = deterministic
        self.transforms1 = transforms1 or []
        self.transforms2 = transforms2 or []

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gets two versions the same an image and a third, different image."""
        # Retrieve the requested sample
        item = self.dataset.validate_index(item)
        same1 = same2 = self.dataset[item]

        # Retrieve a different sample, possibly deterministically
        if self.deterministic:
            rng = random.Random(item)
        else:
            rng = random.Random()
        n = self.dataset.length
        diff = self.dataset[(item + rng.randint(1, n - 1)) % n]
        coin_flip = rng.random() < 0.5

        # Apply transforms
        for transform in self.transforms1:
            same1 = transform(same1)
        for transform in self.transforms2:
            same2 = transform(same2)
        for transform in (self.transforms1 if coin_flip else self.transforms2):
            diff = transform(diff)

        return same1, same2, diff

    def __len__(self):
        """Length of the dataset."""
        return self.dataset.length
