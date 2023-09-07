"""Fractal Perlin noise generator and utilities."""

from typing import Optional, Tuple, Union

import torch

tau = 6.28318530718


# Adapted from pyperlin (https://github.com/duchesneaumathieu/pyperlin/tree/master)
class FractalPerlin2D(object):
    """2D Fractal Perlin noise generator.

    This class generates 2D fractal Perlin noise of any shape.
    If the shape requested is 3D, it will generate a batch of 2D noise,
    (This is different from the 3D Perlin noise generator which has some
    coherence between the slices of the 3D noise).
    """

    def __init__(
        self,
        shape: Union[Tuple[int, int], Tuple[int, int, int]],
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: int = 2,
        resolution: Tuple[int, int] = (1, 1),
        device: torch.device = torch.device("cpu"),
    ):
        """Fractal Perlin noise generator.

        Args:
            shape: The shape of the generated 2D noise. The shape can have 3 dimensions to account
                for the batch size.
            octaves: The number of octaves to use in the fractal noise.
                An octave is a layer of noise with a given frequency and amplitude.
                The higher the number of octaves, the more detailed the noise will be.
                There should be at least 1 octave.
            persistence: The persistence of the fractal noise. (factor of amplitude between two
                octaves). The persistence can be any positive value. Usually, the persistence is
                between 0 and 1 as we want the amplitude of higher octaves to be lower than the
                amplitude of lower octaves.
            lacunarity: The lacunarity of the fractal noise. (factor of frequency between two
                octaves). The lacunarity should be greater than 1. Usually, the lacunarity is
                set to 2 (similar to harmonic intervals in music).
            resolution: The resolution of the noise. (frequency of the first octave)
                The resolution should be greater than 0.
            device: The device on which to generate the noise.

        Raises:
            AssertionError: If the parameters are not valid.

        Warning:
            This will generate several tensors of shape (batch_size, height, width) where
            the height and the width are divisible by the resolution * lacunarity**(octaves-1).
            Depending on the parameters, this can lead to a large memory footprint.
        """
        assert octaves >= 1, "There should be at least 1 octave."
        assert persistence > 0, "The persistence should be positive."
        assert lacunarity > 1, "The lacunarity should be greater than 1."
        assert (
            resolution[0] > 0 and resolution[1] > 0
        ), "The resolution should be greater than 0."

        shape = shape if len(shape) == 3 else (None,) + shape
        self._desired_shape = shape

        # Correct shape so that it is valid
        # We want a shape greater or equal to the desired shape, but divisible by:
        # resolution * lacunarity**(octaves-1) and as small as possible to save memory.
        # i.e. given s and a, we want to find S such that:
        #     - Objective: minimize(S)
        #     - Constraints:
        #         - S >= s
        #         - S % a = 0
        # The solution is:
        #     S = s + (a - s % a) % a
        # In our case, s = shape[i+1] and a = resolution[i] * lacunarity**(octaves-1)
        shape = (
            shape[0],
            shape[1]
            + (
                resolution[0] * lacunarity ** (octaves - 1)
                - shape[1] % (resolution[0] * lacunarity ** (octaves - 1))
            )
            % (resolution[0] * lacunarity ** (octaves - 1)),
            shape[2]
            + (
                resolution[1] * lacunarity ** (octaves - 1)
                - shape[2] % (resolution[1] * lacunarity ** (octaves - 1))
            )
            % (resolution[1] * lacunarity ** (octaves - 1)),
        )
        self._shape = shape

        # Numerical parameters
        self._octaves = octaves
        self._persistence = persistence
        self._lacunarity = lacunarity
        self._resolution = resolution

        self._factors = [self._persistence**i for i in range(self._octaves)]
        self._generator = torch.Generator(device=device)
        self._device = device
        self._resolutions = [
            (
                self._resolution[0] * self._lacunarity**i,
                self._resolution[1] * self._lacunarity**i,
            )
            for i in range(self._octaves)
        ]
        self._grid_shapes = [
            (shape[1] // res[0], shape[2] // res[1]) for res in self._resolutions
        ]

        # precomputed tensors
        self.linxs = [
            torch.linspace(0, 1, gs[1], device=self._device) for gs in self._grid_shapes
        ]
        self.linys = [
            torch.linspace(0, 1, gs[0], device=self._device) for gs in self._grid_shapes
        ]
        self.tl_masks = [
            self.fade(lx)[None, :] * self.fade(ly)[:, None]
            for lx, ly in zip(self.linxs, self.linys)
        ]
        self.tr_masks = [torch.flip(tl_mask, dims=[1]) for tl_mask in self.tl_masks]
        self.bl_masks = [torch.flip(tl_mask, dims=[0]) for tl_mask in self.tl_masks]
        self.br_masks = [torch.flip(tl_mask, dims=[0, 1]) for tl_mask in self.tl_masks]

    def fade(self, t: float) -> float:
        """Fade function used to smooth the noise.

        Args:
            t: The input value.

        Returns:
            The smoothed value.
        """

        # Function introduced by Ken Perlin (Reference: https://en.wikipedia.org/wiki/Smoothstep)
        # Smoothstep is a polynomial approximation of the sigmoid function.
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def perlin_noise(self, octave: int, batch_size: int) -> torch.Tensor:
        """Generate Perlin noise for a given octave.

        Original function introduced by Ken Perlin.

        Args:
            octave: The octave for which to generate the noise.
            batch_size: The batch size.

        Returns:
            The generated noise.
        """
        res = self._resolutions[octave]
        angles = (
            torch.rand((batch_size, res[0] + 2, res[1] + 2), device=self._device) * tau
        )

        # This code is the main chunk of the algorithm
        # - Initializes angles
        # - Computes the gradients
        # - Generates gradients in different quadrants
        # - Calculates contribution of each gradient to the noise
        # - Combines contributions using the masks
        # - Reshapes the noise to the desired shape
        # - Applies offset for tiling (repeating pattern)
        rx = torch.cos(angles)[:, :, :, None] * self.linxs[octave]
        ry = torch.sin(angles)[:, :, :, None] * self.linys[octave]
        prx, pry = rx[:, :, :, None, :], ry[:, :, :, :, None]
        nrx, nry = -torch.flip(prx, dims=[4]), -torch.flip(pry, dims=[3])
        br = prx[:, :-1, :-1] + pry[:, :-1, :-1]
        bl = nrx[:, :-1, 1:] + pry[:, :-1, 1:]
        tr = prx[:, 1:, :-1] + nry[:, 1:, :-1]
        tl = nrx[:, 1:, 1:] + nry[:, 1:, 1:]

        grid_shape = self._grid_shapes[octave]
        grids = (
            self.br_masks[octave] * br
            + self.bl_masks[octave] * bl
            + self.tr_masks[octave] * tr
            + self.tl_masks[octave] * tl
        )
        noise = grids.permute(0, 1, 3, 2, 4).reshape(
            (batch_size, self._shape[1] + grid_shape[0], self._shape[2] + grid_shape[1])
        )

        A = torch.randint(
            0,
            grid_shape[0],
            (batch_size,),
            device=self._device,
            generator=self._generator,
        )
        B = torch.randint(
            0,
            grid_shape[1],
            (batch_size,),
            device=self._device,
            generator=self._generator,
        )
        noise = torch.stack(
            [
                noise[n, a : a - grid_shape[0], b : b - grid_shape[1]]
                for n, (a, b) in enumerate(zip(A, B))
            ]
        )
        return noise

    def __call__(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Generate the fractal Perlin noise.

        Args:
            batch_size: The batch size.

        Returns:
            The generated fractal noise.

        Raises:
            AssertionError: If the batch size is not specified in the shape or in the call.
        """
        if batch_size is None:
            assert (
                len(self._shape) == 3
            ), "The batch size should be specified either in the shape or in the call."
            batch_size = self._shape[0]
        shape = (batch_size,) + self._shape[1:]
        noise = torch.zeros(shape, device=self._device)
        for octave, factor in enumerate(self._factors):
            noise += factor * self.perlin_noise(octave, batch_size=batch_size)

        # Rescale to [0,1]
        noise = (noise + 1) / 2.0
        noise = torch.clamp(noise, 0, 1)

        # Crop to the desired shape
        noise = noise[:, : self._desired_shape[1], : self._desired_shape[2]]
        return noise

    def generate_between_bounds(
        self, batch_size: Optional[int] = None, bounds: Tuple[float, float] = (0.0, 1.0)
    ) -> torch.Tensor:
        """Generate the fractal Perlin noise between bounds.

        Args:
            batch_size: The batch size.
            bounds: The bounds of the generated noise.

        Returns:
            The generated fractal noise.
        """
        return self(batch_size=batch_size) * (bounds[1] - bounds[0]) + bounds[0]

    def generate_with_mean_and_std(
        self,
        batch_size: Optional[int] = None,
        mean_range: Tuple[float, float] = (0.0, 1.0),
        std_range: Tuple[float, float] = (0.0, 1.0),
        clamp: bool = True,
    ) -> torch.Tensor:
        """Generate the fractal Perlin noise with mean and std.

        Args:
            batch_size: The batch size.
            mean_range: The range of the mean of the generated noise.
            std_range: The range of the std of the generated noise.
            clamp: Whether to clamp the noise to [0, 1].

        Returns:
            The generated fractal noise.
        """
        noise = self(batch_size=batch_size)
        batch_size: int = noise.shape[0]

        # Generate mean and std
        mean: torch.Tensor = (
            torch.rand((batch_size, 1, 1), device=self._device)
            * (mean_range[1] - mean_range[0])
            + mean_range[0]
        )
        std: torch.Tensor = (
            torch.rand((batch_size, 1, 1), device=self._device)
            * (std_range[1] - std_range[0])
            + std_range[0]
        )

        # Rescale to [mean - std, mean + std]
        noise = noise * std * 2.0 + mean - std

        # Clamp to [0, 1]
        if clamp:
            noise = torch.clamp(noise, 0, 1)

        return noise
