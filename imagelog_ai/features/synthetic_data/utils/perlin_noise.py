import torch
import math


class PerlinNoise2D(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = None  # torch.Generator(device=self.device)

        self.sqrt_2 = math.sqrt(2)
        self.tau = 6.28318530718

    @staticmethod
    def fade(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def perlin_noise(
        self, shape, resolutions, tileable=(False, False), interpolant=fade
    ):
        delta = (resolutions[0] / shape[0], resolutions[1] / shape[1])
        d = (shape[0] // resolutions[0], shape[1] // resolutions[1])

        grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, resolutions[0], delta[0], device=self.device),
                    torch.arange(0, resolutions[1], delta[1], device=self.device),
                    indexing="ij",
                ),
                dim=-1,
            )
            % 1
        )

        angles = (
            2
            * math.pi
            * torch.rand(resolutions[0] + 1, resolutions[1] + 1, device=self.device)
        )
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

        tile_grads = (
            lambda slice1, slice2: gradients[
                slice1[0] : slice1[1], slice2[0] : slice2[1]
            ]
            .repeat_interleave(d[0], 0)
            .repeat_interleave(d[1], 1)
        )
        dot = lambda grad, shift: (
            torch.stack(
                (
                    grid[: shape[0], : shape[1], 0] + shift[0],
                    grid[: shape[0], : shape[1], 1] + shift[1],
                ),
                dim=-1,
            )
            * grad[: shape[0], : shape[1]]
        ).sum(dim=-1)

        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        t = interpolant(grid[: shape[0], : shape[1]])

        return self.sqrt_2 * torch.lerp(
            torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1]
        )

    def generate_fractal_noise_2d(
        self,
        shape,
        resolutions,
        octaves=1,
        persistence=0.5,
        lacunarity=2,
        tileable=(False, False),
    ):
        noise = torch.zeros(shape, device=self.device)
        frequency = 1
        amplitude = 1
        for _ in range(octaves):
            noise += amplitude * self.perlin_noise(
                shape,
                (frequency * resolutions[0], frequency * resolutions[1]),
                tileable,
            )
            frequency *= lacunarity
            amplitude *= persistence
        return noise
