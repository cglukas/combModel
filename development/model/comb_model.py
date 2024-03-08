"""Model"""
import torch
from torch import nn

from development.model.constants import RELU_ALPHA
from development.model.encoder import Encoder


class Decoder(nn.Module):
    """Decoder part of the encoder-decoder model.

    This model tries to reconstruct the input image of the encoder
    based on the latent vector that the encoder produced.
    """

    def __init__(self):
        super().__init__()
        self.levels = []
        self.rgb_levels = []
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
        )  # 8
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
        )  # 7
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
        )  # 6
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
        )  # 5
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
        )  # 4
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
        )  # 3
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
        )  # 2
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
        )  # 1
        self.add_level(
            nn.Conv2d(512, 512, 4, padding=3),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
        )  # 0
        self.levels = nn.ParameterList(self.levels)  # Fix missing parameters
        self.rgb_levels = nn.ParameterList(self.rgb_levels)  # Fix missing parameters

    def add_level(self, *modules: nn.Module):
        """Add the modules as a level to the model.

        Note:
            The levels will be pushed to the start of the levels list.
            This means you need to construct the network from the last layer to the first.

            It's also necessary that the output on layer L0 can be passed to L1 and so on.

        Args:
            *modules: all the modules that should be part of the layer.
        """
        self.rgb_levels.insert(
            0,
            nn.Sequential(
                torch.nn.Conv2d(modules[-2].out_channels, 3, 1), nn.Sigmoid()
            ),
        )
        modules = nn.Sequential(*modules)
        self.levels.insert(0, modules)

    def from_rgb(self, tensor, level):
        """Pass the tensor through the intermediate rgb transform layer.

        This rgb processing is necessary to transform the representation
        in the different levels to a rgb image that can be compared and
        used for backpropagation.
        All of these layers except the last one will be dropped in the
        trained model.

        Args:
            tensor: batch of latent representations.
            level: current level, used for selecting the correct rgb transform layer.

        Returns:
            Reconstructed rgb image from the representation of the level.
        """
        rgb_conversion = self.rgb_levels[level]
        return rgb_conversion(tensor)

    def forward(self, latent, level):
        """Reconstruct the images until the level is reached.

        Args:
            latent: batch of latent vectors.
            level: desired output level.

        Returns:
            Batch of reconstructed images.
        """
        modules = self.levels[: level + 1]
        for module in modules:
            latent = module(latent)
        image = self.from_rgb(latent, level)
        return image

    def progressive_forward(self, latent, level, last_lvl_influence: float = 0.0):
        """Reconstruct the input image of the encoder.

        Args:
            latent: batch of latent vectors from the encoder.
            level: current level that should be outputted.
            last_lvl_influence: blend between the last level and the current level.
                                Helps to transition the levels while training.

        Returns:
            Batch of reconstructed images
        """
        if level == 0:
            return self.forward(latent, level)

        small_img = self.forward(latent, level - 1)
        img = self.forward(latent, level)
        small_img = nn.functional.interpolate(small_img, scale_factor=2)
        img = last_lvl_influence * img + (1 - last_lvl_influence) * small_img

        return img


class CombModel(nn.Module):
    """Implementation of the comb model from the disney research paper.

    This model allows parallel training of multiple identities that will be encoded in the same latent space.

    Paper:
        https://studios.disneyresearch.com/2020/06/29/high-resolution-neural-face-swapping-for-visual-effects/
    """

    def __init__(self, persons: int = 2, device="cpu"):
        super().__init__()
        self.encoder = Encoder()
        self.encoder.to(device)
        self.decoders = nn.ModuleList()
        self.latent: torch.Tensor
        for _ in range(persons):
            decoder = Decoder()
            decoder.to(device)
            self.decoders.append(decoder)

    def forward(self):
        """Use progressive_forward."""
        raise NotImplementedError

    def progressive_forward(
        self, person: int, batch: torch.Tensor, level: int, last_level_influence: float
    ) -> torch.Tensor:
        """Process the batch in the forward pass through the encoder and corresponding decoder.

        Args:
            person: index of the person decoder.
            batch: batch of images [batchsize, channels, width, height].
            level: determines which subset of the encoder/decoder is used (0-8).
            last_level_influence: blending factor to blend between the smaller level and the current level.

        Returns:
            processed batch with reconstructed images.
        """
        decoder = self.decoders[person]
        self.latent = self.encoder.progressive_forward(
            batch, level, last_lvl_influence=last_level_influence
        )
        return decoder.progressive_forward(
            self.latent, level, last_lvl_influence=last_level_influence
        )
