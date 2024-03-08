"""Module of the encoder of the neuronal network."""
from torch import nn

from development.model.constants import RELU_ALPHA


class Encoder(nn.Module):
    """Encoder part of the encoder-decoder model.

    This model encodes input images to the latent space.
    """

    def __init__(self):
        super().__init__()
        self.levels = []
        self.rgb_levels = []
        self.add_level(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.MaxPool2d(2),
        )  # 8
        self.add_level(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.MaxPool2d(2),
        )  # 7
        self.add_level(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(32, 128, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.MaxPool2d(2),
        )  # 6
        self.add_level(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.MaxPool2d(2),
        )  # 5
        self.add_level(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.MaxPool2d(2),
        )  # 4
        self.add_level(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.MaxPool2d(2),
        )  # 3
        self.add_level(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.MaxPool2d(2),
        )  # 2
        self.add_level(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.MaxPool2d(2),
        )  # 1
        self.add_level(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(RELU_ALPHA),
            nn.Conv2d(512, 512, 4),
            nn.LeakyReLU(RELU_ALPHA),
        )  # 0
        self.levels = nn.ParameterList(self.levels)
        self.rgb_levels = nn.ParameterList(self.rgb_levels)

    def add_level(self, *modules):
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
                nn.Conv2d(3, modules[0].in_channels, 1), nn.LeakyReLU(RELU_ALPHA)
            ),
        )
        modules = nn.Sequential(*modules)
        self.levels.insert(0, modules)

    def from_rgb(self, tensor, level):
        """Process the batch transforming the rgb image to the representation of the level.

        Args:
            tensor: batch of images.
            level: index of the level preprocessor.

        Returns:
            Preprocessed batch that fits the shape of the layers of the level.
        """
        rgb_conversion = self.rgb_levels[level]
        return rgb_conversion(tensor)

    def forward(self, tensor, level):
        """Encode the batch of images from the level.

        Args:
            tensor: batch of images.
            level: determines the entry layer of the network.
                   Image size and level need to fit together (see IMAGE_SIZE enum)

        Returns:
            Latent vectors of the batch.
        """
        tensor = self.from_rgb(tensor, level)
        modules = self.levels[: level + 1]
        modules = reversed(modules)
        for module in modules:
            tensor = module(tensor)
        return tensor

    def progressive_forward(self, tensor, level, last_lvl_influence: float = 0.0):
        """Encode the batch to a batch of latent vectors.

        Args:
            tensor: batch of input images.
            level: level of the network that should be the entry point. Needs to
                   Correspond to the image size. See IMAGE_SIZE enum.
            last_lvl_influence: blend between the last level and the current level.
                                Helps to transition the levels while training.

        Returns:
            Batch of latent vetors.
        """
        if level == 0:
            return self.forward(tensor, level)
        small_tensor = nn.functional.interpolate(tensor, scale_factor=0.5)
        small_tensor = self.from_rgb(small_tensor, level - 1)
        tensor = self.from_rgb(tensor, level)
        tensor = self.levels[level](tensor)
        tensor = last_lvl_influence * tensor + (1 - last_lvl_influence) * small_tensor

        modules = self.levels[:level]
        modules = reversed(modules)
        for module in modules:
            tensor = module(tensor)
        return tensor
