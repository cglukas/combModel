"""Model"""
import torch
from torch import nn
from torch.nn.modules.module import T

alpha = 0.2


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.levels = []
        self.rgb_levels = []
        self.add_level(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2),
        )  # 8
        self.add_level(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2),
        )  # 7
        self.add_level(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(32, 128, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2),
        )  # 6
        self.add_level(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2),
        )  # 5
        self.add_level(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2),
        )  # 4
        self.add_level(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2),
        )  # 3
        self.add_level(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2),
        )  # 2
        self.add_level(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.MaxPool2d(2),
        )  # 1
        self.add_level(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 512, 4),
            nn.LeakyReLU(alpha),
        )  # 0
        self.levels = nn.ParameterList(self.levels)
        self.rgb_levels = nn.ParameterList(self.rgb_levels)

    def add_level(self, *modules):
        self.rgb_levels.insert(
            0,
            nn.Sequential(nn.Conv2d(3, modules[0].in_channels, 1), nn.LeakyReLU(alpha)),
        )
        modules = nn.Sequential(*modules)
        self.levels.insert(0, modules)

    def from_rgb(self, tensor, level):
        rgb_conversion = self.rgb_levels[level]
        return rgb_conversion(tensor)

    def forward(self, tensor, level):
        tensor = self.from_rgb(tensor, level)
        modules = self.levels[: level + 1]
        modules = reversed(modules)
        for module in modules:
            tensor = module(tensor)
        return tensor

    def progressive_forward(self, tensor, level, last_lvl_influence: float = 0.0):
        if level == 0:
            return self.forward(tensor, level)
        small_tensor = nn.functional.interpolate(tensor, scale_factor=0.5)
        small_tensor = self.from_rgb(small_tensor, level-1)
        tensor = self.from_rgb(tensor, level)
        tensor = self.levels[level](tensor)
        tensor = last_lvl_influence * tensor + (1-last_lvl_influence) * small_tensor

        modules = self.levels[: level]
        modules = reversed(modules)
        for module in modules:
            tensor = module(tensor)
        return tensor


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.levels = []
        self.rgb_levels = []
        self.add_level(
            nn.Upsample(scale_factor=2  ),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(alpha),
        )  # 8
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(alpha),
        )  # 7
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(alpha),
        )  # 6
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(alpha),
        )  # 5
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.LeakyReLU(alpha),
        )  # 4
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
        )  # 3
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
        )  # 2
        self.add_level(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
        )  # 1
        self.add_level(
            nn.Conv2d(512, 512, 4, padding=3),
            nn.LeakyReLU(alpha),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.LeakyReLU(alpha),
        )  # 0
        self.levels = nn.ParameterList(self.levels)  # Fix missing parameters
        self.rgb_levels = nn.ParameterList(self.rgb_levels)  # Fix missing parameters

    def add_level(self, *modules):
        self.rgb_levels.insert(
            0,
            nn.Sequential(
                torch.nn.Conv2d(modules[-2].out_channels, 3, 1), nn.Sigmoid()
            ),
        )
        modules = nn.Sequential(*modules)
        self.levels.insert(0, modules)

    def from_rgb(self, tensor, level):
        rgb_conversion = self.rgb_levels[level]
        return rgb_conversion(tensor)

    def forward(self, latent, level):
        modules = self.levels[: level + 1]
        for module in modules:
            latent = module(latent)
        image = self.from_rgb(latent, level)
        return image

    def progressive_forward(self, latent, level, last_lvl_influence: float = 0.0):
        if level == 0:
            return self.forward(latent, level)

        small_img = self.forward(latent, level - 1)
        img = self.forward(latent, level)
        small_img = nn.functional.interpolate(small_img, scale_factor=2)
        img = last_lvl_influence * img + (1 - last_lvl_influence) * small_img

        return img


class CombModel(nn.Module):
    def __init__(self, persons: int = 2, device='cpu'):
        super(CombModel, self).__init__()
        self.encoder = Encoder()
        self.encoder.to(device)
        self.decoders = []
        for person in range(persons):
            decoder = Decoder()
            decoder.to(device)
            self.decoders.append(decoder)

    def progressive_forward(
        self, person: int, tensor: torch.Tensor, level: int, last_level_influence: float
    ):
        latent = self.encoder.progressive_forward(
            tensor, level, last_lvl_influence=last_level_influence
        )
        decoder = self.decoders[person]
        return decoder.progressive_forward(
            latent, level, last_lvl_influence=last_level_influence
        )

    def eval(self: T) -> T:
        self.encoder.eval()
        for decoder in self.decoders:
            decoder.eval()
        return super().eval()

    def train(self: T, mode: bool = True) -> T:
        self.encoder.train(mode)
        for decoder in self.decoders:
            decoder.train(mode)
        return super().train(mode)


if __name__ == "__main__":
    enc = Encoder()
    dec = Decoder()
    comb = CombModel()
    for level, size in enumerate([4, 8, 16, 32, 64, 128, 256, 512, 1024]):
        rand = torch.rand((1, 3, size, size))

        latent = enc.forward(rand, level=level)
        assert (1, 512, 1, 1) == latent.shape
        reconstruct = dec.forward(latent, level=level)
        assert rand.shape == reconstruct.shape

        reconstruct_comb = comb.progressive_forward(0, rand, level, 0.5)
        assert rand.shape == reconstruct_comb.shape

