"""Visualize latent space."""
from pathlib import Path

import torch.cuda
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from code.data_io.dataloader import SCALES, SizeLoader
from code.model.comb_model import CombModel


def main():
    level = 2
    filepath = r"C:\Users\Lukas\PycharmProjects\combModel\code\trainer\230210_213807\model2_1.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_persons = 2
    loarder_a = SizeLoader(
        Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
        max_persons=max_persons,
        person=0,
    )
    loarder_b = SizeLoader(
        Path(r"C:\Users\Lukas\PycharmProjects\combModel\data\preprocessed"),
        max_persons=max_persons,
        person=1,
    )
    SizeLoader.scale = SCALES[level]
    model = CombModel(device=device, persons=2)
    model.load_state_dict(torch.load(filepath))
    model.to(device)
    latents = [torch.zeros(1,512, 1, 1)]*2
    for a, b in zip(loarder_a, loarder_b):
        a, b = a[None, :].to(device), b[None, :].to(device)

        for person in range(max_persons):
            person_image = a if person else b
            model.progressive_forward(person, person_image, level, 1)
            latents[person] = torch.cat((latents[person], model.latent.detach().cpu()), 0)
        break

    tsne = TSNE(2, verbose=4)
    all_latents = torch.cat((latents[0], latents[1]), 0)
    print(latents[0].shape)
    squeeze = all_latents.squeeze()
    print(squeeze.shape)
    tsne_proj = tsne.fit_transform(squeeze)
    fig, ax = plt.subplot(nrow=1)


if __name__ == "__main__":
    main()
