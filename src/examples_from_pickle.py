import pickle
import pathlib

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from nerf.load_blender import pose_spherical
import torch
from tqdm import tqdm

batch_size = 256
num_images = 500
pickle_name = "carla_improved_1600.pkl"
pickle_path = pathlib.Path(f"../data/pickles/{pickle_name}.pkl")
img_path = pathlib.Path(f"../samples/images/{pickle_name}")
interpolation_path = pathlib.Path(f"../samples/interpolation/{pickle_name}")

img_path.mkdir(parents=True, exist_ok=True)
interpolation_path.mkdir(parents=True, exist_ok=True)


def get_interpolated_images(G, num_steps=10):
    z1 = torch.randn(1, 128)
    z2 = torch.randn(1, 128)

    interpolated_vectors = torch.zeros(num_steps, 128)

    for i in tqdm(range(num_steps)):
        alpha = i / (num_steps - 1)
        interpolated_vectors[i] = (1 - alpha) * z1 + alpha * z2

    images = G(
        z=interpolated_vectors,
        c=None,
        poses=[pose_spherical(theta=30, phi=-30, radius=4.0)] * num_steps,
        scale=False,
        crop=False,
        perturb=False,
    )
    images = images.permute((0, 2, 3, 1))

    return images


def make_interpolation_examples(
    G, interpolation_path, interpolation_examples=10, num_steps=10
):
    for i in range(interpolation_examples):
        images = get_interpolated_images(G, num_steps=num_steps)
        plt.clf()
        fig, axs = plt.subplots(1, num_steps, figsize=(50, 5), tight_layout=True)

        for j, ax in enumerate(axs):
            ax.imshow(images[j])
            ax.axis("off")

        fig.savefig(interpolation_path.joinpath(f"{i}.png"))


def make_image_examples(G, num_images, batch_size, result_path):
    epochs = int(num_images / batch_size + 1)
    for i in tqdm(range(epochs)):
        z = torch.randn(batch_size, 128)
        imgs = G(z=z, c=None, scale=False, crop=False, perturb=False)

        for j, img in enumerate(imgs):
            image_name = result_path / f"{i * batch_size + j + 1}.png"
            transforms.ToPILImage()(img).save(image_name)


if __name__ == "__main__":
    with pickle_path.open("rb") as f:
        content = pickle.load(f)

    G = content["G_ema"].eval()

    print("Generating interpolation examples...")
    torch.manual_seed(0)
    make_interpolation_examples(G, interpolation_path)

    print("Generating images...")
    torch.manual_seed(0)
    make_image_examples(G, num_images, batch_size, img_path)
