from math import ceil, sqrt
from matplotlib import pyplot
from PIL.Image import Image
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from __param__ import PATHS
from src.eval import extract_from_embedding, extract_from_tokenized


def predict(model: Module, image: str) -> str:
    """ Predicts a caption for an image. """
    pass


def predictions(model: Module, data: DataLoader, reversed_vocab: dict[int, str], n: int):
    """ Outputs predictions for a number of images from the dataset using the model. """
    model.eval()

    grid_cols = ceil(sqrt(n))
    grid_rows = (n + grid_cols - 1) // grid_cols
    _, axes = pyplot.subplots(grid_rows, grid_cols,
                              figsize=(15, 5 * grid_rows))
    axes = axes.flatten()

    for i, (image_batch, captions_batch) in enumerate(tqdm(data, desc="Predicting", unit="image", total=n)):
        if i >= n:
            break
        image = data.dataset.image(data.dataset.image_name(i))
        caption = captions_batch
        prediction = model(image_batch)
        display_captioned_image_grid(
            image, caption, prediction, reversed_vocab, axes[i])

    for ax in axes:
        ax.axis("off")
    pyplot.tight_layout()
    pyplot.savefig(PATHS.OUT("predictions.png"))
    pyplot.close()


def display_captioned_image_grid(image: Image, caption: Tensor, prediction: Tensor, reversed_vocab: dict[int, str], ax):
    ax.set_title(
        f"Caption: {extract_from_tokenized(caption, reversed_vocab)[0][0]}\nPrediction: {extract_from_embedding(prediction, reversed_vocab)[0]}", fontsize=10, pad=10)
    ax.imshow(image)
