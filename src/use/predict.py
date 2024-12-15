from math import ceil, sqrt
from matplotlib import pyplot
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from __param__ import PATHS, FLAGS
from src.eval import extract_from_embedding, extract_from_tokenized
from src.data import load_vocab, Preprocessor


def predict(model: Module, image_name: str) -> str:
    """ Predicts a caption for an image. """
    vocab = load_vocab()
    features = Preprocessor(vocab).preprocess(image_name)
    prediction = model(features.unsqueeze(0))
    return extract_from_embedding(prediction,  {index: word for word, index in vocab.items()})[0]


def display(image_name: str, caption: str, true_caption: str = None, ax: pyplot.Axes = None):
    """ Displays an image with its caption. """
    image = Preprocessor(load_vocab()).image(image_name)
    pyplot.axis("off")
    pyplot.imshow(image)
    pyplot.title(caption
                 if true_caption is None else f"Prediction: {caption}\nTrue: {true_caption}", fontsize=10, pad=10)
    pyplot.savefig(PATHS.OUT("prediction.png"))
    tqdm.write(f"Prediction saved to '{PATHS.OUT('prediction.png')}'.")
    pyplot.close()


def predictions(model: Module, data: DataLoader, reversed_vocab: dict[int, str], n: int):
    """ Outputs predictions for a number of images from the dataset using the model. """
    model.eval()

    grid_cols = ceil(sqrt(n))
    grid_rows = (n + grid_cols - 1) // grid_cols
    _, axes = pyplot.subplots(grid_rows, grid_cols,
                              figsize=(15, 5 * grid_rows))
    axes = axes.flatten()
    preprocessor = Preprocessor(load_vocab())

    for i, (image_batch, captions_batch) in enumerate(tqdm(data, desc="Predicting", unit="image", total=n, disable=not FLAGS.DEBUG())):
        if i >= n:
            break
        image = preprocessor.image(data.dataset.image_name(i))
        caption = extract_from_tokenized(captions_batch, reversed_vocab)[0][0]
        true_caption = extract_from_embedding(
            model(image_batch), reversed_vocab)[0]
        axes[i].imshow(image)
        axes[i].set_title(caption
                          if true_caption is None else f"Prediction: {caption}\nTrue: {true_caption}", fontsize=10, pad=10)

    for ax in axes:
        ax.axis("off")
    pyplot.tight_layout()
    pyplot.savefig(PATHS.OUT("predictions.png"))
    tqdm.write(
        f"Predictions saved to '{PATHS.OUT('predictions.png')}'.")
    pyplot.close()
