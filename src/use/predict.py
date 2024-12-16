from math import ceil, sqrt
from matplotlib import pyplot
from torch.nn import Module
from tqdm import tqdm

from __param__ import PATHS, FLAGS
from src.data import ImagePreprocessor, CaptionedImageDataset
from src.eval import CaptionPostprocessor


def predict(model: Module, image_name: str) -> str:
    """ Predicts a caption for an image. """
    features = ImagePreprocessor()(image_name)
    prediction = model(features.unsqueeze(0))
    return CaptionPostprocessor()(prediction)[0]


def display(image_name: str, caption: str, true_caption: str = None, ax: pyplot.Axes = None):
    """ Displays an image with its caption. """
    image = ImagePreprocessor().image(image_name)

    if ax:
        ax.imshow(image)
        ax.set_title(caption
                     if true_caption is None else f"Prediction: {caption}\nTrue: {true_caption}", fontsize=8, pad=10)
    else:
        pyplot.imshow(image)
        pyplot.title(caption
                     if true_caption is None else f"Prediction: {caption}\nTrue: {true_caption}", fontsize=10, pad=10)
        pyplot.axis("off")
        pyplot.savefig(PATHS.OUT("prediction.png"))
        tqdm.write(f"Prediction saved to '{PATHS.OUT('prediction.png')}'.")
        pyplot.close()


def predictions(model: Module, data: CaptionedImageDataset, n: int):
    """ Outputs predictions for a number of images from the dataset using the model. """
    model.eval()

    preprocess_image = ImagePreprocessor()
    postprocess_caption = CaptionPostprocessor()

    cols = ceil(sqrt(n))
    rows = (n + cols - 1) // cols
    fig, axes = pyplot.subplots(rows, rows,
                                figsize=(15, 5 * rows))
    axes = axes.flatten()

    for i in tqdm(range(n), desc="Predicting", unit="image", disable=not FLAGS.DEBUG()):
        image_name = data.image_name(i)
        image = preprocess_image(image_name)
        caption = postprocess_caption(model(image.unsqueeze(0)))[0]
        display(image_name, caption, data.captions(i)[0], ax=axes[i])

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    file = PATHS.OUT("predictions.png")
    fig.savefig(file)
    tqdm.write(f"Predictions saved to '{file}'.")
