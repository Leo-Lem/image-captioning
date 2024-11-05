from PIL.Image import Image
from matplotlib import pyplot as plt
from matplotlib import rcParams


def visualise(image: Image, caption: str):
    """
    Display an image and its caption.

    Args:
        image (Image): The image to display.
        caption (str): The caption to display.
    """
    assert image.mode == "RGB", "Image should be in RGB format."

    plt.imshow(image)
    plt.title(caption, wrap=True)
    plt.axis("off")
    plt.show()
