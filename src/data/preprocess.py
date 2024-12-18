from PIL import Image
from torch import tensor, Tensor, no_grad
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import efficientnet_b0
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from __param__ import PATHS, FLAGS, DATA, TRAIN
from .vocab import Vocabulary


class CaptionPreprocessor:
    def __init__(self):
        self.vocab = Vocabulary()

    def __call__(self, caption: str) -> Tensor:
        """ Preprocess the caption and return the index tensor. """
        indices = self.index(caption)
        padded = self.pad_or_truncate(indices)
        padded_tensor = tensor(padded)

        assert padded_tensor.shape == (DATA.CAPTION_LEN,)

        return padded_tensor

    def pad_or_truncate(self, indices: list[int]) -> list[int]:
        """ Pad the indices with the PADDING token or truncate if necessary. """
        if len(indices) < DATA.CAPTION_LEN-1:
            padded = indices + [self.vocab.END] + \
                [self.vocab.PADDING] * (DATA.CAPTION_LEN-1 - len(indices))
        else:
            padded = indices[:DATA.CAPTION_LEN-1] + [self.vocab.END]
        return padded

    def index(self, caption: str) -> list[int]:
        """ Get the indices of the words in the caption. """
        indices = [self.vocab.START] + [self.vocab[word]
                                        for word in caption.split()]
        return indices


class ImagePreprocessor:
    def __init__(self):
        self.encoder = efficientnet_b0(weights="DEFAULT")\
            .eval()\
            .to(TRAIN.DEVICE)

    def __call__(self, image: str) -> Tensor:
        """ Preprocess the image and return the features. """
        image = self.image(image)
        tensor = self.image_tensor(image)
        features = self.image_features(tensor)

        assert features.shape == (1, DATA.FEATURE_DIM)

        return features

    def image_features(self, tensor: Tensor) -> Tensor:
        """ Encode the image using EfficientNet B0. """
        with no_grad():
            features = self.encoder.features(tensor.unsqueeze(0))
            features = adaptive_avg_pool2d(features, (1, 1))
            vector = features.view(features.size(0), -1)
        return vector

    def image_tensor(self, image: Image) -> Tensor:
        """ Get the tensor representation of the image. """
        encoded: Tensor = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(image)
        return encoded.to(TRAIN.DEVICE)

    def image(self, name: str) -> Image:
        """ Get the image with the specified name. """
        return Image.open(PATHS.RESOURCES("Images", name)).convert("RGB")
