from PIL import Image
from torch import tensor, Tensor, no_grad
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import efficientnet_b0
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from __param__ import PATHS, FLAGS, DATA
from .vocab import Vocabulary


class CaptionPreprocessor:
    def __init__(self):
        self.vocab = Vocabulary()

    def __call__(self, caption: str) -> Tensor:
        """ Preprocess the caption and return the indices. """
        indices = self.caption_indices(caption)
        return indices

    def caption_indices(self, caption: str) -> Tensor:
        """ Get the padded tensor representation of the caption. """
        caption = [self.vocab.START] + [self.vocab[word]
                                        for word in caption.split()]
        if len(caption) > DATA.CAPTION_LEN-1:
            padded = caption[:DATA.CAPTION_LEN-1] + [self.vocab.END]
        else:
            padded = caption + [self.vocab.END] + [self.vocab.PADDING] * \
                (DATA.CAPTION_LEN-1 - len(caption))
        return tensor(padded)


class ImagePreprocessor:
    def __init__(self):
        self.encoder = efficientnet_b0(weights="DEFAULT").eval()
        if FLAGS.GPU:
            self.encoder.cuda()

    def __call__(self, image: str) -> Tensor:
        """ Preprocess the image and return the features. """
        image = self.image(image)
        tensor = self.image_tensor(image)
        features = self.image_features(tensor)
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
        tensor: Tensor = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(image)
        if FLAGS.GPU:
            return tensor.cuda()
        return tensor

    def image(self, name: str) -> Image:
        """ Get the image with the specified name. """
        return Image.open(PATHS.RESOURCES("Images", name)).convert("RGB")
