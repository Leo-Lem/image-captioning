from pandas import DataFrame
from PIL import Image
from torch import tensor, stack, Tensor, no_grad
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from __param__ import DEBUG, PATHS, FLAGS, DATA, VOCAB, TRAIN


def preprocess(data: DataFrame, vocab: dict[str, int], shuffle: bool = False, batch: bool = True) -> DataLoader:
    """ Preprocess the specified dataset. """
    dataset = CaptionedImagesDataset(data, vocab)
    loader = dataloader(dataset, shuffle, batch)
    return loader


def dataloader(dataset: Dataset, shuffle: bool, batch: bool) -> DataLoader:
    """ Create a DataLoader from the given dataset. """
    def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        images, captions = zip(*batch)
        images, captions = stack(images), stack(captions)
        # DEBUG(f"Collated: {images.shape} | {captions.shape}")
        return images, captions

    loader = DataLoader(dataset,
                        batch_size=TRAIN.BATCH_SIZE if batch else 1,
                        collate_fn=collate_fn,
                        shuffle=shuffle)
    DEBUG(f"Created DataLoader ({len(loader)} batches)")
    return loader


class CaptionedImagesDataset(Dataset):
    """ Dataset class for image captioning. """

    def __init__(self, data: DataFrame, vocab: dict[str, int]):
        self.data = data
        self.preprocessor = Preprocessor(vocab)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """" Get an image and the corresponding captions as tensors. """
        image = self.preprocessor.preprocess(self.image_name(idx))
        captions = stack([self.preprocessor.caption_tensor(caption)
                         for caption in self.captions(idx)])
        return image, captions

    def image_name(self, idx: int) -> str:
        """ Get the image name at the specified index. """
        image = self.data.iloc[idx]["image"]
        return image

    def captions(self, idx: int) -> list[str]:
        """ Get the captions at the specified index. """
        captions = [str(self.data.iloc[idx][f"caption_{i}"])
                    for i in range(1, 6)]
        return captions


class Preprocessor:
    """ Preprocesses images and captions. """

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.encoder = efficientnet_b0(weights="DEFAULT").eval()
        if FLAGS.GPU:
            self.encoder.cuda()

    def preprocess(self, image: str) -> Tensor:
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

    def caption_tensor(self, caption: str) -> Tensor:
        """ Get the padded tensor representation of the caption. """
        caption = [VOCAB.START] + [self.vocab.get(word, VOCAB.UNKNOWN)
                                   for word in caption.split()]
        if len(caption) > DATA.CAPTION_LEN-1:
            padded = caption[:DATA.CAPTION_LEN-1] + [VOCAB.END]
        else:
            padded = caption + [VOCAB.END] + [VOCAB.PADDING] * \
                (DATA.CAPTION_LEN-1 - len(caption))
        return tensor(padded)
