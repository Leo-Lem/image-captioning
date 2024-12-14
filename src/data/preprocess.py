from logging import debug as DEBUG
from os import path
from pandas import DataFrame
from PIL import Image
from torch import tensor, stack, Tensor, no_grad
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from __param__ import PATHS, TRAIN, FLAGS, DATA


def preprocess(data: DataFrame, vocab: dict[str, int]) -> DataLoader:
    """ Preprocess the specified dataset. """
    dataset = CustomDataset(data, vocab)
    loader = dataloader(dataset)
    return loader


def dataloader(dataset: Dataset) -> DataLoader:
    """ Create a DataLoader from the given dataset. """
    def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        images, captions = zip(*batch)
        images, captions = stack(images), stack(captions)
        DEBUG(f"Collated: {images.shape} | {captions.shape}")
        return images, captions

    loader = DataLoader(dataset,
                        batch_size=TRAIN.BATCH_SIZE,
                        shuffle=True,
                        collate_fn=collate_fn)
    DEBUG(f"Created DataLoader ({len(loader)} batches)")
    return loader


class CustomDataset(Dataset):
    """ Dataset class for image captioning. """

    def __init__(self, data: DataFrame, vocab: dict[str, int]) -> None:
        self.data = data
        self.vocab = vocab
        self.encoder = efficientnet_b0(weights="DEFAULT").eval()
        if FLAGS.GPU:
            self.encoder.cuda()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """" Get an image and the corresponding caption as tensors. """
        image = self.image(self.image_name(idx))
        image_tensor = self.image_tensor(image)
        image = self.image_features(image_tensor)
        caption = self.caption_tensor(self.captions(idx)[0])

        return image, caption

    def row(self, idx: int) -> DataFrame:
        """ Get the row at the specified index. """
        row = self.data.iloc[idx]
        return row

    def image_name(self, idx: int) -> str:
        """ Get the image name at the specified index. """
        image = self.row(idx)["image"]
        return image

    def captions(self, idx: int) -> list[str]:
        """ Get the captions at the specified index. """
        captions = [str(self.row(idx)[f"caption_{i}"]) for i in range(1, 6)]
        return captions

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
        return Image.open(path.join(PATHS.RES, "flickr8k", "Images", name)).convert("RGB")

    def caption_tensor(self, caption: str) -> Tensor:
        """ Get the padded tensor representation of the caption. """
        caption = [DATA.START] + [self.vocab.get(word, DATA.UNKNOWN)
                                  for word in caption.split()]
        if len(caption) > DATA.CAPTION_LEN-1:
            padded = caption[:DATA.CAPTION_LEN-1] + [DATA.END]
        else:
            padded = caption + [DATA.END] + [DATA.PADDING] * \
                (DATA.CAPTION_LEN-1 - len(caption))
        return tensor(padded)
