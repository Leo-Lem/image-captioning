from torch import Tensor, vstack, cat
from torch.utils.data import DataLoader, Dataset
from typing import Literal
from pandas import DataFrame, read_csv

from .preprocess import CaptionPreprocessor, ImagePreprocessor
from __param__ import DATA, PATHS, TRAIN


class CaptionedImageDataset(Dataset):
    """ Dataset class for image captioning. """

    def __init__(self, name: Literal["full", "train", "val", "test"]):
        self.data = self._load_data(name)
        self.preprocess_caption = CaptionPreprocessor()
        self.preprocess_image = ImagePreprocessor()

    def loader(self, shuffle: bool = False, batch: bool = True) -> DataLoader:
        """ Create a DataLoader from the dataset. """
        def collate_fn(batch: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
            images, captions = map(lambda x: cat(x, dim=0), zip(*batch))

            assert captions.size() == (len(batch) * DATA.NUM_CAPTIONS, DATA.CAPTION_LEN)
            assert images.size() == (len(batch) * DATA.NUM_CAPTIONS, DATA.FEATURE_DIM)

            return images.to(TRAIN.DEVICE), captions.to(TRAIN.DEVICE)

        loader = DataLoader(self,
                            batch_size=TRAIN.BATCH_SIZE if batch else 1,
                            collate_fn=collate_fn,
                            shuffle=shuffle,
                            num_workers=4,
                            pin_memory=TRAIN.DEVICE == "cuda")

        return loader

    def _load_data(self, name: Literal["full", "train", "val", "test"]) -> DataFrame:
        """ Load the specified dataset. """
        name = 'sample' if DATA.SAMPLE else name
        data = read_csv(PATHS.OUT(f"data-{name}.csv"), dtype=str)
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """" Get the repeated image and the corresponding captions as tensors. """
        image = self.preprocess_image(self.image_name(idx))
        captions = vstack([self.preprocess_caption(caption)
                           for caption in self.captions(idx)])
        image = image.squeeze(0).repeat(DATA.NUM_CAPTIONS, 1)

        assert captions.size() == (DATA.NUM_CAPTIONS, DATA.CAPTION_LEN)
        assert image.size() == (DATA.NUM_CAPTIONS, DATA.FEATURE_DIM)

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
