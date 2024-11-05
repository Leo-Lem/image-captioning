import os
from torch import tensor, Tensor, zeros, float32
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from pickle import dump, load
from tqdm import tqdm
from pandas import read_csv


from .vocabulary import Vocabulary


class FlickrDataset(Dataset):
    def __init__(self,
                 path: str,
                 caption_limit: int = None,
                 images_folder: str = "Images",
                 captions_file: str = "captions.csv",
                 data_file: str = "preprocessed.pkl",
                 image_transform: Compose = None,
                 vocabulary_threshold: int = 5,
                 len_captions: int = 40):
        self.image_path = os.path.join(path, images_folder)
        self.captions_file = os.path.join(path, captions_file)
        self.data_path = os.path.join(path, data_file)
        self.caption_limit = caption_limit
        self.len_captions = len_captions
        self.image_transform = image_transform or Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if os.path.exists(self.data_path) and caption_limit is None:
            print("Loading preprocessed data...")
            self.vocabulary, self.images, self.captions = self._load()
        else:
            self.vocabulary, self.images, self.captions = self._process(
                vocabulary_threshold)
            print("Saving preprocessed data...")
            self._save(self.vocabulary, self.images, self.captions)

    def _save(self, vocabulary: Vocabulary, images: Tensor, captions: Tensor):
        with open(self.data_path, 'wb') as f:
            dump({'images': images,
                  'captions': captions,
                  'vocabulary': vocabulary}, f)

    def _load(self) -> tuple[Vocabulary, Tensor, Tensor]:
        with open(self.data_path, 'rb') as f:
            data = load(f)
            return (data['vocabulary'],
                    data['images'],
                    data['captions'])

    def _process(self, vocabulary_threshold: int) -> tuple[Vocabulary, Tensor, Tensor]:
        data = read_csv(self.captions_file, nrows=self.caption_limit)
        self.vocabulary = Vocabulary(
            data['caption'].tolist(), threshold=vocabulary_threshold)

        raw_images = data['image'].unique().tolist()
        image_to_captions = data.groupby(
            'image')['caption'].apply(list).to_dict()

        images: Tensor = zeros([len(raw_images),
                                3, 224, 224], dtype=float32)
        captions: Tensor = zeros([len(raw_images),
                                  len(data) // len(raw_images),
                                  self.len_captions+2], dtype=int)

        for img_index, raw_image in enumerate(tqdm(raw_images, desc="Preprocessing", unit="images")):
            images[img_index] = self.image_to_tensor(raw_image)
            for cap_index, raw_caption in enumerate(image_to_captions[raw_image]):
                captions[img_index, cap_index] = \
                    self.caption_to_tensor(raw_caption)

        return self.vocabulary, images, captions

    def __len__(self):
        return self.captions.size(0) * self.captions.size(1)

    def __getitem__(self, index: int) -> Tensor:
        return (self.images[index//self.captions.size(1)],
                self.captions[index//self.captions.size(1)][index % self.captions.size(1)])

    def tensor_to_caption(self, tensor: Tensor) -> str:
        return self.vocabulary.denumericalize(tensor.tolist()[1:-1])

    def caption_to_tensor(self, caption: str) -> Tensor:
        caption_indices = [Vocabulary.sos_index] + \
            self.vocabulary.numericalize(caption) + \
            [Vocabulary.eos_index]

        caption_indices += [Vocabulary.pad_index] * \
            (self.len_captions+2 - len(caption_indices))

        return tensor(caption_indices)

    def image_to_tensor(self, image: str) -> Tensor:
        return self.image_transform(Image.open(os.path.join(self.image_path, image)).convert('RGB'))
