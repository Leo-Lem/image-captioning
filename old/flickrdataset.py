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
                 res_dir: str,
                 out_dir: str,
                 caption_limit: int = None,
                 images_folder: str = "Images",
                 captions_file: str = "captions.csv",
                 vocabulary_threshold: int = 5,
                 image_transform: Compose = None,
                 len_captions: int = 40):
        self.image_path = os.path.join(res_dir, images_folder)
        self.captions_file = os.path.join(res_dir, captions_file)
        self.data_path = os.path.join(out_dir, "data.pkl")
        self.caption_limit = caption_limit
        self.len_captions = len_captions
        self.image_transform = image_transform or Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if os.path.exists(self.data_path):
            self._load()
        else:
            self._process(vocabulary_threshold)
            self._save()

    def _save(self):
        with open(self.data_path, 'wb') as f:
            dump({'vocabulary': self.vocabulary,
                  'images': self.images,
                  'captions': self.captions}, f)

    def _load(self):
        with open(self.data_path, 'rb') as f:
            data = load(f)
            self.vocabulary = data['vocabulary']
            self.images = data['images']
            self.captions = data['captions']

    def _process(self, vocabulary_threshold: int):
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

        self.images = images
        self.captions = captions

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
