from os import path
from pandas import read_csv, DataFrame
from PIL import Image
from torchvision.models import efficientnet_b0
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.nn.functional import adaptive_avg_pool2d
from torch import no_grad, Tensor
from tqdm import tqdm

from __param__ import PATHS, FLAGS, DATA


def reformat():
    """ Preprocess the dataset to group captions and initialize vector encoding. """
    if not DATA.REFORMAT:
        if FLAGS.DEBUG:
            print("Skipping reformatting.", end="\n\n")
        return

    data = read_csv(path.join(PATHS.RES, "flickr8k", "captions.csv"))[:400]
    if FLAGS.DEBUG:
        print(f"Loaded {len(data)} captions:", data.head(), end="\n\n")

    data = group_captions(data)
    if FLAGS.DEBUG:
        print(f"Grouped captions:", data.head(), end="\n\n")

    encode_images(data)
    if FLAGS.DEBUG:
        print(f"Encoded images:", data.head(), end="\n\n")

    train, val, test = split(data)
    if FLAGS.DEBUG:
        print(
            f"Split dataset:", train.head(), val.head(), test.head(), end="\n\n")

    save(train, val, test)
    if FLAGS.DEBUG:
        print(f"Saved dataset.", end="\n\n")

    save_samples(data)
    if FLAGS.DEBUG:
        print(f"Saved samples.", end="\n\n")

    vocab = create_vocab(data)
    if FLAGS.DEBUG:
        print(f"Created vocabulary.", list(vocab)[:10], end="\n\n")


def group_captions(captions: DataFrame) -> DataFrame:
    """ Group captions by image. """
    return captions\
        .groupby('image')['caption']\
        .apply(lambda x: set(x))\
        .reset_index()\
        .set_index('image')\
        .rename(columns={'caption': 'captions'})


def encode_images(grouped_captions: DataFrame):
    """ Encode images using EfficientNet B0. """
    encoder = efficientnet_b0(weights="DEFAULT").eval()
    if FLAGS.GPU:
        encoder.cuda()

    for img_name in tqdm(grouped_captions.index, desc="Encoding images", unit="image", disable=not FLAGS.DEBUG):
        image = Image\
            .open(path.join(PATHS.RES, "flickr8k", "Images", img_name))\
            .convert("RGB")
        tensor: Tensor = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(image)
        if FLAGS.GPU:
            tensor = tensor.cuda()

        with no_grad():
            features = encoder.features(tensor.unsqueeze(0))
            features = adaptive_avg_pool2d(features, (1, 1))
            vector = features.view(features.size(0), -1)

        grouped_captions.at[img_name, 'vector'] = str(vector.cpu().tolist())


def split(data: DataFrame, train: float = 0.6, val: float = 0.2, test: float = 0.2) -> tuple[DataFrame, DataFrame, DataFrame]:
    """ Split the dataset into train, validation, and test sets. """
    data = data.sample(frac=1)
    train_end = int(train * len(data))
    val_end = train_end + int(val * len(data))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def save(train: DataFrame, val: DataFrame, test: DataFrame):
    """ Save the preprocessed dataset. """
    train.to_csv(path.join(PATHS.RES, "train.csv"), index=True)
    val.to_csv(path.join(PATHS.RES, "val.csv"), index=True)
    test.to_csv(path.join(PATHS.RES, "test.csv"), index=True)


def save_samples(data: DataFrame):
    """ Save samples of the preprocessed dataset. """
    data.sample(10).to_csv(path.join(PATHS.RES, "sample.csv"), index=True)


def create_vocab(data: DataFrame) -> dict[str, int]:
    """ Create a vocabulary from the captions. """
    vocab = set()
    for captions in data['captions']:
        for caption in captions:
            vocab.update(caption.lower().split())
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        **{word: i + 2 for i, word in enumerate(vocab)}
    }
    vocab_df = DataFrame(vocab.items(), columns=["word", "index"])\
        .set_index("word")
    vocab_df.to_csv(path.join(PATHS.RES, "vocab.csv"), index=True)
    return vocab
