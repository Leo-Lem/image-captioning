from torch import cuda, device
from torch.utils.data import random_split, DataLoader

from data import FlickrDataset
from models import ImageCaption, ResnetEncoder, VGGEncoder, GRUDecoder, LSTMDecoder
from eval import train, evaluate

DIR = "out"  # "/content/drive/MyDrive/flickr8k"
CAPTION_LIMIT = None
EPOCHS = 5
DEVICE = device("cuda" if cuda.is_available() else "cpu")
BATCH_SIZE = 800 if cuda.is_available() else 400
NUM_WORKERS = 2 if cuda.is_available() else 0
PIN_MEMORY = cuda.is_available()

# --- Data ---
dataset = FlickrDataset(path=DIR, caption_limit=CAPTION_LIMIT)
train_dataset, eval_dataset = random_split(dataset, [.8, .2])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


# --- Model ---
models = {
    "resnet-gru": (ResnetEncoder(), GRUDecoder(vocabulary_size=len(dataset.vocabulary), feature_size=2048)),
    "vgg-gru": (VGGEncoder(), GRUDecoder(vocabulary_size=len(dataset.vocabulary), feature_size=512)),
    "resnet-lstm": (ResnetEncoder(), LSTMDecoder(vocabulary_size=len(dataset.vocabulary), feature_size=2048)),
    "vgg-lstm": (VGGEncoder(), LSTMDecoder(vocabulary_size=len(dataset.vocabulary), feature_size=512)),
}


def train_and_evaluate(encoder, decoder) -> tuple[str, float]:
    model = ImageCaption(encoder, decoder, checkpoint_dir=DIR).to(DEVICE)
    train(model, train_loader, epochs=EPOCHS, device=DEVICE)
    return evaluate(model, eval_loader, dataset, device=DEVICE)


with open(f"{DIR}/results.txt", "a") as f:
    f.write("\nResults\n")
for name, (encoder, decoder) in models.items():
    print(f"Training {name}")
    result = train_and_evaluate(encoder, decoder)
    print(f"BLEU: {result}")
    with open(f"{DIR}/results.txt", "a") as f:
        f.write(f"{name}: {result}\n")
