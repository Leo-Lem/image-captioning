from matplotlib import pyplot as plt
from torch import cuda, device
from torch.utils.data import random_split, DataLoader
from pandas import DataFrame
from tqdm import trange

from data import FlickrDataset
from models import ImageCaption, ResnetEncoder, VGGEncoder, GRUDecoder, LSTMDecoder
from eval import train, evaluate, load, save

RES_DIR = "res"
OUT_DIR = ".out"
CAPTION_LIMIT = 10
EPOCHS = 5

IS_GPU = cuda.is_available()
DEVICE = device("cuda" if IS_GPU else "cpu")
BATCH_SIZE = 800 if IS_GPU else 400
NUM_WORKERS = 2 if IS_GPU else 0

# --- Data ---
dataset = FlickrDataset(res_dir=RES_DIR,
                        out_dir=OUT_DIR,
                        caption_limit=CAPTION_LIMIT)
train_dataset, eval_dataset = random_split(dataset, [.8, .2])
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS,
                          pin_memory=IS_GPU)
eval_loader = DataLoader(eval_dataset,
                         batch_size=BATCH_SIZE,
                         num_workers=NUM_WORKERS,
                         pin_memory=IS_GPU)


# --- Models ---
models = {
    "resnet-gru": (ResnetEncoder(), GRUDecoder(vocabulary_size=len(dataset.vocabulary), feature_size=2048)),
    "vgg-gru": (VGGEncoder(), GRUDecoder(vocabulary_size=len(dataset.vocabulary), feature_size=512)),
    "resnet-lstm": (ResnetEncoder(), LSTMDecoder(vocabulary_size=len(dataset.vocabulary), feature_size=2048)),
    "vgg-lstm": (VGGEncoder(), LSTMDecoder(vocabulary_size=len(dataset.vocabulary), feature_size=512)),
}


# --- Training & Evaluation ---
results = DataFrame(columns=models.keys())
for epoch in trange(EPOCHS):
    for name, (encoder, decoder) in models.items():
        print(f"Training {name}")
        model = ImageCaption(encoder, decoder).to(DEVICE)
        load(model, epoch, OUT_DIR)
        train(model, train_loader, epochs=EPOCHS, device=DEVICE)
        save(model, epoch, OUT_DIR)
        results.loc[epoch, name] = evaluate(
            model, eval_loader, dataset, device=DEVICE)
results.to_csv(f"{OUT_DIR}/results.csv")

# --- Results ---
results.plot()
plt.savefig(f"{OUT_DIR}/results.png")
