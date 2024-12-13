import os
from matplotlib.pyplot import savefig
from torch import cuda, device, save, load
from torch.utils.data import random_split, DataLoader
from pandas import DataFrame, read_csv
from tqdm import trange

from old import FlickrDataset
from models import ImageCaption, ResnetEncoder, VGGEncoder, GRUDecoder, LSTMDecoder
from eval import train, evaluate


# --- Config ---
RES_DIR = "res"
OUT_DIR = ".out"
CAPTION_LIMIT = None
EPOCHS = 10

IS_GPU = cuda.is_available()
DEVICE = device("cuda" if IS_GPU else "cpu")
BATCH_SIZE = 800 if IS_GPU else 200
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
results_path = os.path.join(OUT_DIR, "results.csv")
results = read_csv(results_path, index_col=0) \
    if os.path.exists(results_path) else DataFrame(columns=models.keys())


def store_results():
    results.to_csv(results_path)
    results.plot(xlabel="Epoch",
                 ylabel="BLEU Score",
                 xticks=range(1, EPOCHS+1))
    savefig(f"{OUT_DIR}/results.png")


for name, (encoder, decoder) in models.items():
    if name in results.columns and results[name].notna().all():
        print(f"Skipping {name}, already trained and evaluated.")
        continue

    model = ImageCaption(encoder, decoder).to(DEVICE)
    for epoch in trange(1, EPOCHS+1, unit="epoch", desc=name):
        if epoch in results.index and results.at[epoch, name] > 0:
            print(f"Loaded results in epoch {epoch} for {name}, skipping.")
            continue

        state_file = os.path.join(OUT_DIR, f"{name}-{epoch}.pth")
        if (os.path.exists(state_file)):
            model.load_state_dict(load(state_file, weights_only=False))
            print(f"Loaded {name} in epoch {epoch}, skipping training.")
        else:
            train(model, train_loader, device=DEVICE)
            save(model.state_dict(), state_file)

        results.at[epoch, name] = evaluate(
            model, eval_loader, dataset, device=DEVICE)
        store_results()
