from logging import debug as DEBUG
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import trange

from .epoch import train_epoch
from .val import validate
from __param__ import FLAGS, TRAIN, DATA


def train(model: Module, train: DataLoader, val: DataLoader):
    """ Train the decoder model. """

    optimizer = Adam(model.parameters(), lr=TRAIN.LEARNING_RATE)
    criterion = CrossEntropyLoss(ignore_index=DATA.PADDING)

    start_epoch = model.load(optimizer)
    if FLAGS.EVAL:
        DEBUG("EVAL mode. Skipping training.")
        return
    DEBUG(f"Starting training at epoch {start_epoch}.")

    best_val_loss = float("inf")
    for epoch in trange(start_epoch, TRAIN.EPOCHS, desc="Training", unit="epoch"):
        train_loss = train_epoch(model, train, optimizer, criterion)
        val_loss = validate(model, val, criterion)
        is_best = epoch < 1 or val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        model.save(optimizer, epoch + 1, is_best)

        DEBUG(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
