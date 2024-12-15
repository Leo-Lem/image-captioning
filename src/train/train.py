from pandas import DataFrame
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import trange, tqdm

from .epoch import train_epoch
from .val import validate
from __param__ import DEBUG, FLAGS, TRAIN, VOCAB


def train(model: Module, train: DataLoader, val: DataLoader):
    """ Train the decoder model. """
    optimizer = Adam(model.parameters(), lr=TRAIN.LEARNING_RATE)
    criterion = CrossEntropyLoss(ignore_index=VOCAB.PADDING)
    losses: DataFrame = model.load(optimizer)
    start_epoch = len(losses)

    if FLAGS.EVAL:
        DEBUG("EVAL mode. Skipping training.")
        return
    DEBUG(f"Starting training at epoch {start_epoch}.")

    try:
        for epoch in (epochs := trange(start_epoch, TRAIN.EPOCHS, initial=start_epoch, desc="Training", unit="epoch")):
            losses.loc[epoch, "Train"] = \
                train_epoch(model, train, optimizer, criterion)
            losses.loc[epoch, "Val"] = validate(model, val, criterion)
            epochs.set_postfix(train=losses["Train"][epoch],
                               val=losses["Val"][epoch])
            model.save(optimizer, losses)

            if TRAIN.PATIENCE and epochs_without_improvement(losses):
                tqdm.write(f"Early stopping at epoch {epoch}.")
                break
    except KeyboardInterrupt:
        DEBUG("Training interrupted.")
        pass
    tqdm.write(
        f"Trained for {len(losses)} epochs. Best validation loss: {round(losses['Val'].min(), 5)} at epoch {losses['Val'].idxmin()}.")


def epochs_without_improvement(losses: DataFrame) -> bool:
    """ Returns True if none of the last PATIENCE epochs have improved the best value before them."""
    if len(losses) <= TRAIN.PATIENCE:
        return False

    best_before = losses["Val"][:-TRAIN.PATIENCE].min()
    last_patience = losses["Val"].iloc[-TRAIN.PATIENCE:]

    return all(last_patience >= best_before)
