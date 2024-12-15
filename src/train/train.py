from pandas import DataFrame
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import trange

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

    for epoch in (epochs := trange(start_epoch, TRAIN.EPOCHS, desc="Training", unit="epoch")):
        losses.loc[epoch, "Train"] = \
            train_epoch(model, train, optimizer, criterion)
        losses.loc[epoch, "Val"] = validate(model, val, criterion)
        epochs.set_postfix(train=losses["Train"][epoch],
                           val=losses["Val"][epoch])
        model.save(optimizer, losses)
