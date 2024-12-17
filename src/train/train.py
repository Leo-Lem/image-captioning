from pandas import DataFrame
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import trange, tqdm

from .epoch import train_epoch
from .val import validate
from __param__ import DEBUG, FLAGS, TRAIN
from src.data import Vocabulary, CaptionedImageDataset
from src.model import Decoder


def train(model: Decoder, train: CaptionedImageDataset, val: CaptionedImageDataset):
    optimizer = Adam(model.parameters(), lr=TRAIN.LEARNING_RATE)
    criterion = CrossEntropyLoss(ignore_index=Vocabulary.PADDING)
    losses: DataFrame = model.load(optimizer)
    start_epoch = len(losses)

    if FLAGS.EVAL:
        return

    try:
        for epoch in (epochs := trange(start_epoch, TRAIN.EPOCHS, initial=start_epoch, total=TRAIN.EPOCHS, desc="Epochs", unit="epoch")):
            losses.loc[epoch, "Training"] = \
                train_epoch(model, train.loader(),
                            optimizer, criterion,
                            teacher_forcing_ratio=max(0.5, 1 - epoch / epochs.total))
            losses.loc[epoch, "Validation"] = \
                validate(model, val.loader(), criterion)
            epochs.set_postfix(train=losses["Training"][epoch],
                               val=losses["Validation"][epoch])
            model.save(optimizer, losses)

            if TRAIN.PATIENCE and epochs_without_improvement(losses):
                tqdm.write(f"Early stopping at epoch {epoch}.")
                break
    except KeyboardInterrupt:
        DEBUG("Training interrupted.")
    finally:
        tqdm.write(
            f"Trained for {len(losses)} epochs. Best validation loss: {round(losses['Validation'].min(), 5)} at epoch {losses['Validation'].idxmin()}.")


def epochs_without_improvement(losses: DataFrame) -> bool:
    """ Returns True if none of the last PATIENCE epochs have improved the best value before them."""
    if len(losses) <= TRAIN.PATIENCE:
        return False

    best_before = losses["Validation"][:-TRAIN.PATIENCE].min()
    last_patience = losses["Validation"].iloc[-TRAIN.PATIENCE:]

    return all(last_patience >= best_before)
