from logging import debug as DEBUG
from torch import no_grad
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import trange, tqdm

from .model import Decoder
from __param__ import FLAGS, TRAIN, DATA


def train(model: Decoder, train: DataLoader, val: DataLoader):
    """ Train the decoder model. """
    if FLAGS.EVAL:
        DEBUG("EVAL mode. Skipping training.")
        return

    optimizer = Adam(model.parameters(), lr=TRAIN.LEARNING_RATE)
    criterion = CrossEntropyLoss(ignore_index=DATA.PADDING)

    start_epoch = model.load(optimizer)
    DEBUG(f"Starting training at epoch {start_epoch}.")

    for epoch in trange(start_epoch, TRAIN.EPOCHS, desc="Training", unit="epoch"):
        avg_train_loss = train_iteration(model, train, optimizer, criterion)
        avg_val_loss = validate(model, val, criterion)
        is_best = True if epoch < 1 else \
            avg_val_loss <= min(val_loss for _, _, val_loss in val)
        model.save(optimizer, epoch + 1, is_best)

        DEBUG(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")


def train_iteration(model: Decoder, train: DataLoader, optimizer: Adam, criterion: CrossEntropyLoss) -> float:
    """ Train the model for one epoch. """
    model.train()
    total_loss = 0.0
    for batch in tqdm(train, desc="Epoch", unit="batch"):
        image, captions = batch
        optimizer.zero_grad()

        outputs = model(image, captions[:, 0, :])
        loss = criterion(outputs.view(-1, outputs.size(-1)),
                         captions.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(train)
    return avg_train_loss


def validate(model: Decoder, val: DataLoader, criterion: CrossEntropyLoss) -> float:
    """ Validate the model. """
    model.eval()
    val_loss = 0.0
    with no_grad():
        for batch in tqdm(val, desc="Validation", unit="batch"):
            image, captions = batch
            outputs = model(image, captions[:, 0, :])
            loss = criterion(outputs.view(-1, outputs.size(-1)),
                             captions.view(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val)
    return avg_val_loss
