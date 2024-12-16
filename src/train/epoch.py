from torch import Tensor
from torch.optim import Adam
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from __param__ import DATA, TRAIN
from src.data import Vocabulary
from src.model import Decoder


def train_epoch(model: Decoder, train: DataLoader, optimizer: Adam, criterion: CrossEntropyLoss, teacher_forcing_ratio: float) -> float:
    """ Train the model for one epoch. """
    model.train()
    total_loss = 0.0

    for image, captions in (batches := tqdm(train, desc="Training", unit="batch")):
        image: Tensor = image.to(TRAIN.DEVICE)
        captions: Tensor = captions.to(TRAIN.DEVICE)

        optimizer.zero_grad()

        prediction: Tensor = model(image,
                                   captions[:, 0, :], teacher_forcing_ratio)
        predictions = prediction\
            .unsqueeze(1)\
            .repeat(1, DATA.NUM_CAPTIONS, 1, 1)
        predictions = predictions.view(-1, predictions.size(-1))
        targets: Tensor = captions.view(-1)

        loss: Tensor = criterion(predictions, targets) / \
            targets.ne(Vocabulary.PADDING).sum()
        total_loss += loss.item()
        batches.set_postfix(loss=total_loss / len(batches))

        loss.backward()
        optimizer.step()

    return total_loss / len(train)
