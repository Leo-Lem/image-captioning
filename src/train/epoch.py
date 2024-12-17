from torch import Tensor, stack
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import Decoder


def train_epoch(model: Decoder, train: DataLoader, optimizer: Adam, criterion: CrossEntropyLoss, teacher_forcing_ratio: float) -> float:
    """ Train the model for one epoch. """
    model.train()
    total_loss = 0.0

    for images, captions in (batches := tqdm(train, desc="Training", unit="batch")):
        optimizer.zero_grad(set_to_none=True)

        predictions: Tensor = model(images, captions, teacher_forcing_ratio)
        targets: Tensor = captions[:, 1:]
        assert targets.size() == (predictions.size(0), predictions.size(1))

        loss: Tensor = criterion(predictions.view(-1, predictions.size(-1)),
                                 targets.reshape(-1))
        total_loss += loss.item()
        batches.set_postfix(loss=loss.item())

        loss.backward()
        optimizer.step()

    return total_loss / len(train)
