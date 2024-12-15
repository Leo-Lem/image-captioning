from torch.optim import Adam
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from __param__ import DATA


def train_epoch(model: Module, train: DataLoader, optimizer: Adam, criterion: CrossEntropyLoss) -> float:
    """ Train the model for one epoch. """
    model.train()
    total_loss = 0.0

    for batch in (batches := tqdm(train, desc="Training", unit="batch")):
        image, captions = batch
        optimizer.zero_grad()

        outputs = model(image)
        outputs = outputs.unsqueeze(1).repeat(1, DATA.NUM_CAPTIONS, 1, 1)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = captions.view(-1)
        assert outputs.size(0) == targets.size(0)

        loss = criterion(outputs, targets)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        batches.set_postfix(loss=loss.item())

    return total_loss / len(train)
