from torch.optim import Adam
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model: Module, train: DataLoader, optimizer: Adam, criterion: CrossEntropyLoss) -> float:
    """ Train the model for one epoch. """
    model.train()
    total_loss = 0.0
    for batch in tqdm(train, desc="Epoch", unit="batch"):
        image, caption = batch
        optimizer.zero_grad()

        outputs = model(image)
        loss = criterion(outputs.view(-1, outputs.size(-1)),
                         caption.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    avg_train_loss = total_loss / len(train)
    return avg_train_loss
