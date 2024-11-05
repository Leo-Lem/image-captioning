from torch import device, Tensor
from torch.optim import Adam, Optimizer
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from models import ImageCaption
from data import Vocabulary


def train(model: ImageCaption,
          data: DataLoader,
          epochs: int = 5,
          device: device = device("cpu"),
          optimizer: Optimizer = None,
          criterion: CrossEntropyLoss = None):
    """
    Train the ImageCaption model.

    Args:
        model (ImageCaption): The caption generation model to train.
        data (DataLoader): Dataloader providing the training data.
        epochs (int): Number of epochs to train for (default: 5).
        device (device): Device to use for training (default: CPU).
        optimizer (Optimizer): Optimizer for training (default: Adam).
        criterion (CrossEntropyLoss): Loss function (default: CrossEntropyLoss).

    Returns:
        ImageCaption: The trained model with the latest checkpoint.
    """
    model.to(device).train()

    optimizer = Adam(model.parameters(),
                     lr=1e-3) if optimizer is None else optimizer
    criterion = CrossEntropyLoss(ignore_index=Vocabulary.pad_index
                                 ) if criterion is None else criterion

    epoch_loss = 0.0
    for epoch in trange(model.load_checkpoint()+1, epochs, desc="Training", unit="epoch"):
        batch_loss = 0.0
        for images, captions in tqdm(data, desc=f"Epoch", unit="batch"):
            optimizer.zero_grad()

            images: Tensor = images.to(device)
            captions: Tensor = captions.to(device)

            outputs: Tensor = model(images, max_len=captions.size(1))
            loss: Tensor = criterion(
                outputs.view(-1, outputs.size(-1)), captions.view(-1))

            loss.backward()
            batch_loss = loss.item()
            epoch_loss += batch_loss

            optimizer.step()
        model.save_checkpoint(epoch)
