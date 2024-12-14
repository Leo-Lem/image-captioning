from torch import no_grad
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm


def validate(model: Module, val: DataLoader, criterion: CrossEntropyLoss) -> float:
    """ Validate the model. """
    model.eval()
    val_loss = 0.0
    with no_grad():
        for batch in tqdm(val, desc="Validation", unit="batch"):
            image, caption = batch
            outputs = model(image)
            loss = criterion(outputs.view(-1, outputs.size(-1)),
                             caption.view(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val)
    return avg_val_loss
