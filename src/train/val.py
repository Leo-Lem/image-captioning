from torch import no_grad
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from __param__ import DATA


def validate(model: Module, val: DataLoader, criterion: CrossEntropyLoss) -> float:
    """ Validate the model. """
    model.eval()
    total_loss = 0.0
    with no_grad():
        for image, captions in (batches := tqdm(val, desc="Validation", unit="batch")):
            outputs = model(image)
            outputs = outputs.unsqueeze(1).repeat(1, DATA.NUM_CAPTIONS, 1, 1)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = captions.view(-1)
            assert outputs.size(0) == targets.size(0)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            batches.set_postfix(loss=loss.item())
    return total_loss / len(val)
