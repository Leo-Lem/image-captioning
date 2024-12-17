from torch import no_grad, Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm


def validate(model: Module, val: DataLoader, criterion: CrossEntropyLoss) -> float:
    model.eval()
    total_loss = 0.0
    with no_grad():
        for images, captions in (batches := tqdm(val, desc="Validation", unit="batch")):
            predictions: Tensor = model(images)
            targets: Tensor = captions[:, 1:]
            assert targets.size() == (predictions.size(0), predictions.size(1))

            loss: Tensor = criterion(predictions.view(-1, predictions.size(-1)),
                                     targets.reshape(-1))
            total_loss += loss.item()
            batches.set_postfix(loss=loss.item())
    return total_loss / len(val)
