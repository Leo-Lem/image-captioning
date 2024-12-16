from torch import no_grad, Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from __param__ import DATA, TRAIN
from src.data import Vocabulary


def validate(model: Module, val: DataLoader, criterion: CrossEntropyLoss) -> float:
    model.eval()
    total_loss = 0.0
    with no_grad():
        for image, captions in (batches := tqdm(val, desc="Validation", unit="batch")):
            prediction: Tensor = model(image)
            predictions = prediction\
                .unsqueeze(1)\
                .repeat(1, DATA.NUM_CAPTIONS, 1, 1)
            predictions = predictions.view(-1, predictions.size(-1))
            targets: Tensor = captions.view(-1)

            loss: Tensor = criterion(predictions, targets.to(TRAIN.device)) / \
                targets.ne(Vocabulary.PADDING).sum()
            total_loss += loss.item()
            batches.set_postfix(loss=total_loss / len(batches))
    return total_loss / len(val)
