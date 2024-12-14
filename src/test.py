from nltk.metrics import accuracy, precision, recall, f_measure
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from pandas import DataFrame
from torch.utils.data import DataLoader

from .model import Decoder


def test(model: Decoder, data: DataLoader) -> DataFrame:
    """ Test the model with the test dataset using accuracy, recall, precision, f1, BLEU, METEOR, and NIST metrics. """
    model.eval()
    y_true, y_pred = [], []
    for x, y in data:
        y_hat = model(x)
        y_true.extend(y)
        y_pred.extend(y_hat)

    return DataFrame({
        "Accuracy": [accuracy(y_true, y_pred)],
        "Recall": [recall(y_true, y_pred)],
        "Precision": [precision(y_true, y_pred)],
        "F1": [f_measure(y_true, y_pred)],
        "METEOR": [meteor_score(y_true, y_pred)],
        "BLEU": [corpus_bleu(y_true, y_pred)],
        "NIST": [sentence_nist(y_true, y_pred)]
    })
