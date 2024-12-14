from logging import debug as DEBUG
from nltk.metrics import accuracy, precision, recall, f_measure
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
from pandas import DataFrame
from torch.nn import Module
from torch.utils.data import DataLoader

from .postprocess import unprocess, stringify
from __param__ import PATHS


def test(model: Module, data: DataLoader, reversed_vocab: dict[int, str]) -> DataFrame:
    """ Test the model with the test dataset using accuracy, recall, precision, f1, BLEU, METEOR, and NIST metrics. """
    model.eval()
    y_true, y_pred = set(), set()
    for x, y in data:
        y_true.update([stringify(caption, reversed_vocab) for caption in y])
        y_pred.update(unprocess(model(x), reversed_vocab))
    DEBUG(f"y_true: {len(y_true)}\n{y_true}\ny_pred: {len(y_pred)}\n{y_pred}")

    return DataFrame({
        "Accuracy": [accuracy(y_true, y_pred)],
        "Recall": [recall(y_true, y_pred)],
        "Precision": [precision(y_true, y_pred)],
        "F1": [f_measure(y_true, y_pred)],
        "METEOR": [meteor_score(y_true, y_pred)],
        "BLEU": [corpus_bleu(y_true, y_pred)],
        "NIST": [sentence_nist(y_true, y_pred)]
    })


def store_results(results: DataFrame) -> None:
    """ Store the results in a CSV file. """
    results.to_csv(PATHS.OUT, index=False)
