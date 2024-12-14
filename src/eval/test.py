from logging import debug as DEBUG
from nltk.metrics.scores import accuracy
from nltk import download
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor
from nltk.translate.nist_score import corpus_nist
from pandas import DataFrame
from statistics import mean
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .postprocess import unprocess, stringify
from .store import store_results


def test(model: Module, data: DataLoader, reversed_vocab: dict[int, str]) -> DataFrame:
    """ Test the model with the test dataset using Accuracy, BLEU, METEOR, and NIST metrics. """
    model.eval()
    download("wordnet", quiet=True)
    metrics = DataFrame(columns=["Accuracy", "BLEU", "METEOR", "NIST"])
    for images, captions in tqdm(data, desc="Testing", unit="batch"):
        true = [stringify(caption, reversed_vocab)
                for caption in captions]
        pred = unprocess(model(images), reversed_vocab)
        try:
            metrics.loc[len(metrics)] = ({"Accuracy": accuracy(true, pred),
                                          "BLEU": corpus_bleu([[t] for t in true], pred),
                                          "METEOR": mean([meteor([[t0] for t0 in t], p) for t, p in zip(true, pred)]),
                                          "NIST": corpus_nist([[t] for t in true], pred)})
        except:
            DEBUG(f"Failed to calculate for batch: {true}, {pred}")
            pass
    result = {"Model": model.__class__.__name__, **metrics.mean().round(4)}
    store_results(result)
    DEBUG(f"Metrics: {metrics.head(3)} -> {result}")
    return result
