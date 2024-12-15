from nltk import download
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor
from nltk.translate.nist_score import corpus_nist
from pandas import DataFrame
from statistics import mean
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
from warnings import filterwarnings

from __param__ import DEBUG, FLAGS
from .postprocess import extract_from_embedding, extract_from_tokenized
from .store import store_results


def test(model: Module, data: DataLoader, reversed_vocab: dict[int, str]) -> DataFrame:
    """ Test the model with the test dataset using Accuracy, BLEU, METEOR, and NIST metrics. """
    if FLAGS.DESCRIBE:
        DEBUG("Skipping testing due to the USE flag.")
        return DataFrame()

    filterwarnings("ignore",
                   category=UserWarning,
                   module="nltk.translate.bleu_score")

    model.load(best=True)
    model.eval()
    download("wordnet", quiet=True)
    metrics = DataFrame(columns=["BLEU", "METEOR", "NIST"])

    for images, captions in tqdm(data, desc="Testing", unit="batch"):
        true = [[caption.split(" ")
                for caption in captions]
                for captions in extract_from_tokenized(captions, reversed_vocab)]
        pred = [caption.split(" ")
                for caption in extract_from_embedding(model(images), reversed_vocab)]

        def failed(metric: str, e: Exception) -> int:
            DEBUG(
                f"Failed to calculate {metric} (true: {len(true)}, pred: {len(pred)})\n{true}\n{pred}\n{e}")
            return 0

        try:
            bleu_val = corpus_bleu(true, pred)
        except Exception as e:
            bleu_val = failed("BLEU", e)

        try:
            meteor_val = mean([meteor(t, p) for t, p in zip(true, pred)])
        except Exception as e:
            meteor_val = failed("METEOR", e)

        try:
            nist_val = corpus_nist(true, pred)
        except Exception as e:
            nist_val = failed("NIST", e)

        metrics.loc[len(metrics)] = {
            "BLEU": bleu_val,
            "METEOR": meteor_val,
            "NIST": nist_val
        }

    DEBUG(f"Metrics: {metrics.head(3)}")
    result = DataFrame({"Model": [model.__class__.__name__],
                        "BLEU": [metrics["BLEU"].mean()],
                        "METEOR": [metrics["METEOR"].mean()],
                        "NIST": [metrics["NIST"].mean()]})\
        .round(4)\
        .set_index("Model")
    store_results(result)
    return result
