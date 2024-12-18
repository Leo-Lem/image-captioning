from os.path import exists
from nltk import download
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate import meteor
from nltk.translate.nist_score import corpus_nist
from pandas import DataFrame
from statistics import mean
from tqdm import tqdm
from warnings import filterwarnings

from __param__ import DEBUG, FLAGS, PATHS, MODEL, DATA
from .postprocess import CaptionPostprocessor
from src.data import CaptionedImageDataset
from src.model import Decoder


def test(model: Decoder, data: CaptionedImageDataset):
    """ Test the model with the test dataset using Accuracy, BLEU, METEOR, and NIST metrics. """
    if FLAGS.DESCRIBE:
        DEBUG("Skipping testing due to the USE flag.")
        return

    postprocess = CaptionPostprocessor()

    filterwarnings("ignore",
                   category=UserWarning,
                   module="nltk.translate.bleu_score")

    model.load(best=True)
    download("wordnet", quiet=True)
    metrics = DataFrame(columns=["BLEU", "METEOR", "NIST"])

    for images, captions in tqdm(data.loader(), desc="Testing", unit="batch"):
        true = [
            [caption.split(" ")
             for caption in postprocess(captions[i:i + DATA.NUM_CAPTIONS])]
            for i in range(0, len(captions), DATA.NUM_CAPTIONS)
        ]
        pred = [caption.split(" ")
                for caption in postprocess(model.predict(images[::DATA.NUM_CAPTIONS]))]
        assert len(true) == len(pred)

        def failed(metric: str, e: Exception) -> int:
            DEBUG(
                f"Failed to calculate {metric} (true: {len(true)}, pred: {len(pred)})\n{true}\n{pred}\n{e}")
            return 0

        try:
            bleu_val = corpus_bleu(true, pred, weights=(1, 0, 0, 0))
        except Exception as e:
            bleu_val = failed("BLEU", e)

        try:
            meteor_val = mean([meteor(t, p) for t, p in zip(true, pred)])
        except Exception as e:
            meteor_val = failed("METEOR", e)

        try:
            nist_val = corpus_nist(true, pred, n=4)
        except Exception as e:
            nist_val = failed("NIST", e)

        metrics.loc[len(metrics)] = {
            "BLEU": bleu_val,
            "METEOR": meteor_val,
            "NIST": nist_val
        }
    DEBUG(f"Metrics: {metrics.head(3)}")

    result = DataFrame({"Model": [MODEL.NAME],
                        "BLEU": [metrics["BLEU"].mean()],
                        "METEOR": [metrics["METEOR"].mean()],
                        "NIST": [metrics["NIST"].mean()]})\
        .round(4)\
        .set_index("Model")
    file = PATHS.OUT("metrics.csv")
    result.to_csv(file, index=True, mode="a", header=not exists(file))
