from torch import no_grad, device, Tensor
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from models import ImageCaption
from old import FlickrDataset


def evaluate(model: ImageCaption,
             data: DataLoader,
             dataset: FlickrDataset,
             device: device = device("cpu")) -> float:
    """
    Evaluate the ImageCaption model on the provided dataset using BLEU score.

    Args:
        model (ImageCaption): The image captioning model to evaluate.
        data (DataLoader): DataLoader providing images and reference captions.
        device (torch.device): Device to run evaluation on (default: CPU).

    Returns:
        float: The BLEU score for the evaluated dataset.
    """
    model.to(device).eval()

    references, candidates = [], []
    with no_grad():
        bleu = 0.0
        for images, captions in tqdm(data, desc="Evaluation", unit="batch"):
            images: Tensor = images.to(device)
            predictions: Tensor = model(images)

            for index, prediction in enumerate(predictions):
                references.append([dataset.vocabulary.tokenize(
                    dataset.tensor_to_caption(captions[index]))])
                candidates.append(dataset.vocabulary.tokenize(
                    dataset.tensor_to_caption(prediction.argmax(dim=1))))

            bleu = corpus_bleu(references, candidates)

    return bleu
