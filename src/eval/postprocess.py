from torch import Tensor
from torch.nn.functional import softmax

from __param__ import DATA, VOCAB, TRAIN


def unprocess(predictions: Tensor, reversed_vocab: dict[int, str]) -> list[list[str]]:
    """ Unprocess the model output (tensor embedding) back to a string. """
    assert predictions.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN, VOCAB.SIZE)

    probabilities = softmax(predictions, dim=-1)
    assert probabilities.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN, VOCAB.SIZE)

    decoded_captions = probabilities.argmax(dim=-1)
    assert decoded_captions.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN)

    captions = [stringify(caption, reversed_vocab)
                for caption in decoded_captions]

    return captions


def stringify(caption: Tensor, reversed_vocab: dict[int, str]) -> list[str]:
    """ Convert a caption to a string. """
    caption = caption.tolist()
    caption = [reversed_vocab[token] for token in caption
               if token not in (VOCAB.START, VOCAB.PADDING, VOCAB.END, VOCAB.UNKNOWN)]
    return caption
