from torch import Tensor, tensor, cat, rand
from torch.nn import GRU

from __param__ import DATA, MODEL, TRAIN
from .decoder import Decoder
from src.data import Vocabulary


class GRUDecoder(Decoder):
    """ GRU-based decoder for sequence generation. """

    def __init__(self) -> None:
        super().__init__()
        self.gru = GRU(input_size=MODEL.EMBEDDING_DIM,
                       hidden_size=MODEL.HIDDEN_DIM,
                       num_layers=MODEL.NUM_LAYERS,
                       dropout=MODEL.DROPOUT,
                       batch_first=True)
        self.to(TRAIN.DEVICE)

    def forward(self, image: Tensor, caption: Tensor = None, ratio: float = .5) -> Tensor:
        batch_size = self._validate(image, caption)

        hidden = self._image_to_hidden(image, batch_size)
        index = self._start_index(batch_size)
        logits = []

        for i in range(DATA.CAPTION_LEN-1):
            embedding: Tensor = self.indices_to_embeddings(index.unsqueeze(1))
            assert embedding.size() == (batch_size, 1, MODEL.EMBEDDING_DIM)

            output, hidden = self.gru(embedding, hidden)
            assert output.size() == (batch_size, 1, MODEL.HIDDEN_DIM)

            logit: Tensor = self.hidden_to_logits_fc(output.squeeze(1))
            assert logit.size() == (batch_size, Vocabulary.SIZE)
            logits.append(logit)

            index: Tensor = self._predict_index(logit)\
                if caption is None or ratio < rand(1).item() else caption[:, i]
            assert index.size() == (batch_size,)

        return self._validate_prediction(logits)
