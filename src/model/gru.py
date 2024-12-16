from torch import Tensor, tensor, cat, rand
from torch.nn import GRU

from __param__ import DATA, MODEL, FLAGS
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

    def forward(self, image: Tensor, caption: Tensor = None, ratio: float = .5) -> Tensor:
        batch_size = self._validate(image, caption)

        hidden = self._image_to_hidden(image.to(self.device), batch_size)
        index = self._start_index(batch_size).to(self.device)
        logits = tensor([]).to(self.device)

        for i in range(DATA.CAPTION_LEN):
            embedding: Tensor = self.indices_to_embeddings(index.unsqueeze(1))
            assert embedding.size() == (batch_size, 1, MODEL.EMBEDDING_DIM)

            output, hidden = self.gru(embedding, hidden)
            assert output.size() == (batch_size, 1, MODEL.HIDDEN_DIM)

            logit: Tensor = self.hidden_to_logits_fc(output.squeeze(1))
            assert logit.size() == (batch_size, Vocabulary.SIZE)

            index: Tensor = self._predict_index(logit)\
                if caption is None or rand(1).item() < ratio else caption[:, i]
            assert index.size() == (batch_size,)

            logits = cat(
                [logits, logit.unsqueeze(1)], dim=1)

        return self._validate_prediction(logits)
