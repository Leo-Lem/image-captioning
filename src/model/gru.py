from torch import Tensor, full, tensor, cat, multinomial, softmax, rand
from torch.nn import GRU

from __param__ import DATA, MODEL
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
        batch_size = self.validate(image, caption)

        hidden: Tensor = self.image_to_hidden_fc(image)\
            .squeeze(1)\
            .repeat(MODEL.NUM_LAYERS, 1, 1)
        assert hidden.size() == (MODEL.NUM_LAYERS, batch_size, MODEL.HIDDEN_DIM)

        index: Tensor = full((batch_size,), fill_value=Vocabulary.START)
        assert index.size() == (batch_size,)

        logits_sequence = tensor([])

        for _ in range(DATA.CAPTION_LEN):
            embedding: Tensor = self.indices_to_embeddings(index.unsqueeze(1))
            assert embedding.size() == (batch_size, 1, MODEL.EMBEDDING_DIM)

            output, hidden = self.gru(embedding, hidden)
            assert output.size() == (batch_size, 1, MODEL.HIDDEN_DIM)

            logits: Tensor = self.hidden_to_logits_fc(output.squeeze(1))
            assert logits.size() == (batch_size, Vocabulary.SIZE)

            index: Tensor = multinomial(softmax(logits, dim=-1), num_samples=1).squeeze(-1)\
                if caption is None or rand(1).item() < ratio else caption[:, _]
            assert index.size() == (batch_size,)

            logits_sequence = cat(
                [logits_sequence, logits.unsqueeze(1)], dim=1)

        return self.validate_prediction(logits_sequence)
