from torch import full, long, stack, Tensor
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

    def forward(self, image: Tensor) -> Tensor:
        batch_size = image.size(0)
        assert image.size() == (batch_size, 1, DATA.FEATURE_DIM)

        hidden = self.image_fc(image).squeeze(1).repeat(MODEL.NUM_LAYERS, 1, 1)
        assert hidden.size() == (MODEL.NUM_LAYERS, batch_size, MODEL.HIDDEN_DIM)

        input = full((batch_size, 1), fill_value=Vocabulary.START,
                     device=image.device)
        embedding = self.embedding(input)
        assert embedding.size() == (batch_size, 1, MODEL.EMBEDDING_DIM)

        outputs = []

        for _ in range(DATA.CAPTION_LEN):
            output, hidden = self.gru(embedding, hidden)
            assert output.size() == (batch_size, 1, MODEL.HIDDEN_DIM)

            output = self.fc(output.squeeze(1))
            assert output.size() == (batch_size, Vocabulary.SIZE)

            outputs.append(output)

            input = output.argmax(1).unsqueeze(1)
            assert input.size() == (batch_size, 1)

            embedding = self.embedding(input)
            assert embedding.size() == (batch_size, 1, MODEL.EMBEDDING_DIM)

        outputs = stack(outputs, dim=1)
        assert outputs.size() == (batch_size, DATA.CAPTION_LEN, Vocabulary.SIZE)

        return outputs
