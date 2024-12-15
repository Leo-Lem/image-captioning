from torch import full, long, stack, Tensor, zeros
from torch.nn import LSTM

from __param__ import DATA, MODEL
from .decoder import Decoder
from src.data import Vocabulary


class LSTMDecoder(Decoder):
    """ LSTM-based decoder for sequence generation. """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = LSTM(input_size=MODEL.EMBEDDING_DIM,
                         hidden_size=MODEL.HIDDEN_DIM,
                         num_layers=MODEL.NUM_LAYERS,
                         dropout=MODEL.DROPOUT,
                         batch_first=True)

    def forward(self, image: Tensor) -> Tensor:
        """ Predict the caption for the given image. """
        batch_size = image.size(0)
        assert image.size() == (batch_size, 1, DATA.FEATURE_DIM)

        hidden = (self.image_fc(image).squeeze(1).repeat(MODEL.NUM_LAYERS, 1, 1),
                  zeros((1, batch_size, MODEL.HIDDEN_DIM), device=image.device).repeat(MODEL.NUM_LAYERS, 1, 1))

        assert hidden[0].size() == hidden[1].size() == \
            (MODEL.NUM_LAYERS, batch_size, MODEL.HIDDEN_DIM)

        input = full((batch_size, 1), fill_value=Vocabulary.START,
                     device=image.device)
        embedding = self.embedding(input)
        assert embedding.size() == (batch_size, 1, MODEL.EMBEDDING_DIM)

        outputs = []

        for _ in range(DATA.CAPTION_LEN):
            output, hidden = self.lstm(embedding, hidden)
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
