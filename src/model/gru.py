from torch import full, long, stack, Tensor
from torch.nn import GRU

from __param__ import DATA, VOCAB, MODEL, TRAIN
from .decoder import Decoder


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
        """ Predict the caption for the given image. """
        assert image.size() == (image.size(0), 1, DATA.FEATURE_DIM)

        hidden = None
        input = full((image.size(0), 1), VOCAB.START,
                     dtype=long, device=image.device)
        embedding = self.image_fc(image)
        outputs = []

        for _ in range(DATA.CAPTION_LEN):
            output, hidden = self.gru(embedding, hidden)
            assert output.size() == (image.size(0), 1, MODEL.HIDDEN_DIM)

            output = self.fc(output.squeeze(1))
            assert output.size() == (image.size(0), VOCAB.SIZE)

            outputs.append(output)
            input = output.argmax(1).unsqueeze(1)
            assert input.size() == (image.size(0), 1)

            embedding = self.embedding(input)
            assert embedding.size() == (image.size(0), 1, MODEL.EMBEDDING_DIM)

        outputs = stack(outputs, dim=1)
        assert outputs.size() == (image.size(0), DATA.CAPTION_LEN, VOCAB.SIZE)

        return outputs
