from torch import full, long, stack, Tensor
from torch.nn import LSTM

from __param__ import DATA, VOCAB, MODEL, TRAIN
from .decoder import Decoder


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
        # TODO: allow smaller batch size (to handle the last batch) (same for the other models)
        assert image.size() == (TRAIN.BATCH_SIZE, 1, DATA.FEATURE_DIM)

        # TODO: pass image to hidden layer and intialize the lstm with "<start>" (same for the other models)
        hidden = None
        input = full((TRAIN.BATCH_SIZE, 1),
                     VOCAB.START,
                     dtype=long,
                     device=image.device)
        embedding = self.image_fc(image)
        outputs = []

        for _ in range(DATA.CAPTION_LEN):
            output, hidden = self.lstm(embedding, hidden)
            assert output.size() == (TRAIN.BATCH_SIZE, 1, MODEL.HIDDEN_DIM)

            output = self.fc(output.squeeze(1))
            assert output.size() == (TRAIN.BATCH_SIZE, VOCAB.SIZE)

            outputs.append(output)
            input = output.argmax(1).unsqueeze(1)
            assert input.size() == (TRAIN.BATCH_SIZE, 1)

            embedding = self.embedding(input)
            assert embedding.size() == (TRAIN.BATCH_SIZE, 1, MODEL.EMBEDDING_DIM)

        outputs = stack(outputs, dim=1)
        assert outputs.size() == (TRAIN.BATCH_SIZE, DATA.CAPTION_LEN, VOCAB.SIZE)

        return outputs
