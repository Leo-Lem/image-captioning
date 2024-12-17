from torch import Tensor, arange
from torch.nn import TransformerDecoderLayer, Embedding, Linear
from torch.nn import TransformerDecoder as TransformerDecoderTorch

from __param__ import DATA, MODEL, TRAIN
from .decoder import Decoder
from src.data import Vocabulary


class TransformerDecoder(Decoder):
    """ Transformer-based decoder for sequence generation. """

    def __init__(self) -> None:
        super().__init__()
        self.memory_projection = Linear(MODEL.HIDDEN_DIM, MODEL.EMBEDDING_DIM)
        self.output_projection = Linear(MODEL.EMBEDDING_DIM, Vocabulary.SIZE)
        self.positional_embeddings = Embedding(DATA.CAPTION_LEN-1,
                                               MODEL.EMBEDDING_DIM)
        self.transformer_decoder = TransformerDecoderTorch(
            TransformerDecoderLayer(d_model=MODEL.EMBEDDING_DIM,
                                    nhead=8,
                                    dim_feedforward=MODEL.HIDDEN_DIM,
                                    dropout=MODEL.DROPOUT,
                                    batch_first=True),
            num_layers=MODEL.NUM_LAYERS)
        self.to(TRAIN.DEVICE)

    def forward(self, image: Tensor, caption: Tensor = None, _=None) -> Tensor:
        batch_size = self._validate(image, caption)

        memory = self._image_to_hidden(image, batch_size)
        memory = self.memory_projection(memory).transpose(0, 1)

        positions = arange(DATA.CAPTION_LEN-1, device=TRAIN.DEVICE)
        positional_embedding = self.positional_embeddings(positions)\
            .unsqueeze(0)\
            .repeat(batch_size, 1, 1)

        output = self.transformer_decoder(positional_embedding, memory)
        assert output.size() == (batch_size, DATA.CAPTION_LEN-1, MODEL.EMBEDDING_DIM)

        logits = self.output_projection(output)
        assert logits.size() == (batch_size, DATA.CAPTION_LEN-1, Vocabulary.SIZE)

        return logits
