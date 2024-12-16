from torch import full, long, stack, triu, zeros, ones, Tensor
from torch.nn import Linear, Parameter, TransformerDecoderLayer
from torch.nn import TransformerDecoder as TransformerDecoderTorch

from __param__ import MODEL, DATA
from .decoder import Decoder
from src.data import Vocabulary


class TransformerDecoder(Decoder):
    """ Transformer-based decoder for sequence generation. """

    def __init__(self) -> None:
        super().__init__()
        max_length = 100
        max_length = 100
        self.pos_enc = Parameter(zeros(1, max_length, MODEL.EMBEDDING_DIM))
        self.image_fc = Linear(DATA.FEATURE_DIM, MODEL.EMBEDDING_DIM)
        self.transformer_decoder = TransformerDecoderTorch(
            TransformerDecoderLayer(d_model=MODEL.EMBEDDING_DIM,
                                    nhead=MODEL.ATTENTION_HEADS,
                                    dim_feedforward=MODEL.HIDDEN_DIM,
                                    dropout=MODEL.DROPOUT,
                                    batch_first=True),
            num_layers=MODEL.NUM_LAYERS)
        self.fc = Linear(MODEL.EMBEDDING_DIM, Vocabulary.SIZE)

    def forward(self, image: Tensor) -> Tensor:
        batch_size = image.size(0)
        assert image.size() == (batch_size, 1, DATA.FEATURE_DIM)

        input = full((batch_size, 1), Vocabulary.START, device=image.device)
        memory = self.image_fc(image).squeeze(1).unsqueeze(1)
        assert memory.size() == (batch_size, 1, MODEL.EMBEDDING_DIM)

        outputs = []

        for t in range(DATA.CAPTION_LEN):
            tgt = self.indices_to_embeddings(
                input) + self.pos_enc[:, :t + 1, :]
            tgt_mask = self.mask(t + 1).to(image.device)

            decoder_output = self.transformer_decoder(
                tgt=tgt, memory=memory, tgt_mask=tgt_mask)
            output = self.fc(decoder_output[:, -1, :])
            assert output.size() == (batch_size, Vocabulary.SIZE)

            outputs.append(output)

            input = output.argmax(1).unsqueeze(1)
            assert input.size() == (batch_size, 1)

        outputs = stack(outputs, dim=1)
        assert outputs.size() == (batch_size, DATA.CAPTION_LEN, Vocabulary.SIZE)

        return outputs

    @staticmethod
    def mask(size: int) -> Tensor:
        return triu(ones(size, size), diagonal=1).bool()
