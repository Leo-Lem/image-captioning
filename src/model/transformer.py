from torch import full, long, stack, triu, zeros, ones, Tensor
from torch.nn import Linear, Parameter, TransformerDecoderLayer
from torch.nn import TransformerDecoder as TransformerDecoderTorch

from __param__ import MODEL, DATA, VOCAB
from .decoder import Decoder


class TransformerDecoder(Decoder):
    """ Transformer-based decoder for sequence generation. """

    def __init__(self) -> None:
        super().__init__()
        max_length = 100
        self.pos_enc = Parameter(zeros(1, max_length, MODEL.EMBEDDING_DIM))
        self.transformer_decoder = TransformerDecoderTorch(
            TransformerDecoderLayer(d_model=MODEL.EMBEDDING_DIM,
                                    nhead=MODEL.ATTENTION_HEADS,
                                    dim_feedforward=MODEL.HIDDEN_DIM,
                                    dropout=MODEL.DROPOUT,
                                    batch_first=True),
            num_layers=MODEL.NUM_LAYERS)
        self.fc = Linear(in_features=MODEL.EMBEDDING_DIM,
                         out_features=VOCAB.SIZE)

    # TODO: transformer is predicting empty captions
    def forward(self, image: Tensor) -> Tensor:
        """ Predict the caption for the given image. """
        assert image.size() == (image.size(0), 1, DATA.FEATURE_DIM)

        input = full((image.size(0), 1),
                     DATA.START,
                     dtype=long,
                     device=image.device)
        memory = self.image_fc(image)
        memory += self.pos_enc[:, :1, :]
        outputs = []

        for t in range(DATA.CAPTION_LEN):
            tgt = self.embedding(input) + self.pos_enc[:, :t + 1, :]
            tgt_mask = self.mask(t + 1).to(image.device)

            decoder_output = self.transformer_decoder(tgt=tgt,
                                                      memory=memory,
                                                      tgt_mask=tgt_mask
                                                      )
            output = self.fc(decoder_output[:, -1, :])
            assert output.size() == (image.size(0), VOCAB.SIZE)

            outputs.append(output)
            input = output.argmax(1).unsqueeze(1)
            assert input.size() == (image.size(0), 1)

        outputs = stack(outputs, dim=1)
        assert outputs.size() == (image.size(0), DATA.CAPTION_LEN, VOCAB.SIZE)

        return outputs

    @staticmethod
    def mask(size: int) -> Tensor:
        mask = triu(ones(size, size), diagonal=1).bool()
        return mask
