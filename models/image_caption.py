from torch import Tensor

from .checkpoint import CheckpointModule
from .encoder import ImageEncoder
from .decoder import CaptionDecoder


class ImageCaption(CheckpointModule):
    def __init__(self, encoder: ImageEncoder, decoder: CaptionDecoder, checkpoint_dir: str):
        super().__init__(
            f"{checkpoint_dir}/{encoder.__class__.__name__}-{decoder.__class__.__name__}.checkpoint")
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: Tensor, max_len: int = 15) -> Tensor:
        """
        Generate captions for a batch of images.

        Args:
            images: A tensor of shape (batch_size, channels=3, height=224, width=224) containing the images.
            max_len: The maximum length of the captions to generate.

        Returns:
            A tensor of shape (batch_size, max_len, vocabulary_size) containing the logits for each token position.
        """
        assert images.dim() == 4, "Images must have 4 dimensions."
        assert images.size(1) == 3, "Images must have 3 channels."
        assert images.size(2) == 224 and images.size(
            3) == 224, "Images must have size 224x224."

        features = self.encoder(images)
        logits: Tensor = self.decoder(features, max_len)

        assert logits.dim() == 3, "Logits must have 3 dimensions (batch_size, max_len, vocabulary_size)."
        assert logits.size(0) == images.size(0), "Batch size must match."
        assert logits.size(1) == max_len, "Caption length must match."

        return logits
