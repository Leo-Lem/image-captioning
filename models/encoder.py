from torch import Tensor
from torch.nn import Module, Sequential

from torchvision.models import resnet50, vgg16


class ImageEncoder(Module):
    """ Image encoder using a pretrained model. """

    def __init__(self, model: Module, dimension: int):
        super().__init__()
        self.dimension = dimension
        self.model = model

    def forward(self, images: Tensor) -> Tensor:
        """ Extract features from a batch of image tensors.

        Args:
            images (Tensor): Batch of image tensors of shape (batch_size, channels=3, height=224, width=224).

        Returns:
            Tensor: Encoded image features of shape (batch_size, num_features=49, feature_size=2048).
        """
        assert images.dim() == 4, "Input images must be 4-dimensional."
        assert images.size(1) == 3, "Input images must have 3 channels."
        assert images.size(2) == 224, "Images must have size 224x224."

        features: Tensor = self.model(images)  # (batch_size, dimension, 7, 7)
        # (batch_size, 7, 7, dimension)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))

        assert features.dim() == 3, "Output must have 3 dimensions."
        assert features.size(0) == images.size(0), \
            "Output batch size must match input batch size."
        assert features.size(1) == 49, "Output must have 49 features."
        assert features.size(2) == self.dimension, \
            "Output feature size must be dimension."

        return features


class ResnetEncoder(ImageEncoder):
    """ Image encoder using a pretrained ResNet-50 model. """

    def __init__(self):
        model = resnet50(weights='ResNet50_Weights.DEFAULT').eval()
        for param in model.parameters():
            param.requires_grad_(False)
        truncated_model = Sequential(*list(model.children())[:-2])
        super().__init__(truncated_model, dimension=2048)


class VGGEncoder(ImageEncoder):
    """ Image encoder using a pretrained VGG-16 model. """

    def __init__(self):
        model = vgg16(weights='VGG16_Weights.DEFAULT').eval()
        for param in model.parameters():
            param.requires_grad_(False)
        truncated_model = model.features  # Use only the feature layers of VGG16
        super().__init__(truncated_model, dimension=512)
