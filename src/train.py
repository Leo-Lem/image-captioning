from pandas import DataFrame
from torch import no_grad, Tensor, stack
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import trange

from .model import Decoder, load_model, save_model
from __param__ import FLAGS, TRAIN


def collate_fn(batch) -> tuple[Tensor, Tensor, Tensor]:
    """
    Custom collate function to prepare batches for training/validation.

    Args:
        batch (list): List of tuples (image_features, captions, targets).

    Returns:
        tuple[Tensor, Tensor, Tensor]: Batched image features, captions, and targets.
    """
    # Shape: (batch_size, feature_dim)
    image_features = stack([item['vector'] for item in batch])
    # Shape: (batch_size, seq_len)
    captions = stack([item['caption'] for item in batch])
    # Shape: (batch_size, seq_len)
    targets = stack([item['target'] for item in batch])
    return image_features, captions, targets


def train(model: Decoder, train_data: DataFrame, val_data: DataFrame):
    """ Train the decoder model."""
    if FLAGS.EVAL:
        if FLAGS.DEBUG:
            print("EVAL mode. Skipping training.")
        return

    optimizer = Adam(model.parameters(), lr=TRAIN.LEARNING_RATE)
    criterion = CrossEntropyLoss()

    train_loader = DataLoader(
        train_data, batch_size=TRAIN.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(
        val_data, batch_size=TRAIN.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    start_epoch = load_model(model, optimizer)
    if FLAGS.DEBUG:
        print(f"Starting training at epoch {start_epoch}.")

    for epoch in trange(start_epoch, TRAIN.EPOCHS, desc="Training Epochs"):
        model.train()  # Set model to training mode
        total_loss = 0.0

        # Training loop
        for batch in train_loader:
            image_features, captions, targets = batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(image_features, captions[:, :-1])
            loss = criterion(
                outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with no_grad():
            for batch in val_loader:
                image_features, captions, targets = batch
                outputs = model(image_features, captions[:, :-1])
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)), targets.view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # Log progress
        if FLAGS.DEBUG:
            print(
                f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        is_best = avg_val_loss <= min(
            val_loss for _, _, val_loss in val_loader) if epoch > 0 else True
        save_model(model, optimizer, epoch + 1, is_best)
