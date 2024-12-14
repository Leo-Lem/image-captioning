from logging import basicConfig
from argparse import ArgumentParser
from os import path
from torch import cuda

parser = ArgumentParser(
    description="Train an image captioning pipeline with different decoder architectures")
parser.add_argument("approach", type=str,
                    help="lstm, gru, transformer")

parser.add_argument("-p", "--path", type=str, default="",
                    help="path to load/store the trained model")
parser.add_argument("-d", "--debug", action="store_true",
                    help="run in debug mode")
parser.add_argument("--eval", action="store_true",
                    help="skip training the model")

parser.add_argument("--epochs", type=int, default=10,
                    help="number of epochs to train the model")
parser.add_argument("--stop", type=int, default=3,
                    help="number of epochs without improvement to stop training")
parser.add_argument("--batch", type=int, default=1024,
                    help="batch size for training")

parser.add_argument("--reload", action="store_true",
                    help="reload the data")
parser.add_argument("--sample", action="store_true",
                    help="use sample data instead of full data")

args = parser.parse_args()


class PATHS:
    OUT = path.join(path.dirname(path.dirname(__file__)), ".out")
    RES = path.join(path.dirname(__file__), "res")
    MODEL = args.path if args.path else OUT
    VOCAB = path.join(RES, "vocab.csv")


APPROACH = args.approach


class MODEL:
    HIDDEN_DIM = 512
    EMBEDDING_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.1
    ATTENTION_HEADS = 8


class TRAIN:
    EPOCHS = args.epochs
    STOP_EARLY_AFTER = args.stop
    BATCH_SIZE = args.batch
    LEARNING_RATE = 1e-3


class DATA:
    SAMPLE = args.sample
    RELOAD = args.reload
    VOCAB_SIZE = 8096
    VOCAB_THRESHOLD = 3
    FEATURE_DIM = 1280
    CAPTION_LEN = 20
    NUM_CAPTIONS = 5
    PADDING = 0


class FLAGS:
    DEBUG = args.debug
    EVAL = args.eval
    GPU = cuda.is_available()


if args.debug:
    basicConfig(level="DEBUG")
