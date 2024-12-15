from logging import basicConfig, debug
from argparse import ArgumentParser
from os import path, makedirs
from torch import cuda

parser = ArgumentParser(
    description="Train an image captioning pipeline with different decoder architectures")
parser.add_argument("approach", type=str,
                    help="lstm, gru, transformer")

parser.add_argument("-p", "--path", type=str, default="",
                    help="path to load/store the trained model")
parser.add_argument("-r", "--resources", type=str, default="res",
                    help="path to the resources folder")

parser.add_argument("-d", "--debug", action="store_true",
                    help="run in debug mode")
parser.add_argument("--eval", action="store_true",
                    help="skip training the model")
parser.add_argument("--use", action="store_true",
                    help="no training or testing, only use the model")

parser.add_argument("--epochs", type=int, default=500,
                    help="number of epochs to train the model")
parser.add_argument("--patience", type=int, default=10,
                    help="number of epochs without improvement to stop training")
parser.add_argument("--batch", type=int, default=512,
                    help="batch size for training")

parser.add_argument("--reload", action="store_true",
                    help="reload the data")
parser.add_argument("--sample", action="store_true",
                    help="use sample data instead of full data")

args = parser.parse_args()


class FLAGS:
    EVAL = args.eval or args.use
    USE = args.use
    GPU = cuda.is_available()

    @staticmethod
    def DEBUG(message: str) -> bool:
        if args.debug:
            basicConfig(level="DEBUG")
            debug(message)
        return args.debug


def DEBUG(message: str) -> bool:
    FLAGS.DEBUG(message)


class PATHS:
    @staticmethod
    def RESOURCES(*files: str) -> str:
        res = path.join(path.dirname(__file__), args.resources, *files)
        assert path.exists(res), f"Resource {res} not found!"
        return res

    @staticmethod
    def OUT(*files: str) -> str:
        out = path.join(path.dirname(__file__), ".out")
        makedirs(out, exist_ok=True)
        return path.join(out, *files)

    @staticmethod
    def MODEL(*files: str) -> str:
        if args.path:
            makedirs(args.path, exist_ok=True)
            return path.join(args.path, *files)
        else:
            return PATHS.OUT(*files)


class DATA:
    SAMPLE = args.sample
    RELOAD = args.reload
    FEATURE_DIM = 1280
    CAPTION_LEN = 20
    NUM_CAPTIONS = 5


class VOCAB:
    SIZE = 8096
    THRESHOLD = 3
    PADDING = 0
    START = 1
    END = 2
    UNKNOWN = 3


class MODEL:
    APPROACH = args.approach
    NAME = f"model-{APPROACH}"

    HIDDEN_DIM = 512
    EMBEDDING_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.1
    ATTENTION_HEADS = 8


class TRAIN:
    EPOCHS = args.epochs
    PATIENCE = args.patience
    BATCH_SIZE = args.batch
    LEARNING_RATE = 1e-3
