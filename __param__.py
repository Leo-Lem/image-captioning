from torch import cuda
from argparse import ArgumentParser
from os import path

parser = ArgumentParser(
    description="Train an image captioning pipeline with different decoder architectures")
parser.add_argument("approach", type=str,
                    help="lstm, gru, transformer")

parser.add_argument("-p", "--path", type=str, default="",
                    help="path to load/store the trained model")

parser.add_argument("--epochs", type=int, default=10,
                    help="number of epochs to train the model")
parser.add_argument("--stop", type=int, default=3,
                    help="number of epochs without improvement to stop training")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="learning rate for training")
parser.add_argument("--batch", type=int, default=1024,
                    help="batch size for training")

parser.add_argument("-d", "--debug", action="store_true",
                    help="run in debug mode")
parser.add_argument("--reformat", action="store_true",
                    help="reformat the data")
parser.add_argument("--sample", action="store_true",
                    help="use sample data instead of full data")
parser.add_argument("--eval", action="store_true",
                    help="skip training the model")
args = parser.parse_args()


class PATHS:
    OUT = path.join(path.dirname(path.dirname(__file__)), ".out")
    MODEL = args.path if args.path else OUT
    RES = path.join(path.dirname(__file__), "res")


APPROACH = args.approach


class TRAIN:
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch
    STOP_EARLY_AFTER = args.stop
    LEARNING_RATE = args.lr


class DATA:
    SAMPLE = args.sample
    REFORMAT = args.reformat


class FLAGS:
    DEBUG = args.debug
    EVAL = args.eval
    GPU = cuda.is_available()
