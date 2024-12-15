from src.data import reformat, load_data, vocabularize, devocabularize, preprocess
from src.model import create_decoder
from src.train import train
from src.eval import test
from src.use import print_data, plot_metrics, plot_training, predictions

try:
    reformat()
    vocab = vocabularize(load_data("full"))
    reversed_vocab = devocabularize(vocab)

    training = preprocess(load_data("train"), vocab, shuffle=True)
    validation = preprocess(load_data("val"), vocab)
    model = create_decoder()
    train(model, training, validation)

    testing = preprocess(load_data("test"), vocab)
    results = test(model, testing, reversed_vocab)
except KeyboardInterrupt:
    print("Stopping training and testing…")
finally:
    print_data()
    plot_metrics(references={"BLEU": 0.0, "METEOR": 0.0, "NIST": 0.0})
    plot_training()

    output = preprocess(load_data("test"), vocab, batch=False)
    predictions(model, output, reversed_vocab, n=9)
