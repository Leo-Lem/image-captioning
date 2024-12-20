from src.data import reformat, CaptionedImageDataset
from src.model import decoder
from src.train import train
from src.eval import test
from src.use import print_data, plot_metrics, plot_training, predictions, predict, display

from __param__ import FLAGS

try:
    reformat()

    training = CaptionedImageDataset("train")
    validation = CaptionedImageDataset("val")

    model = decoder()
    train(model, training, validation)

    testing = CaptionedImageDataset("test")
    test(model, testing)
except KeyboardInterrupt:
    print("Stopping training and testing…")

if FLAGS.PREDICT:
    display(FLAGS.PREDICT, predict(model, FLAGS.PREDICT))
else:
    print_data()
    plot_metrics(references={})
    plot_training()
    predictions(model, testing, n=9)
