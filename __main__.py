from src import reformat, load_data, vocabularize, preprocess, Decoder, train

reformat()
vocab = vocabularize(load_data("full"))

training = preprocess(load_data("train"), vocab)
validation = preprocess(load_data("val"), vocab)

model = Decoder.create()
train(model, training, validation)
