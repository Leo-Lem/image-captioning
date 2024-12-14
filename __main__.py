from src import reformat, load_data, vocabularize, devocabularize, preprocess, create_decoder, train, test

reformat()
vocab = vocabularize(load_data("full"))

training = preprocess(load_data("train"), vocab)
validation = preprocess(load_data("val"), vocab)
model = create_decoder()
train(model, training, validation)

testing = preprocess(load_data("test"), vocab)
results = test(model, testing, devocabularize(vocab))

print(results)
