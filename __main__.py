from src import reformat, preprocess, train

reformat()

train = preprocess("train")
val = preprocess("val")
