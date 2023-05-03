from spacy.pipeline import EntityRecognizer
from training_data import data
from spacy.training.example import Example
import random
import spacy

#get instances of data from a list and shuffle it, then select all for training
random.shuffle(data)
train_data = data[:]

#create a blank spacy model with a new NER
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

#add "ECONOMIC_INDICATOR", "CURRENCY" as entities in to the model
ner.add_label("ECONOMIC_INDICATOR")
ner.add_label("CURRENCY")

#convert the training data into spacy's "Example" format
examples = []
for text, annotations in train_data:
    examples.append(Example.from_dict(nlp.make_doc(text.lower()), annotations))
    
print(examples)

#train the model
nlp.initialize()
optimizer = nlp.create_optimizer()
n_iter = 10
for i in range(n_iter):
    random.shuffle(examples)
    for example in examples:
        nlp.update([example], sgd=optimizer)

print("Training complete!")

nlp.to_disk("spacy_forex_entity_model")