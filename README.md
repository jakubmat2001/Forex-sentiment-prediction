# ForexSentimentAnalyzer

#NOTE: Please use Python version 3.9+ (Make sure you're using a default Python interpreter as opposed to Conda)

Change Log:

v1.1.0
Added stratification when training Bert model, previous versions showed lower evaluation accuracy, and therefore this change will hopefully
improve the model's overall performance by distributing an even set of labels of different classes for both the training and testing phases.

v1.2.0
GUI was added with GUI elements changing characters during the runtime
F1 scores were added to further analyze faults in a labeled dataset

v1.3.0
Added more labels to the sentence-sentiment dataset
Some visual changes made to the GUI
Cleaned up code to look more presentable, with more comments being added too
