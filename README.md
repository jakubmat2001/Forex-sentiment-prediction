# ForexSentimentAnalyzer

#NOTE: Please use python version 3.9+ (Make sure it is a default python interpretor as opposed to Conda)

Change Log:

v1.1.0
Added stratification when training bert model, previous versions showed lower evaluation accuracy and therefore this change will hopefully
improve the models overall perormance by distributing even set of lables of different class for both training and testing phases.

v1.2.0
GUI was added with gui elements changing character during the runtime
F1 scores added to further analyze faults in a labeled dataset

v1.3.0
Added more labels to sentence-sentiment dataset
Some visual changes made to the GUI
Cleaned up code to look more presentable, with more comments being added too