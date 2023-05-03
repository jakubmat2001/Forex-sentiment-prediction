#Application Version: 1.3.0v
#By: Jakub Matusik

import time
import pandas as pd
import nltk
import spacy
import torch
import re

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

from torch.cuda.amp import autocast, GradScaler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

scaler = GradScaler()

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------- Pre-Loading/Downloading Necessary Components For Our Imports --------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------
start_time = time.time()
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')

#loading sentenced and labeled with corresponding sentiment dataset, then shuffle the rows
labeled_sentences_df = pd.read_csv("forex_related_datasets/sentenced-sentiment.csv", encoding = "utf-8-sig")
labeled_sentences_df = labeled_sentences_df.sample(frac=1).reset_index(drop=True)

#laoding requiered spacy model for correct NER labeling, loading a basic one too for lemmatization
nlp = spacy.load("spacy_forex_entity_model")
nlp_lemma = spacy.load("en_core_web_sm")

num_labels = 3
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------ Pre-Processing Steps --------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------

#clean the text from any non letter/numberical characters
def clean_text(text):
    text = text.lower() 
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text) 
    return text

#tokenize the text 
def tokenize_text(text):
    tokens = nltk.word_tokenize(text) 
    return tokens

#remove any stopwords appearing in our text, then prevent the word "us" from being removed
def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english')) 
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)
    return filtered_tokens

#lemmatize our tokens using spacy lemmatizer
def lemmatize_tokens(tokens):
    lemmatized_tokens = []
    for token in tokens:
        doc = nlp_lemma(token)
        lemma = doc[0].lemma_
        lemmatized_tokens.append(lemma)
    return lemmatized_tokens

#apply all the processing to our text in one function call,then join the tokens back into a sentnece
def preprocess_text(text):
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stop_words(tokens)
    tokens = lemmatize_tokens(tokens)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

labeled_sentences_df["sentences"] = labeled_sentences_df["sentences"].apply(preprocess_text)
label_counts = labeled_sentences_df['sentiment'].value_counts()

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- Machine Learning Stage -----------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------
unique_labels = labeled_sentences_df['sentiment'].unique()

#set max lenght limit of accpeted sentences
tokens = []
for text in labeled_sentences_df['sentences']:
    tokenized_text = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, padding='max_length', return_attention_mask=True, return_tensors='pt')
    tokens.append(tokenized_text)

#set labels numerical labels from our dataset
labels = []
for label in labeled_sentences_df['sentiment']:
    if label == 'pos':
        labels.append(2)
    elif label == 'neu':
        labels.append(1)
    else:
        labels.append(0)

input_ids = torch.cat([tokenized_text['input_ids'] for tokenized_text in tokens], dim=0)
attention_masks = torch.cat([tokenized_text['attention_mask'] for tokenized_text in tokens], dim=0)
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels)

#getting 80% of the dataset for training, added stratification to the program
#this will allow or ensure that the distribution of labels is approximately the same in both sets
train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(
    input_ids, labels, attention_masks, test_size=0.2, random_state=42, stratify=labels
)

train_inputs = train_inputs.clone().detach()
test_inputs = test_inputs.clone().detach()
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)
train_masks = train_masks.clone().detach()
test_masks = test_masks.clone().detach()

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

#if training with cuda with prominent machine specs set batch size between 4
#however if you don't mind waiting, select batsize between 24-32 for the best accuracy 
batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#set learning rate for training the model
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss()

#attepmting to use cuda, if failed we're going to use cpu for the usage (Note, will take anywhere from 2-3h to train)
#depending on CPU, batchsize and epoch count.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

#change the number of epoch iterations as needed, lower batch size requires more epoch iterations and vice versa to not overfit the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_dataloader:
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids, attention_masks).logits
            #check if the we're at the last batch, if last batch is smaller then the rest then handle it speparatly
            check_is_last_batch = (batch == len(train_dataloader) - 1)
            if check_is_last_batch:
                outputs = outputs.unsqueeze(0)
            else:
                outputs = outputs.squeeze()

            loss = criterion(outputs.squeeze(), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    #free up some memory from the gpu for training usage
    torch.cuda.empty_cache()
    print("Epoch: " + str(epoch+1) + " Training Loss: " + str(train_loss/len(train_dataloader)))

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------- Evaluating Model Performance -----------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------

model.eval()
test_loss = 0
num_correct = 0
num_total = 0
predictions = []
true_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids, attention_masks).logits
        #check if the we're at the last batch, if last batch is smaller then the rest then handle it speparatly again
        check_is_last_batch = (batch == len(train_dataloader) - 1)
        if check_is_last_batch:
            outputs = outputs.unsqueeze(0)
        else:
            outputs = outputs.squeeze()
        loss = criterion(outputs.squeeze(), labels)
        test_loss += loss.item()
        _, predicted_labels = torch.max(outputs, 1)
        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        num_correct += (predicted_labels == labels).sum().item()
        num_total += labels.size(0)

accuracy = num_correct / num_total
f1 = f1_score(true_labels, predictions, average='weighted')

print(f"testing accuracy: {accuracy}")
print(f"F1 score: {f1}")

#calculate TP, FP, TN, FN from the model
confusion = confusion_matrix(true_labels, predictions)

#selecting labels and their position in the confusion matrix
for i, label in enumerate(unique_labels):
    tp = confusion[i, i]
    fp = confusion[:, i].sum() - tp
    fn = confusion[i, :].sum() - tp
    tn = confusion.sum() - (tp + fp + fn)

    print(f"----------------------\nLabel: {label}")
    print(f"True Positives: {tp}" + " ✓")
    print(f"True Negatives: {tn}" + " ✓\n")

    print(f"False Positives: {fp}" + " X")
    print(f"False Negatives: {fn}" + " X")
    print(f"----------------------\n")

#if model not does not exists, save it and check how long it took to train/evaluate
try:
    model_path = "bert/bert_fx_model"
    model.save_pretrained(model_path)
    elapsed_time = (time.time() - start_time) / 60
    print(f"Program took {elapsed_time} min to execute.")
except Exception:
    print("Delete a previous model")



