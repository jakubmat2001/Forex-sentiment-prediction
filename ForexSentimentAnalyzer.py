#Application Version: 1.3.0v
#By: Jakub Matusik

import pandas as pd
from nltk.corpus import stopwords
import spacy
import nltk
import re 
import correlete_currency_data

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tkinter as tk
from tkinter.font import Font
from tkinter import ttk

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------- GUI Elements ------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------


root = tk.Tk()
root.configure(bg="white")
root.winfo_toplevel().title("Forex Sentiment Analyzer")
tk.Tk.resizable(root, width=False, height=False)
frame = tk.Frame(root)

frame.pack(side="top", fill="both", expand=True)
frame.grid_rowconfigure(0, weight=1)
frame.grid_columnconfigure(0, weight=1)

App_font = ("Sans-Serif", 13)
Text_box_font = ("Sans-Serif", 11)
bold_font = Font(family="Sans-Serif", size=13, weight="bold")

main_body = tk.Frame(frame, bg="#FDFDFD")
main_body.grid(row=0, column=0, sticky="we")

element_body = tk.Frame(frame, bg="#FFFFFF")
element_body.grid(row=1, column=0, sticky="we")

text_box_body = tk.Frame(frame, bg="#FDFDFD")
text_box_body.grid(row=2, column=0, sticky="we")

banner_lbl = tk.Label(main_body,text="Economic Data" ,width=20, height=2, bg="#BDBABA",borderwidth=2, highlightthickness=1,  highlightbackground="white smoke", font=bold_font)
banner_lbl.grid(row=1, column=1, sticky="we")

left_banner_sapce = tk.Label(main_body,text="" ,width=70, height=3, bg="#da8383")
left_banner_sapce.grid(row=1, column=0, sticky="we")

right_banner_space = tk.Label(main_body, text="" ,width=70, height=3, bg="#da8383")
right_banner_space.grid(row=1, column=2, sticky="we")

bottom_banner_space = tk.Label(frame, text="" ,width=70, height=3, bg="#da8383")
bottom_banner_space.grid(row=3, column=0, sticky="we")

currency_pair_combo = ttk.Combobox(element_body, width=20, state="readonly")
currency_pair_combo['values'] = ("AUD/USD")
currency_pair_combo.current(0)
currency_pair_combo.grid(row=0, column=0, padx=(150,0), pady=(50,0))

analyze_button = ttk.Button(element_body, text="Analyze", command=lambda:predict_overall_sentiment())
analyze_button.grid(row=0, column=2, padx=(55,0), pady=(50,0))

outlook_label = tk.Label(element_body, text="Pair Outlook: " , bg="#d9d9d9")
outlook_label.grid(row=0, column=3, padx=(300,0), pady=(50,0))

sentiment_label = tk.Label(element_body, text=" Sentiment: None", bg="#da8383")
sentiment_label.grid(row=0, column=4, padx=(25,0), pady=(50,0))

text_box = tk.Text(text_box_body, width=50, height=20, font=Text_box_font ,  highlightthickness=1, highlightbackground="#d9d9d9", bg="#d9d9d9", padx=3)
text_box.grid(row=0, column=0, padx=(150,0), pady=(20,50))

text_box_uneditetable = tk.Text(text_box_body, width=40, height=20, font=Text_box_font ,fg="black", highlightthickness=1, highlightbackground="#d9d9d9", state="disabled", bg="#d9d9d9" ,padx=3)
text_box_uneditetable.grid(row=0, column=1, padx=(150,0), pady=(20, 50))


#------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------- Text Preprocessing  -----------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------

economic_indicators_df = pd.read_csv("forex_related_datasets/extracted_news_dataset.csv", encoding = "ISO-8859-1")

nlp = spacy.load("spacy_forex_entity_model")
nlp_lemma = spacy.load("en_core_web_sm")

#clean the text from any non letter/numberical characters
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w.]', ' ', text) 
    text = re.sub(r'\s+', ' ', text)
    return text

#get rid of fullstops
def clean_fullstops(text):
    text = re.sub(r'[^\w\s]', ' ', text)
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
        if token not in stop_words or token == "us" or token == "US":
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

economic_indicators_df['Data'] = economic_indicators_df['Data'].apply(clean_text)
economic_indicators_df['Currency'] = economic_indicators_df['Currency'].apply(clean_text)

#------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------- Predict Text ------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------

#load our trained BERT model
model = BertForSequenceClassification.from_pretrained("bert/bert_fx_model")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#nlp = spacy.load_model('general_model')

#extract all entities found in our sentence
def extract_entities(text):
    entities = []
    doc = nlp(text.lower())
    for ent in doc.ents:
        if ent.label_ == "ECONOMIC_INDICATOR":
            entities.append(ent.text)
        elif ent.label_ == "CURRENCY":
            entities.append(ent.text)
    return entities

#check if the found indicator name is found within our currency dictionary
def find_indicator(entity, indicators_syn):
    for indicator, synonyms in indicators_syn.items():
        if entity in synonyms:
            return indicator
    return None

#check if the found currency name is found within our currency dictionary
def find_currency(entity, indicators_syn):
    for currency, synonyms in indicators_syn.items():
        if entity in synonyms:
            return currency
    return None

#get rid-off an indicator for a spcified currency, this prevents re-use for the same economic event occuring in the same article text
def validate_indicator(currency, indicator, aud_indicators, usd_indicators):
    selected_currency = currency.lower()
    selected_indicator = indicator.lower()
    
    if selected_indicator in aud_indicators and selected_currency == "australian dollar":
        aud_indicators.remove(selected_indicator)
        return True
    elif selected_indicator in usd_indicators and selected_currency == "united states dollar":
        usd_indicators.remove(selected_indicator)
        return True
    else:
        print("not valid\n")
        print(str(currency) + " " + str(indicator))
        return False


def predict_paragraph(text, economic_indicator_lookup, currency_lookup, aud_indicators, usd_indicators):
    pips = 0
    sentences = nltk.sent_tokenize(text)
    device = torch.device("cpu")
    last_currency = None  
    try:
        for sentence in sentences:
            print(sentence)
            #for every sentence we first find indicators from an unprocessed sentence, then knowing all the indicators we clean-up text
            entities = extract_entities(sentence)
            sentence = clean_fullstops(sentence)
            preprocessed_sentence = preprocess_text(sentence)
            
            tokenized_text = tokenizer.encode_plus(preprocessed_sentence, add_special_tokens=True, max_length=512, padding='max_length', return_attention_mask=True, return_tensors='pt')
            input_ids = tokenized_text['input_ids'].to(device)
            attention_masks = tokenized_text['attention_mask'].to(device)
            outputs = model(input_ids, attention_masks)
            _, predicted_label = torch.max(outputs[0], 1)
            sentiment = predicted_label.item()

            print("---------------------------")
            print(str(preprocessed_sentence))
            print("sentence sentiment: " + str(sentiment))
            print("entities found: " + str(entities))
            #check that the entities were found in a sentence
            #if they were found, check which ones an and the their type
            if len(entities) != 0:
                for entity in entities:
                    indicator = find_indicator(entity, economic_indicator_lookup)
                    if indicator != None:
                        break
                currency = None
                for entity in entities:
                    currency = find_currency(entity, currency_lookup)
                    if currency != None:
                        last_currency = currency 
                        break
                if currency is None:
                    currency = last_currency
                print("(Currency: " + str(currency) + ") (Indicator: " + str(indicator) + ")\n---------------------------" + "\n")
                if indicator is not None and currency is not None and indicator in economic_indicators_df['Data'].values:
                    target_currency = find_currency(currency, currency_lookup)
                    #print("target currency " + str(target_currency) + " actual currency " + str(currency))
                    if currency in target_currency and sentiment != 1:
                        check_indicator_validity = validate_indicator(currency, indicator ,aud_indicators, usd_indicators)
                        if check_indicator_validity != False:
    
                            pip_move = economic_indicators_df[(economic_indicators_df['Data'] == indicator) & (economic_indicators_df['Currency'] == currency)]['average_event_move'].values[0]
                            if sentiment == 2 and currency == "australian dollar":
                                pips += pip_move
                            elif sentiment == 2 and currency == "united states dollar":
                                pips -= pip_move
                            elif sentiment == 0 and currency == "australian dollar":
                                pips -= pip_move
                            elif sentiment == 0 and currency == "united states dollar":
                                pips += pip_move

                            if sentiment == 2:
                                sentence_sentiment = "positive"
                            elif sentiment == 1:
                                sentence_sentiment = "neutral"
                            else:
                                sentence_sentiment = "negative"

                            text_box_uneditetable.config(state="normal")
                            text_box_uneditetable.insert("end","Sentence: " + str(sentence) + "\nSentence Sentiment: "+ str(sentence_sentiment) +  "\nCurrency: "
                                                        + str(currency)  + "\nIndicator: " + str(indicator) + "\nPips: " + str(pip_move) + "\n----------------------------------\n")
                            
                            
                            print(str(pips) + "\n")
    except Exception:
        pass
 
    text_box_uneditetable.insert("end", "Overall expected pip-move: " + str(round(pips, 5)))
    text_box_uneditetable.config(state="disabled")
    return pips

def predict_overall_sentiment():
    text = text_box.get('1.0', 'end')
    print(text)
    
    overall_pips = 0
    paragraph_pips = predict_paragraph(text, correlete_currency_data.economic_indicator_lookup,correlete_currency_data.currency_lookup, correlete_currency_data.aud_indicators, correlete_currency_data.usd_indicators)
    overall_pips += paragraph_pips
    overall_pips = round(overall_pips, 5)

    #determine the overall sentiment based on the total pips
    if overall_pips == 0:
        overall_sentiment = "neutral"
    elif overall_pips > 0:
        overall_sentiment = "positive"
    elif overall_pips < 0:
        overall_sentiment = "negative"

    outlook_label.config(text="Pair Outlook " + str(currency_pair_combo.get()))
    
    sentiment_label.config(text="Sentiment: " + str(overall_sentiment))
    print(f"Overall Sentiment For Paragraphs: {overall_sentiment.upper()} (Total pip-move for {currency_pair_combo.get()}: {overall_pips})")
root.mainloop()