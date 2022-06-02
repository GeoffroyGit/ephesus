import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping


def train_ngap(df):
    '''
    This is a draft of a function to train a model to recognise the NGAP code
    df is expected to contain a column "X" with sentences
    df is expected to contain a column "y" with NGAP codes
    For exemple:
    df = pd.DataFrame({
        "X" : ["prise de sang",
              "test PCR covid-19",
              "Vaccin"
              "Pansement lourd"],
        "y" : ["PSG", "PC19", "", "PSTA"]
    })
    '''
    # list of NGAP codes the model is trained to detect
    list_of_NGAP_codes = \
        ['PSG','PV19','TAID19','PSTA','PC19','TAIC19','SC',\
         'PVAG','IM','ABFM','PLVA','NUR1','PSTL','ADM2','CSC']
    # preprocessing
    def remove_punctuation(text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')
        return text
    def remove_stop_words(tokens):
        stop_words = set(stopwords.words('french'))
        return [token for token in tokens if token not in stop_words]
    # preprocess X
    X = df[["X"]].copy()
    X["X"] = X["X"].apply(str.lower)
    X["X"] = X["X"].apply(lambda x: ''.join(word for word in x if not word.isdigit()))
    X["X"] = X["X"].apply(remove_punctuation)
    X["X"] = X["X"].apply(word_tokenize)
    X["X"] = X["X"].apply(remove_stop_words)
    X = X["X"]
    # preprocess y
    y = df[["y"]].copy()
    for code in list_of_NGAP_codes:
        y[code] = y["y"].apply(lambda x: x == code).astype(int)
    y["_unknown_"] = 0
    for code in list_of_NGAP_codes:
        y["_unknown_"] += y[code]
    y["_unknown_"] = (~ y["_unknown_"].astype(bool)).astype(int)
    y.drop(columns="y", inplace=True)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # create tokenizer and fit it on the train set
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    # apply the tokenization on the train set
    X_train_token = tokenizer.texts_to_sequences(X_train)
    # pad the sequences
    X_train_pad = pad_sequences(X_train_token, dtype='float32', padding='post', value=0, maxlen=16)
    # create the model
    vocab_size = len(tokenizer.word_index) + 1 # +1 for the 0 padding
    sequence_size = len(X_train_pad[0])
    model = Sequential()
    model.add(layers.Embedding(
        input_dim=vocab_size,
        input_length=sequence_size,
        output_dim=100,
        mask_zero=True
    ))
    model.add(layers.SimpleRNN(50))
    model.add(layers.Dense(32))
    model.add(layers.Dense(16, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    # configure early stopping and run the model
    es = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train_pad, y_train, batch_size=8, epochs=50, validation_split=0.3, callbacks=[es])

    # evaluate the model
    # apply the tokenization on the test set
    X_test_token = tokenizer.texts_to_sequences(X_test)
    # pad the sequences
    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post', value=0, maxlen=16)
    # use the trained model to make predictions
    y_pred = model.predict(X_test_pad)
    # evaluate the model
    score = model.evaluate(X_test_pad, y_pred)[1]
    return score

def predict_ngap(X_test):
    '''
    This is a draft of a function to use the model to predict the NGAP code
    '''
    # make predictions
    # apply the tokenization on the test set
    X_test_token = tokenizer.texts_to_sequences(X_test)
    # pad the sequences
    X_test_pad = pad_sequences(X_test_token, dtype='float32', padding='post', value=0, maxlen=16)
    # use the trained model to make predictions
    y_pred = model.predict(X_test_pad)
    # reshape y_pred to a more readable format
    df_pred = pd.DataFrame(y_pred)
    df_pred.columns = y.columns
    df_pred = pd.DataFrame(df_pred.stack()).reset_index()
    df_pred = df_pred.sort_values(0).groupby("level_0").last()
    df_pred_clean = pd.DataFrame(X_test).reset_index().drop(columns="index")
    df_pred_clean["NGAP"] = df_pred["level_1"]
    df_pred_clean["softmax"] = df_pred[0]
    return df_pred_clean
