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

from ephesus.sentence import load_model, return_label
from ephesus.data import get_data_json, get_data_targets_json


class TrainerNGAP():
    def __init__(self, list_of_NGAP_codes = \
                    ['PSG','PV19','TAID19','PSTA','PC19','TAIC19','SC',
                    'PVAG','IM','ABFM','PLVA','NUR1','PSTL','ADM2','CSC',
                    'ADM1', 'OBSD', 'ISCI']
                 ):
        '''
        store list_of_NGAP_codes as a class attribute
        '''
        self.list_of_NGAP_codes = list_of_NGAP_codes
        self.X = None # pandas DataFrame with one column named "X"
        self.y = None # pandas DataFrame with one column named "y"
        self.tokenizer = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.ycols = None

    def get_training_data(self, path):
        '''
        path is the path to the spaCy model that extracts treatments from sentences
        path is a string like "../models/model_v1/model-best"
        '''

        # load the training data for X
        # load the pre-trained spacy model
        model = load_model(path)
        # load the training data
        df = get_data_json()
        # run the model predictions on the data
        df["entities"] = df["translation"].apply(lambda x: return_label(x, model))
        # only keep the treatments
        def keep_treatment_only(entities):
            return [entity[0] for entity in entities if entity[1] == "Treatment"]
        df["treatments"] = df["entities"].apply(keep_treatment_only)
        df_split = pd.DataFrame(df["treatments"].to_list())
        df_split["filename"] = df["fichier"]
        df_stack = pd.DataFrame(df_split.set_index("filename").stack())\
            .reset_index().rename(columns={0 : "treatment"})

        # load the training data for y
        # load targets
        df_targets = get_data_targets_json()[["fichier", "NGAP_1"]]
        # clean filename for merge with df_stack
        def clean_filename(filename):
            return filename[:-17] if "translation.json" in filename else filename[:-16]
        df_stack["filename_cleaned"] = df_stack["filename"].apply(clean_filename)
        df_targets["filename_cleaned"] = df_targets["fichier"].apply(clean_filename)
        # merge with df_stack
        df = df_stack.merge(df_targets, how="left", on="filename_cleaned")
        df = df[["treatment", "NGAP_1"]].rename(columns={"treatment" : "X", "NGAP_1" : "y"})

        # store the training data into self.X and self.y
        self.X = df[["X"]]
        self.y = df[["y"]]

    def apply_tokenizer(self, X, maxlen=16):
        '''
        apply a previously fitted tokenizer
        and padd the sequences
        '''
        # apply the tokenization
        X_token = self.tokenizer.texts_to_sequences(X)
        # pad the sequences with zeros
        X_pad = pad_sequences(X_token, dtype='float32', padding='post', value=0, maxlen=maxlen)
        # return padded sequences
        return X_pad

    def train_ngap(self):
        '''
        This is a draft of a function to train a model to recognise the NGAP code
        '''
        # preprocessing
        def remove_punctuation(text):
            for punctuation in string.punctuation:
                text = text.replace(punctuation, ' ')
            return text
        def remove_stop_words(tokens):
            stop_words = set(stopwords.words('french'))
            return [token for token in tokens if token not in stop_words]
        # preprocess X
        X = self.X.copy()
        X["X"] = X["X"].apply(str.lower)
        X["X"] = X["X"].apply(lambda x: ''.join(word for word in x if not word.isdigit()))
        X["X"] = X["X"].apply(remove_punctuation)
        X["X"] = X["X"].apply(word_tokenize)
        X["X"] = X["X"].apply(remove_stop_words)
        X = X["X"]
        # preprocess y
        y = self.y.copy()
        for code in self.list_of_NGAP_codes:
            y[code] = y["y"].apply(lambda x: x == code).astype(int)
        y["_unknown_"] = 0
        for code in self.list_of_NGAP_codes:
            y["_unknown_"] += y[code]
        y["_unknown_"] = (~ y["_unknown_"].astype(bool)).astype(int)
        y.drop(columns="y", inplace=True)
        self.ycols = y.columns
        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33)
        # create tokenizer and fit it on the train set
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.X_train)
        # apply the tokenization on the train set and pad the train set
        X_train_pad = self.apply_tokenizer(self.X_train)
        # create the model
        vocab_size = len(self.tokenizer.word_index) + 1 # +1 for the 0 padding
        sequence_size = len(X_train_pad[0])
        nb_of_categories = len(self.list_of_NGAP_codes) + 1 # +1 for the _unknown_ category
        self.model = Sequential()
        self.model.add(layers.Embedding(
            input_dim=vocab_size,
            input_length=sequence_size,
            output_dim=100,
            mask_zero=True
        ))
        self.model.add(layers.SimpleRNN(50))
        self.model.add(layers.Dense(32))
        self.model.add(layers.Dense(nb_of_categories, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        # configure early stopping and run the model
        es = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(X_train_pad, self.y_train, batch_size=8, epochs=50, validation_split=0.3, callbacks=[es])

    def eval_ngap(self):
        '''
        evaluate the model
        '''
        # apply the tokenization on the test set and pad the test set
        X_test_pad = self.apply_tokenizer(self.X_test)
        # evaluate the model
        score = self.model.evaluate(X_test_pad, self.y_test)
        return score[1]

    def predict_ngap(self):
        '''
        use the model to make predictions
        '''
        # apply the tokenization on the test set and pad the test set
        X_test_pad = self.apply_tokenizer(self.X_test)
        # use the trained model to make predictions
        y_pred = self.model.predict(X_test_pad)
        # reshape y_pred to a more readable format
        df_pred = pd.DataFrame(y_pred)
        df_pred.columns = self.ycols
        df_pred = pd.DataFrame(df_pred.stack()).reset_index()
        df_pred = df_pred.sort_values(0).groupby("level_0").last()
        df_pred_clean = pd.DataFrame(self.X_test).reset_index().drop(columns="index")
        df_pred_clean["NGAP"] = df_pred["level_1"]
        df_pred_clean["softmax"] = df_pred[0]
        return df_pred_clean

if __name__ == '__main__':
    trainer = TrainerNGAP()
    print("Load data:")
    path = "../models/model_v2/model-best"
    trainer.get_training_data(path)
    print("Train:")
    trainer.train_ngap()
    print("Eval:")
    score = trainer.eval_ngap()
    print(f"model evaluation: {score}")
    print("Predict:")
    print(trainer.predict_ngap().head())
