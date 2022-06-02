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


class TrainerNGAP():
    def __init__(self, X, y,
                 list_of_NGAP_codes = \
                    ['PSG','PV19','TAID19','PSTA','PC19','TAIC19','SC',\
                    'PVAG','IM','ABFM','PLVA','NUR1','PSTL','ADM2','CSC']
                 ):
        '''
        X: pandas DataFrame with one column named "X"
        y: pandas DataFrame with one column named "y"

        For exemple:
        df = pd.DataFrame({
            "X" : ["prise de sang",
                "test PCR covid-19",
                "Vaccin"
                "Pansement lourd"],
            "y" : ["PSG", "PC19", "", "PSTA"]
        })
        X = df[["X"]]
        y = df[["y"]]
        '''
        self.X = X
        self.y = y
        self.list_of_NGAP_codes = list_of_NGAP_codes
        self.tokenizer = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.ycols = None

    def apply_tokenizer(self, X):
        # apply the tokenization on the train set
        X_token = self.tokenizer.texts_to_sequences(X)
        # pad the sequences
        X_pad = pad_sequences(X_token, dtype='float32', padding='post', value=0, maxlen=16)
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
        self.model = Sequential()
        self.model.add(layers.Embedding(
            input_dim=vocab_size,
            input_length=sequence_size,
            output_dim=100,
            mask_zero=True
        ))
        self.model.add(layers.SimpleRNN(50))
        self.model.add(layers.Dense(32))
        self.model.add(layers.Dense(16, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        # configure early stopping and run the model
        es = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(X_train_pad, self.y_train, batch_size=8, epochs=50, validation_split=0.3, callbacks=[es])

    def eval_ngap(self):
        '''
        evaluate the model
        '''
        # train the model
        self.train_ngap()
        # apply the tokenization on the test set and pad the test set
        X_test_pad = self.apply_tokenizer(self.X_test)
        # use the trained model to make predictions
        y_pred = self.model.predict(X_test_pad)
        # evaluate the model
        score = self.model.evaluate(X_test_pad, y_pred)
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
    df = pd.DataFrame({
        "X" : ["prise de sang",
          "test PCR covid-19",
          "Vaccin",
          "Prise de sang",
          "Pansement lourd",
          "refaire les fils",
          "pansement d'amputation",
          "vaccins",
          "gros pensement"],
        "y" : ["PSG",
          "PC19",
          "PVAG",
          "PSG",
          "PSTA",
          "",
          "toto",
          "PVAG",
          "PSTA"]
    })
    X = df[["X"]]
    y = df[["y"]]
    trainer = TrainerNGAP(X, y)
    score = trainer.eval_ngap()
    print(f"model evaluation: {score}")
    print(trainer.predict_ngap())
