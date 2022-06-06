import numpy as np
import pandas as pd
import string
import joblib
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
    def __init__(self,
                 list_of_NGAP_codes = \
                    ['PSG','PV19','TAID19','PSTA','PC19','TAIC19','SC',
                    'PVAG','IM','ABFM','PLVA','NUR1','PSTL','ADM2','CSC',
                    'ADM1', 'OBSD', 'ISCI'],
                 train_on_full_set = True,
                 path_spacy="",
                 path_ngap=""
                 ):
        '''
        Store list_of_NGAP_codes as a class attribute
        If train_on_full_set == False then
        - the train set is splitted in train, validation and test set
        - model evaluation is enabled
        - predictions are made on test set
        If train_on_full_set == True then
        - the whole train set is used for training and the model is saved
        - model evaluation is impossible so it's disabled
        - predictions load the model and make a single prediction
        '''
        self.list_of_NGAP_codes = list_of_NGAP_codes
        self.train_on_full_set = train_on_full_set
        self.path_spacy = path_spacy
        self.path_ngap = path_ngap
        self.X = None # pandas DataFrame with one column named "X"
        self.y = None # pandas DataFrame with one column named "y"
        self.tokenizer = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.ycols = None

    def get_training_data(self):
        '''
        path_spacy is the path to the spaCy model that extracts treatments from sentences
        path_spacy is a string like "../models/model_v1/model-best"
        '''

        # load the training data for X
        # load the pre-trained spacy model
        model = load_model(self.path_spacy)
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

    def remove_punctuation(self, text):
        '''
        preprocessing : remove punctuation
        '''
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')
        return text

    def remove_stop_words(self, tokens):
        '''
        preprocessing : remove stop words
        '''
        stop_words = set(stopwords.words('french'))
        return [token for token in tokens if token not in stop_words]

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
        train a model to recognise the NGAP code
        '''
        # preprocess X
        X = self.X.copy()
        X["X"] = X["X"].apply(str.lower)
        X["X"] = X["X"].apply(lambda x: ''.join(word for word in x if not word.isdigit()))
        X["X"] = X["X"].apply(self.remove_punctuation)
        X["X"] = X["X"].apply(word_tokenize)
        X["X"] = X["X"].apply(self.remove_stop_words)
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

        if self.train_on_full_set:
            # no train test split if we want to train on the whole data set
            self.X_train, self.X_test, self.y_train, self.y_test = X, None, y, None
        else:
            # train test split if we want to be able to evaluate our model
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

        if self.train_on_full_set:
            # save the model to a joblib file
            joblib.dump((self.model, self.tokenizer, self.ycols), self.path_ngap)


    def eval_ngap(self):
        '''
        evaluate the model
        '''

        if self.train_on_full_set:
            # cannot evaluate the model if we already used the whole data set for training
            return -1

        # apply the tokenization on the test set and pad the test set
        X_test_pad = self.apply_tokenizer(self.X_test)
        # evaluate the model
        score = self.model.evaluate(X_test_pad, self.y_test)
        return score[1]

    def predict_ngap(self, sentence=""):
        '''
        Use the model to make predictions
        sentence is one or multiple treatments
        sentence can be a string or a pandas dataframe with a column "X"
        '''
        if self.train_on_full_set:
            # load the model from the joblib file
            try:
                (self.model, self.tokenizer, self.ycols) = joblib.load(self.path_ngap)
            except FileNotFoundError:
                return pd.DataFrame()
            # preprocessing
            if type(sentence) == str:
                X = pd.DataFrame({"X" : [sentence]})
            else:
                X = sentence[["X"]].copy()
            X["X"] = X["X"].apply(str.lower)
            X["X"] = X["X"].apply(lambda x: ''.join(word for word in x if not word.isdigit()))
            X["X"] = X["X"].apply(self.remove_punctuation)
            X["X"] = X["X"].apply(word_tokenize)
            X["X"] = X["X"].apply(self.remove_stop_words)
            X = X["X"]
            # apply the tokenization and padding
            X_pad = self.apply_tokenizer(X)
            # use the trained model to make predictions
            y_pred = self.model.predict(X_pad)
        else:
            # apply the tokenization on the test set and pad the test set
            X_test_pad = self.apply_tokenizer(self.X_test)
            # use the trained model to make predictions
            y_pred = self.model.predict(X_test_pad)

        # reshape y_pred to a more readable format
        df_pred = pd.DataFrame(y_pred)
        df_pred.columns = self.ycols
        df_pred = pd.DataFrame(df_pred.stack()).reset_index()
        df_pred = df_pred.sort_values(0).groupby("level_0").last()

        if self.train_on_full_set:
            df_pred_clean = pd.DataFrame(X).reset_index().drop(columns="index")
        else:
            df_pred_clean = pd.DataFrame(self.X_test).reset_index().drop(columns="index")

        df_pred_clean["NGAP"] = df_pred["level_1"]
        df_pred_clean["softmax"] = df_pred[0]
        return df_pred_clean

class TrainerLocation():
    def __init__(self,
                 train_on_full_set = True,
                 path_spacy="",
                 path_loc=""
                 ):
        '''
        If train_on_full_set == False then
        - the train set is splitted in train, validation and test set
        - model evaluation is enabled
        - predictions are made on test set
        If train_on_full_set == True then
        - the whole train set is used for training and the model is saved
        - model evaluation is impossible so it's disabled
        - predictions load the model and make a single prediction
        '''
        self.train_on_full_set = train_on_full_set
        self.path_spacy = path_spacy
        self.path_ngap = path_ngap
        self.X = None # pandas DataFrame with one column named "X"
        self.y = None # pandas DataFrame with one column named "y"
        self.tokenizer = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def get_training_data(self):
        '''
        path_spacy is the path to the spaCy model that extracts locations from sentences
        path_spacy is a string like "../models/model_v1/model-best"
        '''

        # load the training data for X
        # load the pre-trained spacy model
        model = load_model(self.path_spacy)
        # load the training data
        df = get_data_json()
        # run the model predictions on the data
        df["entities"] = df["translation"].apply(lambda x: return_label(x, model))
        # only keep the locations
        def keep_location_only(entities):
            return [entity[0] for entity in entities if entity[1] == "Location"]
        df["locations"] = df["entities"].apply(keep_location_only)
        df_split = pd.DataFrame(df["locations"].to_list())
        df_split["filename"] = df["fichier"]
        df_stack = pd.DataFrame(df_split.set_index("filename").stack())\
            .reset_index().rename(columns={0 : "location"})

        # load the training data for y
        # load targets
        df_targets = get_data_targets_json()[["fichier", "CareLocation_1"]]
        # clean filename for merge with df_stack
        def clean_filename(filename):
            return filename[:-17] if "translation.json" in filename else filename[:-16]
        df_stack["filename_cleaned"] = df_stack["filename"].apply(clean_filename)
        df_targets["filename_cleaned"] = df_targets["fichier"].apply(clean_filename)
        # merge with df_stack
        df = df_stack.merge(df_targets, how="left", on="filename_cleaned")
        df = df[["location", "CareLocation_1"]].rename(columns={"location" : "X", "CareLocation_1" : "y"})

        # store the training data into self.X and self.y
        self.X = df[["X"]]
        self.y = df[["y"]]

    def remove_punctuation(self, text):
        '''
        preprocessing : remove punctuation
        '''
        for punctuation in string.punctuation:
            text = text.replace(punctuation, ' ')
        return text

    def remove_stop_words(self, tokens):
        '''
        preprocessing : remove stop words
        '''
        stop_words = set(stopwords.words('french'))
        return [token for token in tokens if token not in stop_words]

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

    def train_location(self):
        '''
        train a model to recognise the location
        '''
        # preprocess X
        X = self.X.copy()
        X["X"] = X["X"].apply(str.lower)
        X["X"] = X["X"].apply(lambda x: ''.join(word for word in x if not word.isdigit()))
        X["X"] = X["X"].apply(self.remove_punctuation)
        X["X"] = X["X"].apply(word_tokenize)
        X["X"] = X["X"].apply(self.remove_stop_words)
        X = X["X"]
        # preprocess y
        y = self.y.copy()
        def cabinet_or_domicile(text):
            for loc in ["cabinet", "location"]:
                if text.lower() == loc:
                    return loc
            return None
        y["y"] = y["y"].apply(cabinet_or_domicile)

        if self.train_on_full_set:
            # no train test split if we want to train on the whole data set
            self.X_train, self.X_test, self.y_train, self.y_test = X, None, y, None
        else:
            # train test split if we want to be able to evaluate our model
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
        self.model.add(layers.Dense(1, activation="sigmoid"))
        self.model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
        # configure early stopping and run the model
        es = EarlyStopping(patience=5, restore_best_weights=True)
        self.model.fit(X_train_pad, self.y_train, batch_size=8, epochs=50, validation_split=0.3, callbacks=[es])

        if self.train_on_full_set:
            # save the model to a joblib file
            joblib.dump((self.model, self.tokenizer, self.ycols), self.path_ngap)


    def eval_ngap(self):
        '''
        evaluate the model
        '''

        if self.train_on_full_set:
            # cannot evaluate the model if we already used the whole data set for training
            return -1

        # apply the tokenization on the test set and pad the test set
        X_test_pad = self.apply_tokenizer(self.X_test)
        # evaluate the model
        score = self.model.evaluate(X_test_pad, self.y_test)
        return score[1]

    def predict_ngap(self, sentence=""):
        '''
        Use the model to make predictions
        sentence is one or multiple treatments
        sentence can be a string or a pandas dataframe with a column "X"
        '''
        if self.train_on_full_set:
            # load the model from the joblib file
            try:
                (self.model, self.tokenizer, self.ycols) = joblib.load(self.path_ngap)
            except FileNotFoundError:
                return pd.DataFrame()
            # preprocessing
            if type(sentence) == str:
                X = pd.DataFrame({"X" : [sentence]})
            else:
                X = sentence[["X"]].copy()
            X["X"] = X["X"].apply(str.lower)
            X["X"] = X["X"].apply(lambda x: ''.join(word for word in x if not word.isdigit()))
            X["X"] = X["X"].apply(self.remove_punctuation)
            X["X"] = X["X"].apply(word_tokenize)
            X["X"] = X["X"].apply(self.remove_stop_words)
            X = X["X"]
            # apply the tokenization and padding
            X_pad = self.apply_tokenizer(X)
            # use the trained model to make predictions
            y_pred = self.model.predict(X_pad)
        else:
            # apply the tokenization on the test set and pad the test set
            X_test_pad = self.apply_tokenizer(self.X_test)
            # use the trained model to make predictions
            y_pred = self.model.predict(X_test_pad)

        # reshape y_pred to a more readable format
        df_pred = pd.DataFrame(y_pred)
        df_pred.columns = self.ycols
        df_pred = pd.DataFrame(df_pred.stack()).reset_index()
        df_pred = df_pred.sort_values(0).groupby("level_0").last()

        if self.train_on_full_set:
            df_pred_clean = pd.DataFrame(X).reset_index().drop(columns="index")
        else:
            df_pred_clean = pd.DataFrame(self.X_test).reset_index().drop(columns="index")

        df_pred_clean["NGAP"] = df_pred["level_1"]
        df_pred_clean["softmax"] = df_pred[0]
        return df_pred_clean

if __name__ == '__main__':

    path_spacy = "../models/model_v2/model-best"
    path_ngap = "../model_ngap.joblib"

    test_evaluation = True
    test_predict = True
    test_load_model = True

    if test_evaluation:
        # create trainer for evaluation
        print("Create trainer for evaluation:")
        trainer = TrainerNGAP(train_on_full_set = False, path_spacy=path_spacy)
        print("Load data:")
        trainer.get_training_data()
        print("Train:")
        trainer.train_ngap()
        print("Eval:")
        score = trainer.eval_ngap()
        print(f"model evaluation: {score}")
        print("Predict:")
        print(trainer.predict_ngap())
    if test_predict:
        # create trainer for precitions
        print("Create trainer for predictions:")
        trainer = TrainerNGAP(train_on_full_set = True, path_spacy=path_spacy, path_ngap=path_ngap)
        print("Load data:")
        trainer.get_training_data()
        print("Train:")
        trainer.train_ngap()
        print("Predict:")
        sentence = "Prise de sang"
        print(trainer.predict_ngap(sentence=sentence))
    if test_predict or test_load_model:
        # create a new trainer for precitions
        print("Create new trainer for predictions:")
        trainer = TrainerNGAP(train_on_full_set = True, path_spacy=path_spacy, path_ngap=path_ngap)
        print("Predict one treatment:")
        sentence = "Grand pansement"
        print(trainer.predict_ngap(sentence=sentence))
        print("Predict multiple treatments at once:")
        sentence = pd.DataFrame({"X" : ["Grand pansement",
                                        "Prise de sang",
                                        "Vaccin Covid19",
                                        "Test PCR"]})
        print(trainer.predict_ngap(sentence=sentence))
