from unicodedata import digit
import pandas as pd
import string
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from ephesus.sentence import load_model, return_label
from ephesus.data import get_data_json, get_data_targets_json


class Date():
    def __init__(self, path_spacy=""):
        '''
        store path_spacy in self
        '''
        self.path_spacy = path_spacy
        self.df = None
        self.df_exists = False

    def get_data(self):
        '''
        path_spacy is the path to the spaCy model that extracts dates from sentences
        path_spacy is a string like "../models/model_v1/model-best"
        '''

        # load the pre-trained spacy model
        model = load_model(self.path_spacy)
        # load the training data
        df = get_data_json()
        # run the model predictions on the data
        df["entities"] = df["translation"].apply(lambda x: return_label(x, model))
        # only keep the dates
        def keep_date_only(entities):
            return [entity[0] for entity in entities if entity[1] == "Date"]
        df["dates"] = df["entities"].apply(keep_date_only)
        df_split = pd.DataFrame(df["dates"].to_list())
        df_split["filename"] = df["fichier"]
        df_stack = pd.DataFrame(df_split.set_index("filename").stack())\
            .reset_index().rename(columns={0 : "date"})

        # store the dates in self.df
        self.df = df_stack[["filename", "date"]]
        self.df_exists = True

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

    def force_digits_only(self, tokens):
        '''
        preprocessing : force tokens to be digits only or char only
        '''
        tokens_clean = []
        for token in tokens:
            # check if the token is made of a majority of digits or not
            if sum([letter.isdigit() for letter in token]) >= (len(token) / 2):
                # keep digits only
                token_clean = "".join([letter for letter in token if letter.isdigit()])
            else:
                # remove digits
                token_clean = "".join([letter for letter in token if not letter.isdigit()])
            tokens_clean.append(token_clean)
        return tokens_clean

    def get_n_digit_token(self, tokens, n):
        '''
        return the nth digit token
        if not found, returns "99"
        '''
        count = 0
        for token in tokens:
            if token.isdigit():
                count += 1
                if count >= n:
                    return token
        return "99"

    def get_3_digit_token(self, tokens):
        '''
        return the first 3 digit token as numerics
        if not found, returns "99"
        if one of them is obviously the year, put it at the end
        if one of them is obviously the day, put it at beginning
        '''
        digit_tokens = ["-1", "-1", "-1"]
        i = 0
        for token in tokens:
            if token.isdigit():
                digit_tokens[i] = token
                if i >= 2:
                    break
                i += 1
        # put year at the end
        if pd.to_numeric(digit_tokens[0]) > 31:
            digit_tokens[0], digit_tokens[1], digit_tokens[2] =  \
                digit_tokens[1], digit_tokens[2], digit_tokens[0]
        if pd.to_numeric(digit_tokens[1]) > 31:
            digit_tokens[1], digit_tokens[2] =  \
                digit_tokens[2], digit_tokens[1]
        # put day at the beginning
        if pd.to_numeric(digit_tokens[2]) > 12 and pd.to_numeric(digit_tokens[2]) <= 31:
            digit_tokens[0], digit_tokens[1], digit_tokens[2] =  \
                digit_tokens[2], digit_tokens[0], digit_tokens[1]
        if pd.to_numeric(digit_tokens[1]) > 12 and pd.to_numeric(digit_tokens[1]) <= 31:
            digit_tokens[0], digit_tokens[1] =  \
                digit_tokens[1], digit_tokens[0]
        # replace "-1" with "99"
        return ["99" if token == "-1" else token for token in digit_tokens]

    def text_to_digit(self, tokens):
        '''
        change from numbers in plain text to numbers in digits
        only for one-word numbers since we assume that the speach to text already use digits for bigger numbers
        '''
        dic = {
            "zero" : "0",
            "premier" : "1",
            "un" : "1",
            "deux" : "2",
            "trois" : "3",
            "quatre" : "4",
            "cinq" : "5",
            "six" : "6",
            "sept" : "7",
            "huit" :  "8",
            "neuf" : "9",
            "dix" : "10",
            "onze" : "11",
            "douze" : "12",
            "treize" : "13",
            "quatorze" : "14",
            "quinze" : "15",
            "seize" : "16",
        }
        return [dic[token] if token in dic else token for token in tokens]

    def month_to_digit(self, tokens):
        '''
        look for month in plain text
        and return month in digits
        '''
        dic = {
            "janvier" : "1",
            "février" : "2",
            "mars" : "3",
            "avril" : "4",
            "mai" : "5",
            "juin" : "6",
            "juillet" : "7",
            "août" :  "8",
            "septembre" : "9",
            "octobre" : "10",
            "novembre" : "11",
            "décembre" : "12"
        }
        for token in tokens:
            if token in dic:
                return dic[token]
        return "99"

    def day_of_week(self, tokens):
        '''
        look for day of week in plain text
        like "lundi", "mardi", etc
        '''
        dic = {
            "lundi" : 0,
            "mardi" : 1,
            "mercredi" : 2,
            "jeudi" : 3,
            "vendredi" : 4,
            "samedi" : 5,
            "dimanche" : 6
        }
        for day in dic.keys():
            if day in tokens:
                return dic[day]
        return 99

    def day_from_today(self, tokens):
        '''
        look for days from today in plain text
        like "hier" or "demain"
        '''
        dic = {
            "hier" : -1,
            "demain" : 1,
            "aujourd" : 0,
            "jour" : 0,
            "matin" : 0,      # in the context of a date, "ce matin" is today
            "après midi" : 0, # same, "cet après midi" is today
            "midi" : 0,       # same, "ce midi" is today
            "soir" : 0        # same, "ce soir" is today
        }
        for day in dic.keys():
            if sum([word in tokens for word in day.split()]) == len(day.split()):
                return dic[day]
        return 99

    def transform_data(self, sentence=""):
        '''
        transform groups of words into dates
        sentence is one or multiple dates
        sentence can be a string or a pandas dataframe with a column "X"
        '''

        if len(sentence) == 0:
            # if sentence is not provided, make transformations from self.df
            if not self.df_exists:
                return pd.DataFrame()
            df = self.df[["date"]].copy().rename(columns={"date" : "data"})
        else:
            # if sentence is provided, transform the date in sentence
            if type(sentence) == str:
                df = pd.DataFrame({"data" : [sentence]})
            else:
                df = sentence[["X"]].copy().rename(columns={"X" : "data"})

        # preprocess the words
        df["data"] = df["data"].apply(str.lower)
        df["data"] = df["data"].apply(self.remove_punctuation)
        df["data"] = df["data"].apply(word_tokenize)
        df["data"] = df["data"].apply(self.remove_stop_words)
        df["data"] = df["data"].apply(self.force_digits_only)
        df["data"] = df["data"].apply(self.text_to_digit)

        # store the first 3 digit tokens with the year at the end if obvious
        df["digit_tokens"] = df["data"].apply(self.get_3_digit_token)
        # we assume that the first digit token is the day
        df["day"] = pd.to_numeric(df["digit_tokens"].apply(self.get_n_digit_token, args=(1,)))
        # the month can be in plain text like "janvier"
        df["month"] = pd.to_numeric(df["data"].apply(self.month_to_digit))
        # if we didn't find a month in plain text, we assume the month is the 2nd digit token
        mask = df["month"] == 99
        df_temp = df[["digit_tokens"]][mask].copy()
        if df_temp.shape[0] > 0:
            df_temp["month"] = pd.to_numeric(df_temp["digit_tokens"].apply(self.get_n_digit_token, args=(2,)))
            df_temp.drop(columns="digit_tokens", inplace=True)
            df.update(df_temp)
            df["month"] = df["month"].astype(int)
        # we assume the 3rd digit token is the year
        df["year"] = pd.to_numeric(df["digit_tokens"].apply(self.get_n_digit_token, args=(3,)))
        # the day could also be in plain text like "ce jour"
        df["day_of_week"] = df["data"].apply(self.day_of_week)
        df["day_from_today"] = df["data"].apply(self.day_from_today)

        # return df
        return df.drop(columns="digit_tokens")

class Time():
    def __init__(self, path_spacy=""):
        '''
        store path_spacy in self
        '''
        self.path_spacy = path_spacy
        self.df = None
        self.df_exists = False

    def get_data(self):
        '''
        path_spacy is the path to the spaCy model that extracts times from sentences
        path_spacy is a string like "../models/model_v1/model-best"
        '''

        # load the pre-trained spacy model
        model = load_model(self.path_spacy)
        # load the training data
        df = get_data_json()
        # run the model predictions on the data
        df["entities"] = df["translation"].apply(lambda x: return_label(x, model))
        # only keep the times
        def keep_time_only(entities):
            return [entity[0] for entity in entities if entity[1] == "Time"]
        df["times"] = df["entities"].apply(keep_time_only)
        df_split = pd.DataFrame(df["times"].to_list())
        df_split["filename"] = df["fichier"]
        df_stack = pd.DataFrame(df_split.set_index("filename").stack())\
            .reset_index().rename(columns={0 : "time"})

        # store the times in self.df
        self.df = df_stack[["filename", "time"]]
        self.df_exists = True

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

    def get_std_time(self, tokens):
        '''
        guess time from multiple time formats
        return a tuple (hour, minute)
        '''
        # we try the following formats
        time_formats = ["%Hh%M", "%Hh", "%H"]
        hour, minute = 99, 99
        for token in tokens:
            correct_format_found = False
            hour_min = False
            for time_format in time_formats:
                try:
                    hour_min = time.strptime(token, time_format)
                    correct_format_found = True
                except ValueError:
                    correct_format_found = False
            if hour_min:
                if hour == 99:
                    hour = hour_min.tm_hour
                    minute = hour_min.tm_min
        return hour, minute

    def get_plain_text_time(self, tokens):
        '''
        guess time from plain text like "midi"
        return a tuple (hour, minute)
        '''
        hour, minute = 99, 99
        dic_hour = {
            "midi" : 12,
            "minuit" : 0
        }
        dic_min = {
            "demi" : 30,
            "quart" : 15
        }
        for text in dic_hour.keys():
            if text in tokens:
                hour = dic_hour[text]
                break
        for text in dic_min.keys():
            if text in tokens:
                minute = dic_min[text]
                break
        return hour, minute

    def transform_data(self, sentence=""):
        '''
        transform groups of words like "18h" or "18h00" into times
        sentence is one or multiple times
        sentence can be a string or a pandas dataframe with a column "X"
        '''

        if len(sentence) == 0:
            # if sentence is not provided, make transformations from self.df
            if not self.df_exists:
                return pd.DataFrame()
            df = self.df[["time"]].copy().rename(columns={"time" : "data"})
        else:
            # if sentence is provided, transform the time in sentence
            if type(sentence) == str:
                df = pd.DataFrame({"data" : [sentence]})
            else:
                df = sentence[["X"]].copy().rename(columns={"X" : "data"})

        # preprocess the words
        df["data"] = df["data"].apply(str.lower)
        df["data"] = df["data"].apply(self.remove_punctuation)
        df["data"] = df["data"].apply(word_tokenize)
        df["data"] = df["data"].apply(self.remove_stop_words)

        # get time if it's in a standard format
        df["time"] = df["data"].apply(self.get_std_time)
        # if we didn't find the time we try plain text like "midi"
        mask = df["time"] == (99, 99)
        df_temp = df[["data"]][mask].copy()
        if df_temp.shape[0] > 0:
            df_temp["time"] = df["data"].apply(self.get_plain_text_time)
            df_temp.drop(columns="data", inplace=True)
            df.update(df_temp)

        # return df
        return df


if __name__ == '__main__':

    test_date = True
    test_time = True
    test_load_too = True

    if test_date:
        if test_load_too:
            path_spacy = "../models/model_v2/model-best"
            test = Date(path_spacy=path_spacy)
            test.get_data()
            print(test.transform_data())
        else:
            # test one date
            sentence = "2 septembre 2022"
            print("Transform one date")
            test = Date()
            print(test.transform_data(sentence=sentence))
            # test multiple dates
            sentence = pd.DataFrame({"X" : ["2 septembre 2022",
                                            "30 juillet",
                                            "ce jour",
                                            "demain"]})
            print("Transform multiple dates")
            test = Date()
            print(test.transform_data(sentence=sentence))

    if test_time:
        if test_load_too:
            path_spacy = "../models/model_v2/model-best"
            test = Time(path_spacy=path_spacy)
            test.get_data()
            print(test.transform_data())
        else:
            # test one time
            sentence = "18h00"
            print("Transform one time")
            test = Time()
            print(test.transform_data(sentence=sentence))
            # test multiple dates
            sentence = pd.DataFrame({"X" : ["18h00",
                                            "17h",
                                            "6H05",
                                            "minuit et demi",
                                            "midi et quart",
                                            "12 heure moins le quart"]})
            print("Transform multiple times")
            test = Time()
            print(test.transform_data(sentence=sentence))
