from ephesus.data import get_data_csv
from ephesus.sentence import load_model, return_label
from ephesus.sentence import return_label, load_model
import numpy as np
import pandas as pd
import re
import datetime
from ephesus.timedate import Date

class Duration():
    def __init__(self, path_spacy=""):
        '''
        store path_spacy in self
        '''
        self.path_spacy = path_spacy
        self.df = None
        self.df_exists = False
        self.liste_mois = '(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)'

    def get_data(self):
        '''
        path_spacy is the path to the spaCy model that extracts dates from sentences
        path_spacy is a string like "../models/model_v1/model-best"
        '''
        #load the pre-trained spacy model
        model = load_model(self.path_spacy)
        # load the training data
        df = pd.DataFrame(get_data_csv()["translation"])
        # run the model predictions on the data
        df["entities"] = df["translation"].apply(lambda x: return_label(x, model))
        # extract predictions
        def extract(prediction, label):
            '''
            extract span given a label for one prediction using return_label()
            '''
            for elm in prediction:
                if elm[1] == label:
                    if elm[0] == np.nan:
                        return np.nan
                    else:
                        return elm[0]
        # only keep the date and Duration
        df["raw_date"] = df["entities"].apply(lambda x: extract(x,'Date'))
        df["Duration"] = df["entities"].apply(lambda x: extract(x,'Duration'))

        #We need (month, day) for the given date
        def apply_date(sentence):
            if sentence == None:
                return None
            else:
                df = Date().transform_data(sentence)
                return [df.iloc[0,2], df.iloc[0,1]]

        df["Date"] = df["raw_date"].apply(lambda x: apply_date(x))

        df.drop(columns=["entities", "raw_date"], inplace=True)

        # store in self.df
        self.df = df
        self.df_exists = True

        return df

    def explicit_duration(self, sentence=None, df=False):
        '''
        return CareDuration and CareDurationType given a sentence. Ex:
        Pendant 3 jours
        Pour une durée de 3 semaines..
        '''
        if sentence == None:
            return np.nan
        dico = {}
        x = re.search('\d{1,3}\s(jours|mois|semaines|an)', sentence)
        if x == None:
            return None
        x = x.group().split()
        care_duration = x[0]
        care_duration_type = x[1]
        care_duration_type = care_duration_type.replace("jours", "Days")
        care_duration_type = care_duration_type.replace("mois", "Months")
        care_duration_type = care_duration_type.replace("semaines", "Weeks")
        care_duration_type = care_duration_type.replace("an", "Year")
        dico["CareDuration"] = [care_duration]
        dico["CareDurationType"] = [care_duration_type]
        dico["CareEnd"] = [np.nan]
        if df:
            return pd.DataFrame.from_dict(dico)
        return dico

    def implicit_duration_1(self, sentence=None, CareBeginDate=None, df=False):
        '''
        return full duration if CareBeginDate not None or return all dateframe
        CareBeginDate is a tuple(month,day)
        Careduration by default returned as Days
        '''
        dic_months = {
            1:"janvier",
            2:"février",
            3:"mars",
            4:"avril",
            5:"mai",
            6:"juin",
            7:"juillet",
            8:"août",
            9:"septembre",
            10:"octobre",
            11:"novembre",
            12:"décembre"
        }
        if sentence == None:
            return np.nan
        dico = {}
        identifier = re.search("jusqu\'au\s", sentence)
        if identifier == None:
            return None
        df_end_date = Date().transform_data(sentence)
        if df_end_date.iloc[0,1] == 99 or df_end_date.iloc[0,2] == 99:
            dico["CareDuration"] = [np.nan]
            dico["CareDurationType"] = [np.nan]
            dico["CareEnd"] = [np.nan]
            if df == True:
                return pd.DataFrame.from_dict(dico)
            return dico
        end_date_text = f"{df_end_date.iloc[0,1]} {dic_months[df_end_date.iloc[0,2]]}"
        if CareBeginDate == None or CareBeginDate[0] == 99 or CareBeginDate[1] == 99:
            dico["CareDuration"] = [np.nan]
            dico["CareDurationType"] = [np.nan]
            dico["CareEnd"] = [end_date_text]
            if df == True:
                return pd.DataFrame.from_dict(dico)
            return dico
        #transformation de end_date
        end_date_datetime = datetime.date(2022,df_end_date.iloc[0,2],df_end_date.iloc[0,1])
        #transformation de CareBeginDate
        start_date = datetime.date(2022,CareBeginDate[0],CareBeginDate[1])
        diff = (end_date_datetime - start_date).days
        if diff <= 0:
            dico["CareDuration"] = [np.nan]
        else:
            dico["CareDuration"] = [diff]
        # by default, CareDurationType is Days
        dico["CareDurationType"] = ["Days"]
        dico["CareEnd"] = [end_date_text]
        if df == True:
            return pd.DataFrame.from_dict(dico)
        return dico

    def replace_un_une(self, sentence):
        '''
        simply replace 'un' or 'une' by 1 if its date related
        '''
        if re.search('(un|une)\s(jour|mois|semaine|an)', sentence) is not None:
            sentence = sentence.replace("une","1")
            sentence = sentence.replace("un","1")
        return sentence

    def get_duration(self, sentence=None, CareBeginDate=None, df=False):
        '''
        '''
        if sentence == None:
            return None
        sentence = self.replace_un_une(sentence)
        sentence = sentence.replace('premier', '1')
        sentence = re.sub(' +', ' ', sentence).lower()
        if re.search('\d{1,3}\s(jours|mois|semaines|an)', sentence) is not None:
            return self.explicit_duration(sentence,df=df)
        if re.search("jusqu\'au\s", sentence) is not None:
            return self.implicit_duration_1(sentence, CareBeginDate=CareBeginDate, df=df)

    def apply_to_df(self, df):
        df['temp'] = df.Duration.combine(df.Date, func=self.get_duration)
        return df

    if __name__ == '__main__':
        path_spacy = "../models/model_full/model-best"
        print("creating data")
        test = Duration(path_spacy=path_spacy)
        df = test.get_data()
        apply_to_df(df)
