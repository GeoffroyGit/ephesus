import locale
import pandas as pd
import seaborn as sns
from ephesus.data import get_data_json, get_data_targets_json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def get_dates(pd_dataframe):
    '''
    Takes a pandas dataframe pd_dataframe and return a dataframe
    with one more column containing the dates found in the dataframe
    (pd_dataframe is expected to have a column "translation" with the sentences)
    '''
    df = pd_dataframe.copy()
    # get the pretrained tokenizer and model from hugging face
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
    model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates")
    # build pipeline
    nlp = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    # run the pretrained pipeline (this may take a while)
    df["camembert"] = nlp(df["translation"].to_list())
    # only keep the dates
    def keep_date_only(entities_serie):
        return [[entity.get("word") for entity in entities if entity.get("entity_group", "empty") == "DATE"] for entities in entities_serie]
    df["date_words"] = keep_date_only(df["camembert"])
    # add number of dates detected for simplicity
    df["date_words_len"] = df["date_words"].map(len)
    # return a dataframe
    return df.drop(columns="camembert")

def get_dates_targets(pd_dataframe):
    '''
    Takes a pandas dataframe and return a dataframe
    which has more columns containing the day, month, year and hour in multiple formats
    (pd_dataframe is expected to have a column "CareBeginDate_1" with the target date)
    '''
    df = pd_dataframe.copy()
    # change dummy year 0001 to dummy year 2000
    def change_year(s):
        if s[:4] == "0001":
            return "2000" + s[4:]
        else:
            return s
    df["CareBeginDate_fix"] = df["CareBeginDate_1"].apply(change_year)
    # change date from string to datetime
    df["CareBeginDate_datetime"] = pd.to_datetime(df["CareBeginDate_fix"], infer_datetime_format=True)
    # trick to change pandas' locale to French
    df["CareBeginDate_datetime"].dt.month_name(locale = ('fr_FR', 'UTF-8'))
    # create multiple formats for date and hour
    def trim_number(n):
        return n.lstrip("0")
    df["CareBeginDate_day_format01"] = df["CareBeginDate_datetime"].dt.strftime("%A")
    df["CareBeginDate_day_format02"] = df["CareBeginDate_datetime"].dt.strftime("%d")
    df["CareBeginDate_day_format03"] = df["CareBeginDate_day_format02"].apply(trim_number)
    df["CareBeginDate_month_format01"] = df["CareBeginDate_datetime"].dt.strftime("%B")
    df["CareBeginDate_month_format02"] = df["CareBeginDate_datetime"].dt.strftime("%m")
    df["CareBeginDate_month_format03"] = df["CareBeginDate_month_format02"].apply(trim_number)
    df["CareBeginDate_year_format01"] = df["CareBeginDate_datetime"].dt.strftime("%y")
    df["CareBeginDate_year_format02"] = df["CareBeginDate_datetime"].dt.strftime("%Y")
    df["CareBeginDate_hour_format01"] = df["CareBeginDate_datetime"].dt.strftime("%Hh%M").apply(trim_number)
    df["CareBeginDate_hour_format02"] = df["CareBeginDate_datetime"].dt.strftime("%Hh").apply(trim_number)
    df["CareBeginDate_hour_format03"] = df["CareBeginDate_datetime"].dt.strftime("%H").apply(trim_number)
    df["CareBeginDate_hour_format04"] = df["CareBeginDate_datetime"].dt.strftime("%Ih%M").apply(trim_number)
    df["CareBeginDate_hour_format05"] = df["CareBeginDate_datetime"].dt.strftime("%Ih").apply(trim_number)
    df["CareBeginDate_hour_format06"] = df["CareBeginDate_datetime"].dt.strftime("%I").apply(trim_number)
    # return a dataframe
    return df.drop(columns="CareBeginDate_fix")

if __name__ == '__main__':
    # test get_dates function
    df = get_data_json()
    # try get dates on small sample
    sample_size = 30
    df_sample = df.sample(sample_size).copy()
    df_sample = get_dates(df_sample)
    print(f"shape of df: {df_sample.shape}")
    # test get_dates_targets function
    df = get_data_targets_json()
    df = get_dates_targets(df)
    print(f"shape of df: {df.shape}")
