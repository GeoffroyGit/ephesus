import pandas as pd
import seaborn as sns
from ephesus.data import get_data_json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def get_dates(pd_dataframe):
    '''
    Takes a pandas dataframe df and return a dataframe
    with one more column containing the dates found in the dataframe
    (df is expected to have a column "translation" with the sentences)
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

if __name__ == '__main__':
    # get data
    df = get_data_json()
    # try get dates on small sample
    sample_size = 30
    df_sample = df.sample(sample_size).copy()
    df_sample = get_dates(df_sample)
    print(f"shape of df: {df_sample.shape}")
