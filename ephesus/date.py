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
    df["words"] = keep_date_only(df["camembert"])
    # add number of dates detected for simplicity
    df["words_len"] = df["words"].map(len)
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

def evaluate_date(pd_dataframe_X, pd_dataframe_y, format_X=True, X_is_date=True):
    '''
    evaluate if the date was correctly detected
    pd_dataframe_y is expected to be a dataframe in the format returned by get_dates_targets()
    pd_dataframe_X can have two different formats:
    - if format_X == False then it's expected to be in the format returned by get_dates()
    - if format_X == True then it's expected to be a dataframe with:
        - a column "fichier"
        - a column "words"
      For exemple:
        pd_dataframe_X = pd.DataFrame({
            "fichier" : [
                "9d42ce6f-8537-49a9-a121-c8ae1dc7cda8_5264fbd..._extraction.json",
                "c619f0e1-7c22-4047-adb2-df4dec6643ba_141bd63..._extraction.json",
                "9f980dcf-b431-4e67-876f-2b8e288b7900_819bc8a..._extraction.json",
                "9d42ce6f-8537-49a9-a121-c8ae1dc7cda8_9b13a58..._extraction.json",
                "55d674cc-3389-4cf6-ab7c-1f1b9fa1b6ed_456f8b9..._extraction.json",
                "2d035c4b-cdfa-4982-87dc-916fe07a0824_1275c4e..._extraction.json"
            ],
            "words" : ["29 décembre",
                     "jeudi 16 décembre 2021",
                     "29 mars",
                     "16 novembre",
                     "30 juillet 2020",
                     "24 01 22"]
        })
    if X_is_date == True then words is expected to contain a date
    if X_is_date == False then words is expected to contain a time
    '''
    df_X = pd_dataframe_X.copy()
    df_y = pd_dataframe_y.copy()
    # clean filename for merge
    def clean_filename(filename):
        return filename[:-17] if "translation.json" in filename else filename[:-16]
    df_X["filename_cleaned"] = df_X["fichier"].apply(clean_filename)
    df_y["filename_cleaned"] = df_y["fichier"].apply(clean_filename)
    # merge
    df = df_X.merge(df_y, how="left", on="filename_cleaned")

    # calculate score
    df_score = df[["filename_cleaned", "words"]].copy()

    if format_X:
        # spaCy format
        if X_is_date:
            df_score["score_day_format01"] = df.apply(lambda x: x["CareBeginDate_day_format01"] in x["words"], axis=1)
            df_score["score_day_format02"] = df.apply(lambda x: x["CareBeginDate_day_format02"] in x["words"], axis=1)
            df_score["score_day_format03"] = df.apply(lambda x: x["CareBeginDate_day_format03"] in x["words"], axis=1)
            df_score["score_month_format01"] = df.apply(lambda x: x["CareBeginDate_month_format01"] in x["words"], axis=1)
            df_score["score_month_format02"] = df.apply(lambda x: x["CareBeginDate_month_format02"] in x["words"], axis=1)
            df_score["score_month_format03"] = df.apply(lambda x: x["CareBeginDate_month_format03"] in x["words"], axis=1)
            df_score["score_year_format01"] = df.apply(lambda x: x["CareBeginDate_year_format01"] in x["words"], axis=1)
            df_score["score_year_format02"] = df.apply(lambda x: x["CareBeginDate_year_format02"] in x["words"], axis=1)
        else:
            df_score["score_hour_format01"] = df.apply(lambda x: x["CareBeginDate_hour_format01"] in x["words"], axis=1)
            df_score["score_hour_format02"] = df.apply(lambda x: x["CareBeginDate_hour_format02"] in x["words"], axis=1)
            df_score["score_hour_format03"] = df.apply(lambda x: x["CareBeginDate_hour_format03"] in x["words"], axis=1)
            df_score["score_hour_format04"] = df.apply(lambda x: x["CareBeginDate_hour_format04"] in x["words"], axis=1)
            df_score["score_hour_format05"] = df.apply(lambda x: x["CareBeginDate_hour_format05"] in x["words"], axis=1)
            df_score["score_hour_format06"] = df.apply(lambda x: x["CareBeginDate_hour_format06"] in x["words"], axis=1)
    else:
        # camember format
        if X_is_date:
            df_score["score_day_format01"] = df.apply(lambda x: x["CareBeginDate_day_format01"] in " ".join(x["words"]), axis=1)
            df_score["score_day_format02"] = df.apply(lambda x: x["CareBeginDate_day_format02"] in " ".join(x["words"]), axis=1)
            df_score["score_day_format03"] = df.apply(lambda x: x["CareBeginDate_day_format03"] in " ".join(x["words"]), axis=1)
            df_score["score_month_format01"] = df.apply(lambda x: x["CareBeginDate_month_format01"] in " ".join(x["words"]), axis=1)
            df_score["score_month_format02"] = df.apply(lambda x: x["CareBeginDate_month_format02"] in " ".join(x["words"]), axis=1)
            df_score["score_month_format03"] = df.apply(lambda x: x["CareBeginDate_month_format03"] in " ".join(x["words"]), axis=1)
            df_score["score_year_format01"] = df.apply(lambda x: x["CareBeginDate_year_format01"] in " ".join(x["words"]), axis=1)
            df_score["score_year_format02"] = df.apply(lambda x: x["CareBeginDate_year_format02"] in " ".join(x["words"]), axis=1)
        else:
            df_score["score_hour_format01"] = df.apply(lambda x: x["CareBeginDate_hour_format01"] in " ".join(x["words"]), axis=1)
            df_score["score_hour_format02"] = df.apply(lambda x: x["CareBeginDate_hour_format02"] in " ".join(x["words"]), axis=1)
            df_score["score_hour_format03"] = df.apply(lambda x: x["CareBeginDate_hour_format03"] in " ".join(x["words"]), axis=1)
            df_score["score_hour_format04"] = df.apply(lambda x: x["CareBeginDate_hour_format04"] in " ".join(x["words"]), axis=1)
            df_score["score_hour_format05"] = df.apply(lambda x: x["CareBeginDate_hour_format05"] in " ".join(x["words"]), axis=1)
            df_score["score_hour_format06"] = df.apply(lambda x: x["CareBeginDate_hour_format06"] in " ".join(x["words"]), axis=1)

    if X_is_date:
        df_score["score_day"] = \
            df_score["score_day_format01"] | \
            df_score["score_day_format02"] | \
            df_score["score_day_format03"]
        df_score["score_month"] = \
            df_score["score_month_format01"] | \
            df_score["score_month_format02"] | \
            df_score["score_month_format03"]
        df_score["score_year"] = \
            df_score["score_year_format01"] | \
            df_score["score_year_format02"]
    else:
        df_score["score_hour"] = \
            df_score["score_hour_format01"] | \
            df_score["score_hour_format02"] | \
            df_score["score_hour_format03"] | \
            df_score["score_hour_format04"] | \
            df_score["score_hour_format05"] | \
            df_score["score_hour_format06"]

    if X_is_date:
        df_score["score"] = (df_score["score_day"].astype(int) * 0.4 +
                            df_score["score_month"].astype(int) * 0.4 +
                            df_score["score_year"].astype(int) * 0.2)
    else:
        df_score["score"] = df_score["score_hour"].astype(int) * 1.0

    # return a dataframe
    return df_score[["filename_cleaned", "score"]]

if __name__ == '__main__':

    test_camembert = False # change this boolean to test with camembert model
    test_spacy = True      # change this boolean to test with spacy model

    # test get_dates function
    df_1 = get_data_json()

    # test get_dates_targets function
    df_2 = get_data_targets_json()
    df_2 = get_dates_targets(df_2)
    print(f"shape of df: {df_2.shape}")

    if test_camembert:
        print("Test CamemBERT:")
        # try get dates on small sample
        sample_size = 30
        df_sample = df_1.sample(sample_size).copy()
        df_sample = get_dates(df_sample)
        print(f"shape of df: {df_sample.shape}")
        # test evaluate_date
        df_3 = evaluate_date(df_sample, df_2, format_X=False, X_is_date=True)
        score = df_3["score"].mean()
        print(f"score: {score}")

    if test_spacy:
        print("Test Spacy on dates:")
        # dummy data for now
        df_sample = pd.DataFrame({
            "fichier" : [
                "9d42ce6f-8537-49a9-a121-c8ae1dc7cda8_403bfcb4-8646-4b45-a1a4-06cdb747f8fc_extraction.json",
                "f0caa21b-c11d-40a3-98ec-e2df3d2b9cc4_790cfe40-a13d-4719-8f94-1b5bc308102c_extraction.json",
                "2d035c4b-cdfa-4982-87dc-916fe07a0824_3a20d2f2-def7-4abd-a0ec-5e5a8ffb773b_extraction.json",
                "74b154c1-e265-4cb9-8e71-0c9bbc3dc880_5ea80fbe-1cc2-4b8a-af5e-cc01107c4320_extraction.json",
                "2d035c4b-cdfa-4982-87dc-916fe07a0824_0509f8ab-e39b-4e8a-ae5c-ab2acd708867_extraction.json",
                "2206f47b-a858-4f23-8696-d10e1050f2d7_3af26627-bc72-4c41-9254-ae7e257d335c_extraction.json"
            ],
            "words" : ["25 septembre",
                     "8 septembre",
                     "7 février",
                     "lundi 21 février",
                     "17 09 2020",
                     "28 02 22"]
        })
        # test evaluate_date
        df_3 = evaluate_date(df_sample, df_2, format_X=True, X_is_date=True)
        score = df_3["score"].mean()
        print(f"score: {score}")

        print("Test Spacy on times:")
        # dummy data for now
        df_sample = pd.DataFrame({
            "fichier" : [
                "9d42ce6f-8537-49a9-a121-c8ae1dc7cda8_403bfcb4-8646-4b45-a1a4-06cdb747f8fc_extraction.json",
                "f0caa21b-c11d-40a3-98ec-e2df3d2b9cc4_790cfe40-a13d-4719-8f94-1b5bc308102c_extraction.json",
                "2d035c4b-cdfa-4982-87dc-916fe07a0824_3a20d2f2-def7-4abd-a0ec-5e5a8ffb773b_extraction.json",
                "74b154c1-e265-4cb9-8e71-0c9bbc3dc880_5ea80fbe-1cc2-4b8a-af5e-cc01107c4320_extraction.json",
                "2d035c4b-cdfa-4982-87dc-916fe07a0824_0509f8ab-e39b-4e8a-ae5c-ab2acd708867_extraction.json",
                "2206f47b-a858-4f23-8696-d10e1050f2d7_3af26627-bc72-4c41-9254-ae7e257d335c_extraction.json"
            ],
            "words" : ["8h",
                     "6h",
                     "midi",
                     "11h00",
                     "6h30",
                     "18h"]
        })
        # test evaluate_date
        df_3 = evaluate_date(df_sample, df_2, format_X=True, X_is_date=False)
        score = df_3["score"].mean()
        print(f"score: {score}")
