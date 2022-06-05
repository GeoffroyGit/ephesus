from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import joblib

from ephesus.nlp import TrainerNGAP
from ephesus.timedate import Date, Time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/test")
def test(sentence):
    '''
    return a dummy sentence with entities
    this is to enable development of the front end
    '''
    # create dummy sentence to be returned (ignore incoming sentence)
    dummy_sentence = (
        "Un ",
        ("grand pansement", "traitement", "#8ef"),
        " à ",
        ("domicile", "lieux", "#faa"),
        (" à partir du "),
        ("3 juin", "date", "#afa"),
        " , ",
        ("tous les deux jours", "fréquence", "#fea"),
        " ",
        ("pendant 3 semaines", "durée", "#8ef"),
        " à ",
        ("18h", "heure", "#afa"),
        "."
    )

    # load the pre-trained model
    #try:
    #    model = joblib.load("model.joblib")
    #except FileNotFoundError:
    #    return {"error" : "oops"}

    # compute prediction
    #y_pred = model.predict(X_pred)

    # return predicted
    return {"entities" : dummy_sentence}

@app.get("/predict")
def predict():
    return {"greeting": "not coded yet"}

@app.get("/treatment")
def treatment(sentence):
    '''
    return NGAP code corresponding to the treatment in sentence
    '''
    # initiate path to pre-trained models
    path_spacy = "../models/model_v2/model-best"
    path_ngap = "../model_ngap.joblib"
    # create a new trainer for precitions
    trainer = TrainerNGAP(train_on_full_set = True, path_spacy=path_spacy, path_ngap=path_ngap)
    df = trainer.predict_ngap(sentence=sentence)
    result = {
        "NGAP" : df["NGAP"][0],
        "softmax" : df["softmax"][0]
    }
    return result

@app.get("/date")
def date(sentence):
    '''
    return date in date format
    '''
    test = Date()
    df = test.transform_data(sentence=sentence)
    result = {
        "day" : df["day"][0],
        "month" : df["month"][0],
        "year" : df["year"][0],
        "day_of_week" : df["day_of_week"][0],
        "day_from_today" : df["day_from_today"][0]
    }
    return result

@app.get("/time")
def time(sentence):
    '''
    return time in time format
    '''
    test = Time()
    df = test.transform_data(sentence=sentence)
    result = {
        "hour" : df["time"][0][0],
        "minute" : df["time"][0][1]
    }
    return result

if __name__ == "__main__":
    print(test("hello"))
    print(treatment("Prise de sang"))
    print(date("2 septembre 2022"))
    print(time("18h45"))
