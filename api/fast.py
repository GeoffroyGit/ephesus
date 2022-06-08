from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import joblib
import base64

from ephesus.nlp import TrainerLocation, TrainerNGAP
from ephesus.timedate import Date, Time
from ephesus.sentence import load_model

from copy import deepcopy

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
    return {"entities" : dummy_sentence}

@app.get("/predict")
def predict(sentence):
    path_spacy = "../models/model_v2/model-best"
    # load the spacy model
    model = load_model(path_spacy)
    # run the model predictions on the sentence
    labels = model.get_pipe("ner").labels
    doc = model(sentence)
    doc_bytes = base64.b64encode(doc.to_bytes()).decode()
    vocab_bytes = base64.b64encode(model.vocab.to_bytes()).decode()
    # return prediction with labels
    return {"labels" : labels, "vocab" : vocab_bytes, "doc" : doc_bytes}

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
        "NGAP" : str(df["NGAP"][0]),
        "softmax" : float(df["softmax"][0])
    }
    return result

@app.get("/location")
def location(sentence):
    '''
    return Cabinet or Domicile as the care location
    '''
    # initiate path to pre-trained models
    path_spacy = "../models/model_v2/model-best"
    path_loc = "../model_location.joblib"
    # create a new trainer for precitions
    trainer = TrainerLocation(train_on_full_set = True, path_spacy=path_spacy, path_loc=path_loc)
    df = trainer.predict_location(sentence=sentence)
    result = {
        "location" : str(df["location"][0]),
        "sigmoid" : float(df["sigmoid"][0])
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
        "day" : int(df["day"][0]),
        "month" : int(df["month"][0]),
        "year" : int(df["year"][0]),
        "day_of_week" : int(df["day_of_week"][0]),
        "day_from_today" : int(df["day_from_today"][0])
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
        "hour" : int(df["time"][0][0]),
        "minute" : int(df["time"][0][1])
    }
    return result

@app.get("/all")
def all(sentence):
    '''
    return a JSON with all the info we can extract from sentence
    '''
    # define path
    path_spacy = "../models/model_v2/model-best"
    path_ngap = "../model_ngap.joblib"
    path_loc = "../model_location.joblib"
    # initiate the JSON to be returned
    empty_dict = {
        "TreatmentCares" : {
            "TreatementDetected" : "",
            "NGAP": {},
            "CotationDetected" : "",
            "Cotation": "",
            "HourDetected" : "",
            "Hour": ""
        },
        "CareBeginDateDetected" : "",
        "CareBeginDate" : {},
        "CareLocationDetected" : "",
        "CareLocation": {},
        "CareDurationDetected" : "",
        "CareFrequencyDetected" : ""
    }
    result_dict = {
        "Voice" : {
            "Transcription" : sentence,
            "Treatments" : [deepcopy(empty_dict)]
        }
    }
    # load the spacy model
    model_spacy = load_model(path_spacy)
    # run the model predictions on the sentence
    doc = model_spacy(sentence)
    entities = [(str(ent), ent.label_) for ent in doc.ents]
    known_entities = ("Treatment", "Cotation", "Time", "Date", "Location", "Duration", "Frequency")
    # run the other models
    treatment_count = 0
    for entity in entities:
        sentence = entity[0]
        if entity[1] == known_entities[0]: # Treatment
            # create new item in list of treatments if needed
            if result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["TreatementDetected"]\
                or result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["CotationDetected"]:
                    result_dict["Voice"]["Treatments"].append(deepcopy(empty_dict))
                    treatment_count += 1
            # add sentence to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["TreatementDetected"] = sentence
            # create a trainer for treatment
            trainer = TrainerNGAP(train_on_full_set = True, path_spacy=path_spacy, path_ngap=path_ngap)
            df = trainer.predict_ngap(sentence=sentence)
            ngap = str(df["NGAP"][0])
            softmax = float(df["softmax"][0])
            # add code and softmax to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["NGAP"] = {
                "NGAP" : ngap,
                "softmax" : softmax
            }
        elif entity[1] == known_entities[1]: # Cotation
            # create new item in list of treatments if needed
            if result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["TreatementDetected"]\
                or result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["CotationDetected"]:
                    result_dict["Voice"]["Treatments"].append(deepcopy(empty_dict))
                    treatment_count += 1
            # add sentence to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["CotationDetected"] = sentence
            result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["Cotation"] = sentence
        elif entity[1] == known_entities[2]: # Time
            # create new item in list of treatments if needed
            if result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["HourDetected"]:
                result_dict["Voice"]["Treatments"].append(deepcopy(empty_dict))
                treatment_count += 1
            # add sentence to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["HourDetected"] = sentence
            # make prediction
            test = Time()
            df = test.transform_data(sentence=sentence)
            hour = int(df["time"][0][0])
            minute = int(df["time"][0][1])
            # add hour and minute to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["TreatmentCares"]["Hour"] = f"{hour}:{minute}"
        elif entity[1] == known_entities[3]: # Date
            # create new item in list of treatments if needed
            if result_dict["Voice"]["Treatments"][treatment_count]["CareBeginDateDetected"]:
                result_dict["Voice"]["Treatments"].append(deepcopy(empty_dict))
                treatment_count += 1
            # add sentence to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["CareBeginDateDetected"] = sentence
            # make prediction
            test = Date()
            df = test.transform_data(sentence=sentence)
            day = int(df["day"][0])
            month = int(df["month"][0])
            year = int(df["year"][0])
            day_of_week = int(df["day_of_week"][0])
            day_from_today = int(df["day_from_today"][0])
            # make year into four digits format
            year_long = 9999 if year == 99 else year
            if year_long < 70:
                year_long += 2000
            elif year_long < 100:
                year_long += 1900
            # add date to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["CareBeginDate"] = {
                "date" : f"{year_long}{month:02}{day:02}",
                "day" : day,
                "month" : month,
                "year" : year,
                "day_of_week" : day_of_week,
                "day_from_today" : day_from_today
            }
        elif entity[1] == known_entities[4]: # Location
            # create new item in list of treatments if needed
            if result_dict["Voice"]["Treatments"][treatment_count]["CareLocationDetected"]:
                result_dict["Voice"]["Treatments"].append(deepcopy(empty_dict))
                treatment_count += 1
            # add sentence to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["CareLocationDetected"] = sentence
            # create a trainer for location
            trainer = TrainerLocation(train_on_full_set = True, path_spacy=path_spacy, path_loc=path_loc)
            df = trainer.predict_location(sentence=sentence)
            location = str(df["location"][0])
            sigmoid = float(df["sigmoid"][0])
            # add location and sigmoid to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["CareLocation"] = {
                "location" : location,
                "sigmoid" : sigmoid
            }
        elif entity[1] == known_entities[5]: # Duration
            # create new item in list of treatments if needed
            if result_dict["Voice"]["Treatments"][treatment_count]["CareDurationDetected"]:
                result_dict["Voice"]["Treatments"].append(deepcopy(empty_dict))
                treatment_count += 1
            # add sentence to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["CareDurationDetected"] = sentence
        elif entity[1] == known_entities[6]: # Frequency
            # create new item in list of treatments if needed
            if result_dict["Voice"]["Treatments"][treatment_count]["CareFrequencyDetected"]:
                result_dict["Voice"]["Treatments"].append(deepcopy(empty_dict))
                treatment_count += 1
            # add sentence to JSON
            result_dict["Voice"]["Treatments"][treatment_count]["CareFrequencyDetected"] = sentence
    # return JSON
    return result_dict

if __name__ == "__main__":
    print(test("hello"))
    print(treatment("Prise de sang"))
    print(date("2 septembre 2022"))
    print(time("18h45"))
    print(location("chez le patient"))
    #print(predict("Prise de sang au cabinet le 2 septembre 2020"))
    print(all("Prise de sang au cabinet le 3 septembre 2020 à 18h35 mais aussi une ami2 à 9h demain et tous les jours"))
