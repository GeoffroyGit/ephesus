from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import joblib


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

if __name__ == "__main__":
    print(test("hello"))
