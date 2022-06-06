FROM python:3.8.13-buster

COPY ./api /api
COPY ./ephesus /ephesus
COPY ./requirements.txt /requirements.txt
COPY ./models /models
COPY ./model_ngap.joblib /model_ngap.joblib

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python /api/install_nltk_punkt.py

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
