# Project Ephesus

Interpret textual data generated from medical vocal memos

In the Library of Celsus in Ephesus, built in the 2nd century, there are four statues depicting wisdom (Sophia), knowledge (Episteme), intelligence (Ennoia) and excellence (Arete). Our project is named after this city and the goddess Sophia.

# What it's all about

After visiting a patient nurses and doctors need to quickly and easily send information

So they record a vocal memo after each visit

Today these memos are read by humans and the infos are manually entered in the database

We want to ease their work by automatically extracting informations from the vocal memos and pre-filling the informations to be entered in the database

# Dataset

4000 vocal memo recordings (4000 sentences)

14 targets to predict (up to 14 different pieces of informations per memo)

# Example

Here is an example of a memo

![memo example](/readme_pictures/exemple.png)

And here is the corresponding informations we need to extract

![memo infos example](/readme_pictures/exemple_infos.png)

# Our approach

## Preprocessing

Clean the data from stop words and punctuation

## Identify groups of words

We identify which part of the memo (which group of words) corresonds to which information

For this, we build a Named Entity Recognition model (NER) using the spaCy library

## Convert each group of word into meaningfull information

We build models to convert each information into the target classes using the nltk library

# Demo

## Try it yourself

You can play around with our demo [here](https://ephesus-web.herokuapp.com/)

In this demo, we let you try your own sentences and see the results from our models

## Going further

Show our success percentage

Give our feedback on possible improvement points and share the hypotheses we used to build our models

# Run our code yourself

## Install the Ephesus package

Clone the project:

```bash
git clone git@github.com:GeoffroyGit/ephesus.git
```

We recommend you to create a fresh virtual environment

Create a python3 virtualenv and activate it:

```bash
cd ephesus
pyenv virtualenv ephesus
pyenv local ephesus
```

Upgrade pip if needed:

```bash
pip install --upgrade pip
```

Install the package:

```bash
pip install -r requirements.txt
pip install -e .
```

## Run the API locally

Run the API on your machine:

```bash
make run_api
```

## Run the API in a Docker container in the cloud

Build the docker image:

```bash
make docker_build
```

Run a container on your machine:

```bash
make docker_run
```

Stop the container running on your machine

```bash
docker ps
docker stop <container id>
```

Push the image to Google Cloud Platform (GCP):

```bash
make docker_push
```

Run a container on GCP:

```bash
make docker_deploy
```

# Train the models yourself

## Training data

You'll need similar training data in order to train the models

We're sorry we can't share our data

## Train the NER with spaCy

### Create folders

```bash
mkdir models
mkdir models/config
```

### Download config

Download base config on https://spacy.io/usage/training (select only French and NER) and save it to models/config/base_config.cfg

![spacy config](/readme_pictures/spacy_config.png)

### Fill config

Fill config file with default values:

```bash
cd models/config/
python -m spacy init fill-config base_config.cfg config.cfg
```

### Create data sets

Create train set and test set for the model:

```bash
cd ephesus/
python sentence.py
```

Create variable to host training data file name (put the same names as in ephesus/sentence.py):

```bash
export EPHESUS_TRAINING_DATA = "train_set_v2.spacy"
export EPHESUS_TEST_DATA = "test_set_v2.spacy"
```

### Train

Train the model:

```bash
cd models/
mkdir model_v2
cd models/config/
python -m spacy train config.cfg --output ../model_v2 --paths.train ../../raw_data/$EPHESUS_TRAINING_DATA --paths.dev ../../raw_data/$EPHESUS_TRAINING_DATA
```

### Evaluate

Evaluate the model:

```bash
cd models/model_v2/
mkdir eval
cd models/config/
python -m spacy evaluate ../model_v2/model-best ../../raw_data/$EPHESUS_TEST_DATA -dp ../model_v2/EVAL -o ../model_v2/EVAL/model_v2_scores.json
```

## Train the RNN with nltk

Train and evaluate the models for treatment and location:

```bash
cd ephesus/
python nlp.py
```

## Check other objects

Check the classes for date and time:

```bash
cd ephesus/
python timedate.py
```

# You're done

Congratulations!
