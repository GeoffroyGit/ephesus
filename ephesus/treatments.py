import json
import random

def extract_json(file_name):
    '''
    extraction du json
    '''
    with open("../raw_data/" + file_name) as mon_fichier:
        data = json.load(mon_fichier)
    return data

def tune_data(data):
    '''
    création du dictionnaire
    input = fichier json extrait de segment.ai
    return = un dictionnaire 2 keys ['classes','annotations']
    annotations est une liste de dictionnaires.
    EX: {'text': 'Donc vaccin antigrippale, domicile non facturés.',
    'entities': [(5, 24, 'Treatment'), (26, 34, 'Location')]}
    '''
    #Définition des catégories
    entities = {
    0:"Treatment",
    1:"Cotation",
    2:"Date",
    3:"Time",
    4:"Frequency",
    5:"Duration",
    6:"Location",
    }
    #Création du dictionnaire attendu pour entrainer le NER
    my_dico = {'classes' :
                        ['Treatment', 'Cotation', 'Date', 'Time',
                        'Frequency', 'Duration', 'Location'],
                    'annotations': []}

    sentences = data['dataset']['samples']

    for sentence in sentences:
        temp_dict = {}
        temp_dict['text'] = sentence["attributes"]["text"]
        temp_dict["entities"] = []
        for annotation in sentence["labels"]["ground-truth"]["attributes"]["annotations"]:
            start = annotation["start"]
            end = annotation["end"]
            label = entities[annotation["category_id"]]
            temp_dict["entities"].append((start,end,label))
        my_dico['annotations'].append(temp_dict)

    return my_dico

def train_test_split_custom(my_dico, train_size=0.8):
    '''
    Séparation du dictionnaire obtennu via tune_data
    Return = deux dictionnaires sous le même format
    '''
    #Split du train et du test
    nb_elm = len(my_dico["annotations"]) * train_size
    train = random.sample(my_dico["annotations"], nb_elm)
    test = [i for i in train if i not in train]

    train_set = {'classes' :
                        ['Treatment', 'Cotation', 'Date', 'Time',
                        'Frequency', 'Duration', 'Location'],
                    'annotations': train}
    test_set = {'classes' :
                        ['Treatment', 'Cotation', 'Date', 'Time',
                        'Frequency', 'Duration', 'Location'],
                    'annotations': test}

    return train_set, test_set
