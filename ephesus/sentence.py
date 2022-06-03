import json
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from tqdm import tqdm



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

def create_training_set(dico, size=0.8):
    '''
    return train and test data with good format for spacy
    '''
    train, test = train_test_split(dico["annotations"],
                            train_size=size)
    train_set = {'classes':dico['classes'], 'annotations':train}
    test_set = {'classes':dico['classes'], 'annotations':test}
    return train_set, test_set

def create_training_data(train_set, filename):
    '''
    create training data and saved in raw_data
    please specifiy name which should ends with ".spacy"
    '''
    nlp = spacy.blank("fr")
    doc_bin = DocBin()
    for training_example in tqdm(train_set["annotations"]):
        text = training_example["text"]
        labels = training_example["entities"]
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in labels:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        filtered_ents = filter_spans(ents)
        doc.ents = filtered_ents
        doc_bin.add(doc)
        doc_bin.to_disk(f"../raw_data/{filename}")
    print(f"training data created under '../raw_data/{filename}'")

def return_label(sentence, model):
    '''
    return a list of tuple with (text, label)
    needs to take a spacy.model as input
    '''
    liste = []
    doc = model(sentence)
    for index, ent in enumerate(doc.ents):
        liste.append((str(doc.ents[index]), ent.label_))
    return liste


### Commands to train model --> to lauch in config file###

###to fill default parameters
'''
python -m spacy init fill-config base_config.cfg config.cfg
'''
###train the model
'''
python -m spacy train config.cfg --output ../ --paths.train ../../raw_data/training_data.spacy --paths.dev ../../raw_da
ta/training_data.spacy
'''
if __name__ == '__main__':
    print("Ca marhce po")
