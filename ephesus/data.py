from os import listdir
from os.path import isfile, join
import pandas as pd
import json


def get_data_json():

    """
    cette fonction permet d'ouvrir et de récupérer le contenu de chaque memo vocal
    rendu : un dataframe (nom du fichier, translation du mémo)
    Attention il faut avoir les données json dans le dossier LOCAL_PATH
    """

    LOCAL_PATH = "../raw_data/input_json"

    # monRepertoire = "Data"

    # récupération des noms des fichiers output translation des memos vocaux
    fichiers = [fichier for fichier in listdir(LOCAL_PATH) if isfile(join(LOCAL_PATH, fichier))]

    # ouverture des fichiers json et récupération de la translation exemple ci dessous
    # OUTPUT MEMO VOCAL
    # {
    #    "Translation": "Une prise de sang à enregistrer sur la tournée de numéro un le 25 janvier à 8h45 sur la commune de Gémenos, 26 km montagne. Merci."
    # }

    data = []
    for fichier in fichiers :
        lib_fichier = LOCAL_PATH + "/" + fichier
        with open(lib_fichier) as mon_fichier:
            data.append(json.load(mon_fichier))

    # récupération seulement de la phrase = sentence du mémo
    data = [data[i]["Translation"] for i in range(len(data))]

    # création d'un dataframe avec nom du fichier et la phrase
    df_translation = pd.DataFrame({'fichier': fichiers, 'translation': data})

    return df_translation

if __name__ == '__main__':
    df = get_data_json()
    df.to_csv("../raw_data/input_data.csv")