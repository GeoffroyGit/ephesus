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

def  get_data_targets_json():

    """
    cette fonction permet d'ouvrir et de récupérer le contenu des targets
    rendu : un dataframe (nom du fichier, targets )

    ATTENTION - il faut avoir les données json dans le dossier LOCAL_PATH

    ATTENTION - on ne traite pas la variable CareIntervals qui peut être une liste vite

    """
    LOCAL_PATH = "../raw_data/targets"

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

    # récupération pour le 1er traitement

    TreatmentDetected_1 = [data[i]["Treatments"][0]["TreatmentCares"][0]["TreatmentDetected"] for i in range(len(data))]
    NGAP_1 = [data[i]["Treatments"][0]["TreatmentCares"][0]["NGAP"] for i in range(len(data))]
    Cotation_1 = [data[i]["Treatments"][0]["TreatmentCares"][0]["Cotation"] for i in range(len(data))]
    Hour_1 = [data[i]["Treatments"][0]["TreatmentCares"][0]["Hour"] for i in range(len(data))]
    AldRelated_1 = [data[i]["Treatments"][0]["TreatmentCares"][0]["AldRelated"] for i in range(len(data))]

    # on ne traite pas la variable CareIntervals qui peut être une liste vite

    # IntervalType_1 = []
    # for i in range(len(data)) :
    #     if len(data[i]["Treatments"][0]["CareIntervals"]) > 0 :
    #         IntervalType_1 = IntervalType_1.append(data[i]["Treatments"][0]["CareIntervals"][0]["IntervalType"])
    #     else :
    #         IntervalType_1 = IntervalType_1.append("null")

    # IntervalInDays_1 = [data[i]["Treatments"][0]["CareIntervals"][0]["IntervalInDays"] for i in range(len(data)) if len(data[i]["Treatments"][0]["CareIntervals"]) > 0 ]
    # PassingDayOfWeek_1 = [data[i]["Treatments"][0]["CareIntervals"][0]["PassingDayOfWeek"] for i in range(len(data)) if len(data[i]["Treatments"][0]["CareIntervals"]) > 0 ]
    # PassingDate_1 = [data[i]["Treatments"][0]["CareIntervals"][0]["PassingDate"] for i in range(len(data)) if len(data[i]["Treatments"][0]["CareIntervals"]) > 0 ]

    CareBeginDate_1 = [data[i]["Treatments"][0]["CareBeginDate"] for i in range(len(data))]
    CareDuration_1 = [data[i]["Treatments"][0]["CareDuration"] for i in range(len(data))]
    CareDurationTypeEnum_1 = [data[i]["Treatments"][0]["CareDurationTypeEnum"] for i in range(len(data))]
    CareOnPublicHoliday_1 = [data[i]["Treatments"][0]["CareOnPublicHoliday"] for i in range(len(data))]
    CareLocation_1 = [data[i]["Treatments"][0]["CareLocation"] for i in range(len(data))]
    CareBeginHpName_1 = [data[i]["Treatments"][0]["CareBeginHpName"] for i in range(len(data))]
    ZoneName_1 = [data[i]["Treatments"][0]["ZoneName"] for i in range(len(data))]
    IK_1 = [data[i]["Treatments"][0]["IK"] for i in range(len(data))]

    # création d'un dataframe avec nom du fichier et les targets
    df_target = pd.DataFrame({
        "fichier": fichiers,
        "TreatmentDetected_1" : TreatmentDetected_1,
        "NGAP_1" : NGAP_1,
        "Cotation_1" : Cotation_1,
        "Hour_1" : Hour_1,
        "AldRelated_1" : AldRelated_1,

        # "IntervalType_1" : IntervalType_1,
        # "IntervalType_1" : IntervalInDays_1,
        # "PassingDayOfWeek_1" : PassingDayOfWeek_1,
        # "PassingDate_1" : PassingDate_1,

        "CareBeginDate_1" : CareBeginDate_1,
        "CareDuration_1" : CareDuration_1,
        "CareDurationTypeEnum_1" : CareDurationTypeEnum_1,
        "CareOnPublicHoliday_1" : CareOnPublicHoliday_1,
        "CareLocation_1" : CareLocation_1,
        "CareBeginHpName_1" : CareBeginHpName_1,
        "ZoneName_1" : ZoneName_1,
        "IK_1" : IK_1
        })

    return df_target

LOCAL_PATH_TRANSLATION = "../raw_data/input_data.csv"
LOCAL_PATH_TARGET = "../raw_data/input_data_target.csv"

def get_data_csv():
    '''returns a DataFrame avec x = translation et y = target '''
    df_translation = pd.read_csv(LOCAL_PATH_TRANSLATION)
    df_target = pd.read_csv(LOCAL_PATH_TARGET)

    df_translation.drop(columns=["Unnamed: 0"], inplace=True)
    df_translation["fichier"] = df_translation["fichier"].str[0:-17]

    df_target.drop(columns = ["Unnamed: 0"], inplace=True)
    df_target["fichier"] = df_target["fichier"].str[0:-16]

    df = df_translation.merge(df_target)

    df.set_index("fichier", inplace=True)

    return df

if __name__ == '__main__':
    df = get_data_json()
    df.to_csv("../raw_data/input_data.csv")
    df_target = get_data_targets_json()
    df_target.to_csv("../raw_data/input_data_target.csv")
    df_final = get_data_csv()
    df_final.to_csv("../raw_data/input_data_final.csv")
