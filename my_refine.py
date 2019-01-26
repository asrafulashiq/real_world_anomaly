from pathlib import Path
import random
import os
from collections import defaultdict


def write_to_file(fp, lines):
    for line in lines:
        fp.write()


_HOME = os.environ["HOME"]
PARENT_FOLDER = Path(_HOME+"/dataset/UCF_crime")


split_folder = PARENT_FOLDER / "custom_split"
split_folder.mkdir(exist_ok=True)

orig_split_train = PARENT_FOLDER / "Anomaly_Detection_splits/Anomaly_Train.txt"
orig_split_test = PARENT_FOLDER / "Anomaly_Detection_splits/Anomaly_Test.txt"

feature_name = "C3D_features"
feature_folder = PARENT_FOLDER / feature_name / "Avg"


LABEL_ANOMS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
               'Explosion', 'Fighting', 'RoadAccidents', 'Robbery',
               'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

# LABEL_ANOMS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary']
DOWN_RATIO = None  # 5. / 13  # or None


""" create dict """
dict_test = defaultdict(list)
dict_train = defaultdict(list)
list_train = []
DICT_LABELS = LABEL_ANOMS + ['Testing_Normal_Videos_Anomaly',
                             'Training-Normal-Videos']

with orig_split_test.open("r") as fp:
    for line in fp:
        line = line.strip()
        line_split = line.split("/")
        if line_split[0] in DICT_LABELS:
            dict_test[line_split[0]].append(line_split[1])

for anom in LABEL_ANOMS:
    folder = PARENT_FOLDER / "Anomaly-Videos" / anom
    for fp in folder.iterdir():
        name = fp.name
        if name not in dict_test[anom]:
            list_train.append(anom+"/"+name)

for fp in (PARENT_FOLDER/"Training-Normal-Videos").iterdir():
    name = fp.name
    list_train.append("Training-Normal-Videos/"+name)

with orig_split_train.open("w") as fp:
    fp.writelines("\n".join(sorted(list_train)))