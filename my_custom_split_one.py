""" Split all videos into training and testing
"""


from pathlib import Path
import random
import os
from collections import defaultdict


MINI = '_mini'

_HOME = os.environ["HOME"]
PARENT_FOLDER = Path(_HOME+"/dataset/UCF_crime")

orig_split_train = PARENT_FOLDER / "Anomaly_Detection_splits/Anomaly_Train.txt"
orig_split_test = PARENT_FOLDER / "Anomaly_Detection_splits/Anomaly_Test.txt"

feature_name = "C3D"
feature_folder = PARENT_FOLDER / (feature_name+'_features') / "Avg"
split_folder = PARENT_FOLDER / ("one_custom_split_"+feature_name)
split_folder.mkdir(exist_ok=True)

# LABEL_ANOMS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
#                'Explosion', 'Fighting', 'RoadAccidents', 'Robbery',
#                'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

# LABEL_ANOMS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary']
# LABEL_ANOMS = ['Abuse', 'Arrest', 'Assault', 'Fighting', 'Arson']
LABEL_ANOMS = ['Arson', 'Assault']

DOWN_RATIO = 5./13  # None
NORM_LEN = 80


""" create dict """
dict_test = defaultdict(list)
dict_train = defaultdict(list)
DICT_LABELS = LABEL_ANOMS + ['Testing_Normal_Videos_Anomaly',
                             'Training-Normal-Videos']

with orig_split_test.open("r") as fp:
    for line in fp:
        line = line.strip()
        line_split = line.split("/")
        if line_split[0] in DICT_LABELS:
            dict_test[line_split[0]].append(line_split[1])


with orig_split_train.open("r") as fp:
    for line in fp:
        line = line.strip()
        line_split = line.split("/")
        if line_split[0] in DICT_LABELS:
            dict_train[line_split[0]].append(line_split[1])


if DOWN_RATIO is not None:
    for _dict in (dict_test, dict_train):
        normal_keys = ['Testing_Normal_Videos_Anomaly',
                       'Training-Normal-Videos']
        for k in normal_keys:
            if k in _dict:
                new_len = NORM_LEN  #int(DOWN_RATIO * len(_dict[k]))
                _dict[k] = random.sample(
                    _dict[k], new_len
                )


# save
anom_train = []
normal_train = []
all_test = []

for k in dict_train:
    if k in LABEL_ANOMS:
        fnames = [
            str(feature_folder / "Anomaly-Videos" / (v+".npy"))
            for v in dict_train[k]
        ]
        # assert fname.exists()
        anom_train.extend(fnames)
    else:
        fnames = [
            str(feature_folder / k / (v+".npy"))
            for v in dict_train[k]
        ]
        normal_train.extend(fnames)


for k in dict_test:
    if k in LABEL_ANOMS:
        fnames = [
            str(feature_folder / "Anomaly-Videos" / (v+".npy"))
            for v in dict_test[k]
        ]
        # assert fname.exists()
        all_test.extend(fnames)
    else:
        fnames = [
            str(feature_folder / k / (v+".npy"))
            for v in dict_test[k]
        ]
        all_test.extend(fnames)


print("Train:")
print(f" Anomaly: {len(anom_train)}")
print(f" Normal: {len(normal_train)}")
print(f"Test: {len(all_test)}")

file_train_anom = split_folder / ('Custom_train_split'+MINI+'_abnormal.txt')
file_train_normal = split_folder / ('Custom_train_split'+MINI+'_normal.txt')
file_test = split_folder / ('Custom_test_split'+MINI+'.txt')

with file_train_anom.open("w") as fp:
    fp.writelines(os.linesep.join(anom_train))
with file_train_normal.open("w") as fp:
    fp.writelines(os.linesep.join(normal_train))
with file_test.open("w") as fp:
    fp.writelines(os.linesep.join(all_test))
