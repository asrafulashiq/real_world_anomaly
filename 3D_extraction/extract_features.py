import os
from helper import *
from pathlib2 import Path

PARENT_FOLDER = Path('/media/ash/New Volume/dataset/UCF_crime/')
ANOM_FOLDER = PARENT_FOLDER / 'Anomaly-Videos'

assert PARENT_FOLDER.exists()

# extract anomaly folder
FEAT_PARENT_FOLDER = PARENT_FOLDER / '3D_features'
FEAT_ANOM_FOLDER = FEAT_PARENT_FOLDER / 'Anomaly-Videos'

FEAT_ANOM_FOLDER.mkdir(parents=True, exist_ok=True)

for anom in ANOM_FOLDER.iterdir():
    anom_type = anom.name

    # create feature folder for this type

    for vid_file in anom.iterdir():
        vid_file_name = vid_file.name

        # create temporary csv file tmp.csv
        write_csv_for_lmdb(vid_file_name)

        # extract lmdb file format
        # TODO:
        # extract features
        feat_path = FEAT_ANOM_FOLDER / (vid_file_name+'.pkl')
        # TODO

