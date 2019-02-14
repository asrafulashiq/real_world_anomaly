"""
save number of frame off all videos
"""

import os
import re
from pathlib import Path
import cv2
import pickle
from collections import defaultdict
from tqdm import tqdm


def get_num_frame(vid_file):
    """get the number of frames in a video

    Arguments:
        vid_file {string/pathlib.Path} -- video file name
    """
    vid_file = str(vid_file)

    assert os.path.exists(vid_file), \
        "file (%s) not found".format(vid_file)

    cap = cv2.VideoCapture(vid_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


do_overwrite = False
_HOME = os.environ['HOME']
PARENT_FOLDER = Path(_HOME) / 'dataset' / 'UCF_crime'
ANOM_FOLDER = PARENT_FOLDER / 'Anomaly-Videos'
assert PARENT_FOLDER.exists()

Data = defaultdict(int)

video_list = []
save_dir = []

for anom in ANOM_FOLDER.iterdir():
    for vid_file in tqdm(sorted(anom.iterdir())):
        vid_file_name = vid_file.name
        num = get_num_frame(vid_file)
        Data[vid_file_name] = num

# extract normal folder
normal_test_train = ['Training-Normal-Videos',
                        'Testing_Normal_Videos_Anomaly']

for normal_folder in normal_test_train:
    normal = PARENT_FOLDER / normal_folder

    for vid_file in tqdm(sorted(normal.iterdir())):
        vid_file_name = vid_file.name
        num = get_num_frame(vid_file)
        Data[vid_file_name] = num

with open("frame_num.pkl", "wb") as fp:
    pickle.dump(Data, fp)
