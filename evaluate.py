""" Evaluate AUC for testing videos """

from pathlib import Path
import pickle
import os
import re
from utils import get_num_frame, get_frames_32_seg
import numpy as np
import pandas as pd


_HOME = os.environ['HOME']
PRED_PATH = Path('./results/predictions/')
DATA_HOME = Path(_HOME + '/dataset/UCF_crime/')
ANOM_DIR = DATA_HOME / "Anomaly-Videos"
TEST_NORM_DIR = DATA_HOME / "Testing_Normal_Videos_Anomaly"
SEG = 32
TEMP_ANN_FILE = './Temporal_Anomaly_Annotation.txt'
df_temp_ann = pd.read_csv(
    TEMP_ANN_FILE,
    delimiter=" ",
    header=None,
    skipinitialspace=True
)


def get_ind_from_pd(df):
    # print(df.head())
    indices = list(df.iloc[0, 2:])
    ret_ind = []
    for k in range(0, len(indices)-1, 2):
        if indices[k] == -1:
            continue
        ret_ind.append((indices[k], indices[k+1]))
    return ret_ind


all_score_pred = np.array([])
all_score_gt = np.array([])

for pred_file in PRED_PATH.iterdir():
    if pred_file.suffix != '.pkl':
        continue
    with pred_file.open('rb') as fp:
        _pred = pickle.load(fp)

    # search this pred_file video
    vid_name = pred_file.stem  # remove suffix from file name

    if str(vid_name).startswith('Normal'):  # normal video
        vid_path = TEST_NORM_DIR / vid_name
        assert vid_path.exists()
    else:  # anomaly video
        anom_type = re.split('[^a-zA-Z]', vid_name)[0]
        vid_path = ANOM_DIR / anom_type / vid_name
        assert vid_path.exists()

    # get frame number from video
    num_frames = get_num_frame(vid_path)
    indices = get_frames_32_seg(num_frames, SEG)

    # get prediction for each frame
    score_pred = np.zeros(num_frames)
    for counter, ind in enumerate(indices):
        start_ind = ind[0]
        end_ind = ind[1]
        score_pred[start_ind:end_ind+1] = _pred[counter]

    all_score_pred = np.concatenate(
        (all_score_pred, score_pred)
    )

    # get gt score for each frame
    score_gt = np.zeros(num_frames)
    x_row = df_temp_ann.loc[df_temp_ann[0] == vid_name]
    gt_ind = get_ind_from_pd(x_row)

    for counter, ind in enumerate(gt_ind):
        start_ind = ind[0]
        end_ind = ind[1]
        score_gt[start_ind:end_ind+1] = 1
    all_score_gt = np.concatenate(
        (all_score_gt, score_gt)
    )