import os
import re
from pathlib import Path
from utils import get_num_frame
import pickle


_HOME = os.environ['HOME']
DATA_HOME = Path(_HOME + '/dataset/UCF_crime/')
ANOM_DIR = DATA_HOME / "Anomaly-Videos"
TEST_NORM_DIR = DATA_HOME / "Testing_Normal_Videos_Anomaly"
TEMP_ANN_FILE = Path('./Temporal_Anomaly_Annotation.txt')

dict_frame_num = {}

with TEMP_ANN_FILE.open("r") as fp:
    for line in fp:
        try:
            vid_name = line.split(" ")[0]
        except Exception:
            continue

        if str(vid_name).startswith('Normal'):  # normal video
            vid_path = TEST_NORM_DIR / vid_name
            assert vid_path.exists()
        else:  # anomaly video
            anom_type = re.split('[^a-zA-Z]', vid_name)[0]
            vid_path = ANOM_DIR / anom_type / vid_name
            assert vid_path.exists()
        num_frames = get_num_frame(vid_path)
        dict_frame_num[str(vid_name)] = num_frames

with open("frame_num.pkl", "wb") as pkl:
    pickle.dump(dict_frame_num, pkl)
