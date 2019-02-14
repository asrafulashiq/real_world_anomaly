from pathlib2 import Path
import os
import numpy as np
import pickle
from numpy.linalg import norm
import platform
from tqdm import tqdm


_HOME = os.environ['HOME']
PARENT_DATA_FOLDER = Path(_HOME + '/dataset/UCF_crime')
# sys.path.append("/home/islama6a/local/pytorch/build")

feature_name = "C3D_features"
layer_name = 'fc6'

FEATURE_3D_PATH = PARENT_DATA_FOLDER / feature_name
FEATURE_3D_PATH_SEG = PARENT_DATA_FOLDER / feature_name / "npy"

FEATURE_3D_PATH_SEG.mkdir(parents=True, exist_ok=True)

if feature_name == "C3D_features":
    dim_features = 4096  # dimension of 3d feature
else:
    dim_features = 512
    layer_name = 'final_avg'

SEG = 32  # number of segments in a video clip

for ifolder in FEATURE_3D_PATH.iterdir():  # ifolder contains 'Anomaly-Videos', 'Train-Normal', 'Test-Nomral'
    # get pkl file for a particular video clip
    all_files = [fp for fp in ifolder.iterdir() if fp.suffix == '.pkl']
    data = None
    saved_path = FEATURE_3D_PATH_SEG / ifolder.name
    saved_path.mkdir(exist_ok=True)

    for fp in tqdm(all_files):
        # read pkl file into 4096 dimensional numpy array
        with fp.open(mode='rb') as f:
            data = pickle.load(f)[layer_name]
            data = data.squeeze()
        assert data.shape[1] == dim_features

        saved_file = saved_path / (fp.stem+".npy")
        with saved_file.open(mode='wb') as target:
            np.save(target, data)
