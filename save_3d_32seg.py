from pathlib2 import Path
import os
import numpy as np
import pickle
from numpy.linalg import norm
import platform
from tqdm import tqdm


if platform.node() == 'ash-ubuntu':
    PARENT_DATA_FOLDER = Path('/media/ash/New Volume/dataset/UCF_crime/')
else:
    PARENT_DATA_FOLDER = Path('/home/islama6a/local/UCF_crime')
    import sys
    sys.path.append("/home/islama6a/local/pytorch/build")

FEATURE_3D_PATH = PARENT_DATA_FOLDER / "C3D_features"
FEATURE_3D_PATH_SEG = PARENT_DATA_FOLDER / "C3D_features/Avg"

FEATURE_3D_PATH_SEG.mkdir(parents=True, exist_ok=True)

dim_features = 4096  # dimension of 3d feature
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
            data = pickle.load(f)['fc6']
        assert data.shape[1] == dim_features

        n_dim = data.shape[0]  # original number of segments
        seg_data = np.zeros((SEG, dim_features))

        thirty2_shots = np.round(np.linspace(0, n_dim-1, SEG+1)).astype(np.int)
        for counter, ishots in enumerate(range(len(thirty2_shots)-1)):
            ss = thirty2_shots[ishots]
            ee = thirty2_shots[ishots+1] - 1

            tmp_vector = np.zeros(SEG)

            if ss == ee or ss > ee:
                tmp_vector = data[ss]
            else:
                tmp_vector = np.mean(data[ss:ee+1, :], axis=0)
            tmp_vector = tmp_vector / norm(tmp_vector)
            seg_data[counter, :] = tmp_vector

            if np.any(np.isnan(seg_data)) or np.any(np.isinf(seg_data)):
                raise ValueError, "data contain nan/inf"

        # Write C3D features in text file to load in
        # Training_AnomalyDetector_public
        saved_file = saved_path / (fp.stem+".npy")
        with saved_file.open(mode='wb') as target:
            np.save(target, seg_data)
