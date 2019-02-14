"""
Make 32 segments to the whole video
"""
import os
import numpy as np
import pickle
from numpy.linalg import norm
from tqdm import tqdm
from pathlib2 import Path


def divide_frames_seg(num_frames, seg):
    """ get total frames into seg segments """
    ind = []
    for i in range(seg):
        b = int(np.floor(max(0, i * num_frames/seg)))
        e = int(np.floor(max(0, (i+1) * num_frames/seg)))
        if e < b:
            e = b
        ind.append((b, e))
    return ind

_HOME = os.environ['HOME']
PARENT_DATA_FOLDER = Path(_HOME + '/dataset/UCF_crime')
# sys.path.append("/home/islama6a/local/pytorch/build")

feature_name = "C3D_features"
layer_name = 'fc6'

FEATURE_3D_PATH = PARENT_DATA_FOLDER / feature_name
FEATURE_3D_PATH_SEG = PARENT_DATA_FOLDER / feature_name / "seg_500"

FEATURE_3D_PATH_SEG.mkdir(parents=True, exist_ok=True)

if feature_name == "C3D_features":
    dim_features = 4096  # dimension of 3d feature
else:
    dim_features = 512
    layer_name = 'final_avg'

SEG = 500  # number of segments in a video clip

for ifolder in FEATURE_3D_PATH.iterdir():
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

        n_dim = data.shape[0]  # original number of segments
        seg_data = np.zeros((SEG, dim_features))
        indices = divide_frames_seg(n_dim, SEG)
        for counter, ishots in enumerate(indices):
            ss, ee = ishots
            if ss == ee:
                tmp_vector = data[ss]
            else:
                tmp_vector = np.mean(data[ss:ee+1, :], axis=0)

            # tmp_vector = tmp_vector
            seg_data[counter, :] = tmp_vector

            if np.any(np.isnan(seg_data)) or np.any(np.isinf(seg_data)):
                raise ValueError, "data contain nan/inf"

        # Write C3D features in text file
        saved_file = saved_path / (fp.stem+".npy")
        with saved_file.open(mode='wb') as target:
            np.save(target, seg_data)
