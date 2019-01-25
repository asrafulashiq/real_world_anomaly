""" Split all videos into training and testing
"""

#%%
from pathlib import Path
from random import shuffle
import os



_HOME = os.environ["HOME"]
PARENT_FOLDER = Path(_HOME+"/dataset/UCF_crime")

ANOM_FOLDER = PARENT_FOLDER / 'Anomaly-Videos/'
TRAIN_NORMAL_FOLDER = PARENT_FOLDER / 'Training-Normal-Videos/'
TEST_NORMAL_FOLDER = PARENT_FOLDER / 'Testing_Normal_Videos_Anomaly/'

LABEL_ANOMS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
               'Explosion', 'Fighting', 'RoadAccidents', 'Robbery',
               'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
print(len(LABEL_ANOMS))

PERCENT_TEST_TO_TOTAL = 0.15  # 15 percent for testing from total videos



def refine_path_to_3d_path(file_path, is_normal=True, feature_name="C3D_features"):
    """convert file path to the path of 3d feature"""
    file_path = file_path.resolve()
    _parts = file_path.parts
    if is_normal:
        base_path = PARENT_FOLDER / feature_name / \
                "Avg" / _parts[-2] / (_parts[-1]+'.npy')
    else:
        base_path = PARENT_FOLDER / feature_name / \
            "Avg" / _parts[-3] / (_parts[-1]+'.npy')
    assert base_path.exists()
    return str(base_path)


#%%
"""Split anomalous videos
"""
train_split_path = []
test_split_path = []

for anomaly_type in LABEL_ANOMS:
    type_folder = ANOM_FOLDER / f'{anomaly_type}'
    assert type_folder.exists()
    # print(type_folder)

    all_segments = list(type_folder.iterdir())

    # split into train test
    shuffle(all_segments)
    test_idx = int(len(all_segments) * 0.15)
    test_segments = all_segments[:test_idx]
    train_segments = all_segments[test_idx:]

    train_split_path.extend(
        [refine_path_to_3d_path(seg, is_normal=False) for seg
            in train_segments if seg.exists()]
    )

    test_split_path.extend(
        [refine_path_to_3d_path(seg, is_normal=False) for seg
            in test_segments if seg.exists()]
    )

print("Abnormal:")
print(f" Test: {len(test_split_path)}")
print(f" Train: {len(train_split_path)}")


"""Split Normal event
"""

train_split_path.extend(
    [refine_path_to_3d_path(seg, is_normal=True) for seg
     in TRAIN_NORMAL_FOLDER.iterdir() if seg.exists()]
)

test_split_path.extend(
    [refine_path_to_3d_path(seg, is_normal=True) for seg
     in TEST_NORMAL_FOLDER.iterdir() if seg.exists()]
)

print("Total:")
print(f" Test: {len(test_split_path)}")
print(f" Train: {len(train_split_path)}")

"""write to file
"""
split_folder = PARENT_FOLDER / "custom_split"
split_folder.mkdir(exist_ok=True)

file_train = split_folder / 'Custom_train_split.txt'
file_test = split_folder / 'Custom_test_split.txt'

with file_train.open('w') as train_file:
    for item in train_split_path:
        train_file.write(f"{item}\n")

with file_test.open('w') as test_file:
    for item in test_split_path:
        test_file.write(f"{item}\n")
