""" Split all videos into training and testing
"""


from pathlib import Path
from random import shuffle
from random import sample
import os


_HOME = os.environ["HOME"]
PARENT_FOLDER = Path(_HOME+"/dataset/UCF_crime")

ANOM_FOLDER = PARENT_FOLDER / 'Anomaly-Videos/'
TRAIN_NORMAL_FOLDER = PARENT_FOLDER / 'Training-Normal-Videos/'
TEST_NORMAL_FOLDER = PARENT_FOLDER / 'Testing_Normal_Videos_Anomaly/'

split_folder = PARENT_FOLDER / "custom_split"
split_folder.mkdir(exist_ok=True)

# LABEL_ANOMS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
#                'Explosion', 'Fighting', 'RoadAccidents', 'Robbery',
#                'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

LABEL_ANOMS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary']

DOWN_RATIO = 5. / 13

print(len(LABEL_ANOMS))


def refine_path_to_3d_path(file_path, is_normal=True,
                           feature_name="C3D_features"):
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


def extract_random_files(folder_name, rat=0.2):
    """sample files of a folder into

    Arguments:
        folder_name {string} -- folder name that contains the files

    Keyword Arguments:
        rat {float} -- ration of the first set to the total number of files
        (default: {0.2})
    """
    if type(folder_name) == str:
        folder_name = Path(folder_name)
    assert folder_name.exists()

    all_segments = list(folder_name.iterdir())  # list of all video files
    down_segments = sample(all_segments, int(len(all_segments) * rat))
    return down_segments


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


file_train = split_folder / 'Custom_train_split_mini_abnormal.txt'
# file_test = split_folder / 'Custom_test_split_mini_abnormal.txt'

with file_train.open('w') as train_file:
    for item in train_split_path:
        train_file.write(f"{item}\n")

# with file_test.open('w') as test_file:
#     for item in test_split_path:
#         test_file.write(f"{item}\n")


"""Split Normal event
"""

test_normal = extract_random_files(TEST_NORMAL_FOLDER, DOWN_RATIO)
train_normal = extract_random_files(TRAIN_NORMAL_FOLDER, DOWN_RATIO)

train_split_path = [refine_path_to_3d_path(seg, is_normal=True) for seg
                    in train_normal if seg.exists()]

test_split_path.extend([refine_path_to_3d_path(seg, is_normal=True) for seg
                   in test_normal if seg.exists()])


print("Total:")
print(f" Test: {len(test_split_path)}")
print(f" Train: {len(train_split_path)}")


file_train = split_folder / 'Custom_train_split_mini_normal.txt'
file_test = split_folder / 'Custom_test_split_mini.txt'

with file_train.open('w') as train_file:
    for item in train_split_path:
        train_file.write(f"{item}\n")

with file_test.open('w') as test_file:
    for item in test_split_path:
        test_file.write(f"{item}\n")
