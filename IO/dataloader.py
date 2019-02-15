import random
import numpy as np
import os
from pathlib import Path
import pandas as pd


np.random.seed(0)


def tmp_load(abnormal_list_path,
             normal_list_path, batch_size=60,
             segment_size=32, feat_size=4096):
    """load abnormal and normal video for a batch.
    return:
        data: (batch_size, 32, 4096)
        lables: (batch_size, 32, 6)
            The first value is abnormality score, next 5
            values are one hot encoding of classes
    """
    import re
    from sklearn.preprocessing import LabelEncoder
    # from keras.utils import to_categorical

    prog = re.compile('[^a-zA-Z]')

    abnormal_list = []
    normal_list = []
    _categories = []

    with open(abnormal_list_path, "r") as fp:
        for line in fp:
            line = line.rstrip()
            abnormal_list.append(line)
            name = line.split(os.path.sep)[-1]
            _categories.append(prog.split(name)[0])

    _categories.append("Normal")

    encoder = LabelEncoder()
    # label_encoded = encoder.fit_transform(_categories)

    with open(normal_list_path, "r") as fp:
        normal_list = [line.rstrip() for line in fp]

    abnormal_list = np.array(abnormal_list)
    normal_list = np.array(normal_list)
    _categories = np.array(_categories)

    while True:
        abn_indices = np.random.choice(len(abnormal_list), batch_size//2)
        sampled_abnormal_list = abnormal_list[abn_indices]
        # sampled_abnormal_labels = label_encoded[abn_indices]

        sampled_normal_list = np.random.choice(normal_list, batch_size//2)

        data = np.zeros((batch_size, segment_size, feat_size), dtype=np.float)
        # label_1 = np.zeros((batch_size, segment_size, 1), dtype=np.float)
        # label_2 = np.zeros((batch_size, segment_size,
                            # len(encoder.classes_) - 1), dtype=np.float)

        label_1 = np.zeros((batch_size, 1), dtype=np.float)
        label_2 = np.ones((batch_size, segment_size, 1), dtype=np.float)

        paths = []

        for i in range(batch_size//2):
            feat_abnormal = np.load(open(sampled_abnormal_list[i], 'rb'))
            feat_normal = np.load(open(sampled_normal_list[i], 'rb'))

            data[i] = feat_abnormal / np.linalg.norm(feat_abnormal)
            data[batch_size//2 + i] = feat_normal / np.linalg.norm(feat_normal)

            label_1[i] = 1
            label_1[batch_size//2 + i] = 0

            # label_2[i*2, :, int(sampled_abnormal_labels[i])] = 1
            # label_2[i*2+1, :, int(sampled_abnormal_labels[-1])] = 1
            paths.append(sampled_abnormal_list[i])
            paths.append(sampled_normal_list[i])

        paths = [paths[i] for i in range(0, batch_size, 2)] +\
                [paths[i] for i in range(1, batch_size, 2)]
        # labels = np.concatenate((label_1, label_2), axis=-1)
        labels = [label_1, label_2]
        yield data, labels, paths, encoder


def load_dataset_batch_with_segment_tcn(abnormal_list_path,
                                        normal_list_path, batch_size=60,
                                        segment_size=32, feat_size=4096):
    """load abnormal and normal video for a batch.
    return:
        data: (batch_size, 32, 4096)
        lables: (batch_size, 32, 6)
            The first value is abnormality score, next 5
            values are one hot encoding of classes
    """
    import re
    from sklearn.preprocessing import LabelEncoder
    # from keras.utils import to_categorical

    prog = re.compile('[^a-zA-Z]')

    abnormal_list = []
    normal_list = []
    _categories = []

    with open(abnormal_list_path, "r") as fp:
        for line in fp:
            line = line.rstrip()
            abnormal_list.append(line)
            name = line.split(os.path.sep)[-1]
            _categories.append(prog.split(name)[0])

    _categories.append("Normal")

    encoder = LabelEncoder()
    label_encoded = encoder.fit_transform(_categories)

    with open(normal_list_path, "r") as fp:
        normal_list = [line.rstrip() for line in fp]

    abnormal_list = np.array(abnormal_list)
    normal_list = np.array(normal_list)
    _categories = np.array(_categories)

    while True:
        abn_indices = np.random.choice(len(abnormal_list), batch_size//2)
        sampled_abnormal_list = abnormal_list[abn_indices]
        sampled_abnormal_labels = label_encoded[abn_indices]

        sampled_normal_list = np.random.choice(normal_list, batch_size//2)

        data = np.zeros((batch_size, segment_size, feat_size), dtype=np.float)
        label_1 = np.zeros((batch_size, segment_size, 1), dtype=np.float)
        label_2 = np.zeros((batch_size, segment_size,
                            _categories.classes_ - 1), dtype=np.float)

        for i in range(batch_size//2):
            feat_abnormal = np.load(open(sampled_abnormal_list[i], 'rb'))
            feat_normal = np.load(open(sampled_normal_list[i], 'rb'))

            data[i*2] = feat_abnormal
            data[i*2+1] = feat_normal

            label_1[i*2] = 1
            label_1[i*2+1] = 0

            label_2[i*2, :, int(sampled_abnormal_labels[i])] = 1
            # label_2[i*2+1, :, int(sampled_abnormal_labels[-1])] = 1

        labels = np.concatenate((label_1, label_2), axis=-1)
        yield data, labels


def load_dataset_batch(abnormal_list_path, normal_list_path,
                       batch_size=60, segment_size=32, feat_size=4096):
    """load abnormal and normal video for a batch.
    for each type, feature size will be \
    batch_size/2 * segment_size * [feature_size]
    """

    with open(abnormal_list_path, "r") as fp:
        abnormal_list = [line.rstrip() for line in fp]

    with open(normal_list_path, "r") as fp:
        normal_list = [line.rstrip() for line in fp]

    sampled_normal_list = random.sample(normal_list, batch_size//2)
    sampled_abnormal_list = random.sample(abnormal_list, batch_size//2)

    data = np.zeros((batch_size * segment_size, feat_size), dtype=np.float)
    labels = np.zeros(batch_size * segment_size, dtype=np.float)

    for i in range(batch_size//2):
        # load abnormal video
        feat_abnormal = np.load(open(sampled_abnormal_list[i], 'rb'))
        # size : segment_size * feat_size

        feat_normal = np.load(open(sampled_normal_list[i], 'rb'))

        data[i*2*segment_size: (i*2*segment_size + segment_size)] = \
            feat_abnormal
        data[(i*2*segment_size + segment_size): (i+1)*2*segment_size] = \
            feat_normal

        labels[i*2*segment_size: (i*2*segment_size + segment_size)] = 1

    return data, labels


def load_dataset_batch_with_segment(abnormal_list_path,
                                    normal_list_path, batch_size=60,
                                    segment_size=32, feat_size=4096):
    """load abnormal and normal video for a batch.
    for each type, feature size will be \
    batch_size/2 * segment_size * [feature_size]
    """

    with open(abnormal_list_path, "r") as fp:
        abnormal_list = [line.rstrip() for line in fp]

    with open(normal_list_path, "r") as fp:
        normal_list = [line.rstrip() for line in fp]

    while True:

        sampled_normal_list = random.sample(normal_list, batch_size//2)
        sampled_abnormal_list = random.sample(abnormal_list, batch_size//2)

        data = np.zeros((batch_size, segment_size, feat_size), dtype=np.float)
        # labels = np.zeros(batch_size, dtype=np.float)

        for i in range(batch_size//2):
            # load abnormal video
            feat_abnormal = np.load(open(sampled_abnormal_list[i], 'rb'))
            # size : segment_size * feat_size

            feat_normal = np.load(open(sampled_normal_list[i], 'rb'))

            data[i*2] = feat_abnormal
            data[i*2+1] = feat_normal

            # labels[i*2*segment_size: (i*2*segment_size + segment_size)] = 1

        labels = np.tile(
            np.array([1, 0], dtype=np.float),
            batch_size // 2
        )

        yield data, labels


def load_one_video(test_file, log=None, normalize=True):
    """ get one video features """
    with open(test_file, "r") as fp:
        test_list = [line.rstrip() for line in fp]
    np.random.seed(0)
    np.random.shuffle(test_list)

    for f_vid in test_list:
        if log:
            log.info(f'test for {f_vid}')
        feature = np.load(open(f_vid, "rb"))
        if normalize:
            feature = feature / np.linalg.norm(feature)
        _path = Path(f_vid).stem
        # return video name and features
        yield _path, feature


def get_ind_from_pd(df):
    # print(df.head())
    indices = list(df.iloc[0, 2:])
    ret_ind = []
    for k in range(0, len(indices)-1, 2):
        if indices[k] == -1:
            continue
        ret_ind.append((indices[k], indices[k+1]))
    return ret_ind


def load_valid_batch(valid_list_file, tmp_ann_file, normalize=True):
    """get validation data of batch size batch"""

    df_temp_ann = pd.read_csv(
        tmp_ann_file,
        delimiter=" ",
        header=None,
        skipinitialspace=True
    )

    with open(valid_list_file, 'r') as fp:
        valid_list = sorted([i.rstrip() for i in fp])

    for vid_file in valid_list:
        valid_feat = np.load(open(vid_file, 'rb'))
        valid_feat = valid_feat / np.linalg.norm(valid_feat)
        vname = Path(vid_file).stem
        # vid_name = vid_file.stem
        x_row = df_temp_ann.loc[df_temp_ann[0] == vname]
        gt_ind = get_ind_from_pd(x_row)

        yield vname, gt_ind, valid_feat


if __name__ == "__main__":
    # test data loading
    _HOME = os.environ['HOME']
    abnormal_path = _HOME + '/dataset/UCF_crime/' +\
        'custom_split/Custom_train_split_mini_abnormal.txt'
    normal_path = _HOME + '/dataset/UCF_crime/' +\
        'custom_split/Custom_train_split_mini_normal.txt'

    data, labels = load_dataset_batch(abnormal_path, normal_path)

    assert data.shape == (60*32, 4096), "data shape does not match"
    assert labels.shape == (60*32,), "label shape does not match"

    test_path = _HOME + '/dataset/UCF_crime/' +\
        'custom_split/Custom_test_split_mini.txt'
    test_data = load_one_video(test_path)
    for x in test_data:
        assert x.shape == (32, 4096), "shape does nt match"
        break