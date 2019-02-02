import random
import numpy as np
import os
from pathlib import Path


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

    return data, labels


def load_one_video(test_file, log=None):
    """ get one video features """
    with open(test_file, "r") as fp:
        test_list = [line.rstrip() for line in fp]
    np.random.seed(0)
    np.random.shuffle(test_list)

    for f_vid in test_list:
        if log:
            log.info(f'test for {f_vid}')
        feature = np.load(open(f_vid, "rb"))

        # return video name and features
        yield Path(f_vid).stem, feature


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
