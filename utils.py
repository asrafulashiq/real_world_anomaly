import random
import numpy as np


def load_model(json_path, weight_path=None):
    """function to load model [and weight]"""
    from keras.models import model_from_json

    with open(json_path, 'r') as fp:
        model = model_from_json(fp.read())
    if weight_path:
        model.load_weights(weight_path)
    return model


def load_weights(model, weight_path):
    """function to load weights"""
    model.load_weights(weight_path)
    return model


def save_model(model, json_path=None, weight_path=None):
    """function to save model and weight"""
    if json_path:
        model_json = model.to_json()
        with open(json_path, 'w') as fp:
            fp.write(model_json)
    if weight_path:
        model.save_weights(weight_path)


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

    Data = np.zeros((batch_size * segment_size, feat_size), dtype=np.float)
    Labels = np.zeros(batch_size * segment_size, dtype=np.float)

    for i in range(batch_size//2):
        # load abnormal video
        feat_normal = np.load(open(sampled_abnormal_list[i], 'rb'))
        # size : segment_size * feat_size

        feat_abnormal = np.load(open(sampled_normal_list[i], 'rb'))

        Data[i*2*segment_size: (i*2*segment_size + segment_size)] = \
            feat_abnormal
        Data[(i*2*segment_size + segment_size): (i+1)*2*segment_size] = \
            feat_normal

        Labels[i*2*segment_size: (i*2*segment_size + segment_size)] = 1

    return Data, Labels


if __name__ == "__main__":
    # test data loading
    abnormal_path = '/home/islama6a/local/UCF_crime/\
        custom_split/Custom_train_split_mini_abnormal.txt'
    normal_path = '/home/islama6a/local/UCF_crime/\
        custom_split/Custom_train_split_mini_normal.txt'

    data, labels = load_dataset_batch(abnormal_path, normal_path)

    assert data.shape == (60*32, 4095), "data shape does not match"
    assert labels.shape == (60*32,), "label shape does not match"
