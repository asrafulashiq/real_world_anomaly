from scipy.io import loadmat, savemat
from keras.models import model_from_json
from pathlib import Path
import random
import numpy as np
import logging


def load_model(json_path):  # Function to load the model
    model = model_from_json(open(json_path).read())
    return model


def load_weights(model, weight_path):  # Function to load the model weights
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model


def conv_dict(dict2):
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict


def save_model(model, json_path, weight_path):  # Function to save the model
    json_string = model.to_json()
    open(json_path, 'w').write(json_string)
    dict = {}
    i = 0
    for layer in model.layers:
        weights = layer.get_weights()
        my_list = np.zeros(len(weights), dtype=np.object)
        my_list[:] = weights
        dict[str(i)] = my_list
        i += 1
    savemat(weight_path, dict)



def load_dataset_batch(abnormal_list_path, normal_list_path, batch_size=60,
                    segment_size=32, feat_size = 4096):
    """load abnormal and normal video for a batch.
    for each type, feature size will be batch_size/2 * segment_size * [feature_size]
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
        feat_normal = np.load(open(sampled_abnormal_list[i], 'rb')) # size : segment_size * feat_size
        feat_abnormal = np.load(open(sampled_normal_list[i], 'rb')) # "

        Data[i*2*segment_size: (i*2*segment_size + segment_size)] = feat_abnormal
        Data[(i*2*segment_size + segment_size): (i+1)*2*segment_size] = feat_abnormal

        Labels[i*2*segment_size: (i*2*segment_size + segment_size)] = 1

    return Data, Labels

if __name__ == "__main__":

    # test data loading
    abnormal_path = '/home/islama6a/local/UCF_crime/custom_split/Custom_train_split_mini_abnormal.txt'
    normal_path = '/home/islama6a/local/UCF_crime/custom_split/Custom_train_split_mini_normal.txt'

    data, labels = load_dataset_batch(abnormal_path, normal_path)

    assert data.shape == (60*32, 4095), "data shape does not match"
    assert labels.shape == (60*32,), "label shape does not match"