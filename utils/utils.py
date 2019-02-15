import cv2
import os
import numpy as np
from sklearn.metrics import roc_auc_score


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


def get_frames_seg(num_frames, seg, frames_per_seg=16):
    num_feat = num_frames // frames_per_seg
    div_ind = divide_frames_seg(num_feat, seg)
    ind_array = []
    for counter, ishots in enumerate(div_ind):
        ss, ee = ishots
        ind = (ss*frames_per_seg, (ee+1)*frames_per_seg-1)
        ind_array.append(ind)
    ind_array[-1] = (ind_array[-1][0], num_frames-1)
    return ind_array


def get_frames_32_seg(num_frames, seg, frames_per_seg=16):
    """ get indices when equally divide 32 segments """
    # TODO: Check if it can be modified
    num_feat = num_frames // frames_per_seg
    thirty2_shots = np.round(
        np.linspace(0, num_feat-1, seg+1)
    ).astype(np.int)
    ind_array = []
    for counter, ishots in enumerate(range(len(thirty2_shots)-1)):
        ss = thirty2_shots[ishots]
        ee = thirty2_shots[ishots+1] - 1

        if ss == ee or ss > ee:
            ind = (ss*frames_per_seg, (ss+1)*frames_per_seg-1)
        else:
            ind = (ss*frames_per_seg, (ee+1)*frames_per_seg-1)
        ind_array.append(ind)
    ind_array[-1] = (ind_array[-1][0], num_frames-1)
    return ind_array


def get_num_frame(vid_file):
    """get the number of frames in a video

    Arguments:
        vid_file {string/pathlib.Path} -- video file name
    """
    vid_file = str(vid_file)

    assert os.path.exists(vid_file), \
        "file (%s) not found".format(vid_file)

    cap = cv2.VideoCapture(vid_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


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


def load_weights_from_mat(model, weight_path):
    """Function to load the model weights"""
    from scipy.io import loadmat
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model


def save_model(model, json_path=None, weight_path=None):
    """function to save model and weight"""
    if json_path:
        model_json = model.to_json()
        with open(json_path, 'w') as fp:
            fp.write(model_json)
    if weight_path:
        model.save_weights(weight_path)


def valid_res(list_gt_ind, list_gt_pred, list_vid_path, dict_frame, seg=32):
    """calculate validation result """
    all_score_pred = np.array([])
    all_score_gt = np.array([])
    for gt_ind, gt_pred, vid_path in zip(list_gt_ind, list_gt_pred,
                                         list_vid_path):
        num_frames = dict_frame[vid_path]
        indices = get_frames_32_seg(num_frames, seg)
        score_pred = np.zeros(num_frames)
        for counter, ind in enumerate(indices):
            start_ind = ind[0]
            end_ind = ind[1]
            score_pred[start_ind:end_ind+1] = gt_pred[counter]

        all_score_pred = np.concatenate(
            (all_score_pred, score_pred)
        )

        # get gt score for each frame
        score_gt = np.zeros(num_frames)
        for counter, ind in enumerate(gt_ind):
            start_ind = ind[0]
            end_ind = ind[1]
            score_gt[start_ind:end_ind+1] = 1

        all_score_gt = np.concatenate(
            (all_score_gt, score_gt)
        )

    roc_auc = roc_auc_score(all_score_gt, all_score_pred)
    return roc_auc
