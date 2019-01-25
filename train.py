import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Reshape
from keras.regularizers import l2
from keras.optimizers import SGD, adam, Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
# import theano.tensor as T
from keras import backend as K
# import theano
import csv
# import ConfigParser
import collections
import time
import csv
import os
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
import numpy
from datetime import datetime
# import path
from os.path import basename
import glob
# import theano.sandbox
# theano.sandbox.cuda.use('gpu0')

from utils import *

""" hyper-parameters """

batch_size = 60
segment_size = 32
lambda1 = 0.00008
lambda2 = 0.00008


""" define model """

print("Create Model")
model = Sequential()
model.add(Dense(512, input_dim=4096, kernel_initializer='glorot_normal',
                kernel_regularizer=l2(0.001), activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(32, kernel_initializer='glorot_normal',
                kernel_regularizer=l2(0.001)))
model.add(Dropout(0.6))
model.add(Dense(1, kernel_initializer='glorot_normal',
                kernel_regularizer=l2(0.001), activation='sigmoid'))


def custom_loss(y_true, y_pred):
    """custom objective function

    Arguments:
        y_true {tensor} -- size: batch_size/2 * 2 * seg_size
        y_pred {tensor} -- size: y_true
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    n_seg = segment_size
    n_vid = batch_size
    n_each = n_vid / 2

    assert y_true.shape == n_seg * \
        n_vid, f"y_true shape: {y_true.shape} != {n_seg * n_vid}"

    assert y_pred.shape == n_seg * \
        n_vid, f"y_true shape: {y_pred.shape} != {n_seg * n_vid}"

    _dtye = y_true.dtype
    Loss1 = K.constant([], dtype=_dtye)
    Loss2 = K.constant([], dtype=_dtye)
    Loss3 = K.constant([], dtype=_dtye)

    for i_b in range(n_vid / 2):
        # the first n_seg segments are for abnormal and next n_seg normal
        # y_true_batch_all = y_true[i_b*2*n_seg: (i_b+1)*2*n_seg]
        y_pred_batch_all = y_pred[i_b*2*n_seg: (i_b+1)*2*n_seg]

        y_batch_abn = y_pred_batch_all[:n_seg] # score for an abnormal video
        y_batch_nor = y_pred_batch_all[n_seg:] # score for a normal video

        loss_1 = K.max([0,1 - K.max(y_batch_abn) + K.max(y_batch_nor)])
        loss_2 = K.sum(K.square(y_batch_abn[:-1]-y_batch_abn[1:]))
        loss_3 = K.sum(y_batch_abn)

        Loss1 = K.concatenate([Loss1, loss_1])
        Loss2 = K.concatenate([Loss2, loss_2])
        Loss3 = K.concatenate([Loss3, loss_3])

    total_loss = K.mean(Loss1) + lambda1 * K.sum(Loss2) + lambda2 * K.sum(Loss3)


""" create network """

adagrad = Adagrad(lr=0.01, epsilon=1e-08) # optimizer

model.compile(loss=custom_loss, optimizer=adagrad)
