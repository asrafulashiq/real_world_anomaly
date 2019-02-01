from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.optimizers import Adagrad


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

    # import pdb; pdb.set_trace()
    _dtye = y_true.dtype
    Loss1 = K.constant([], dtype=_dtye)
    Loss2 = K.constant([], dtype=_dtye)
    Loss3 = K.constant([], dtype=_dtye)

    for i_b in range(n_vid // 2):
        # the first n_seg segments are for abnormal and next n_seg normal
        # y_true_batch_all = y_true[i_b*2*n_seg: (i_b+1)*2*n_seg]
        y_pred_batch_all = y_pred[i_b*2*n_seg: (i_b+1)*2*n_seg]

        y_batch_abn = y_pred_batch_all[:n_seg]  # score for an abnormal video
        y_batch_nor = y_pred_batch_all[n_seg:]  # score for a normal video

        loss_1 = K.max([0, 1 - K.max(y_batch_abn) + K.max(y_batch_nor)])
        loss_2 = K.sum(K.square(y_batch_abn[:-1]-y_batch_abn[1:]))
        loss_3 = K.sum(y_batch_abn)

        Loss1 = K.concatenate([Loss1, [loss_1]])
        Loss2 = K.concatenate([Loss2, [loss_2]])
        Loss3 = K.concatenate([Loss3, [loss_3]])

    total_loss = K.mean(Loss1) + lambda1 * K.sum(Loss2) + \
        lambda2 * K.sum(Loss3)
    return total_loss


def create_model_3d(lamb=0.01):
    """ define model for less input dim """

    model = Sequential()
    model.add(Dense(512, input_dim=512, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(lamb), activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(lamb)))
    model.add(Dropout(0.6))
    model.add(Dense(1, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(lamb), activation='sigmoid'))
    return model


def create_model(lamb=0.01):
    """ define model """

    model = Sequential()
    model.add(Dense(512, input_dim=4096, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(lamb), activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(lamb)))
    model.add(Dropout(0.6))
    model.add(Dense(1, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(lamb), activation='sigmoid'))
    return model


def create_model_with_attention(lamb):

