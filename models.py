from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Input, multiply, Lambda
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adagrad
from keras.layers import Activation, Flatten, Reshape


# default hyper-parameters
batch_size = 60
segment_size = 32
lambda1 = 8e-5
lambda2 = 8e-5

num_iters = 20000
lr = 0.001


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


def custom_loss_attn(y_true, y_pred):
    """custom objective function for attention """
    return K.binary_crossentropy(y_true, y_pred)


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


def create_model(lamb=0.01, feat_size=4096):
    """ define model """

    model = Sequential()
    model.add(Dense(512, input_dim=feat_size,
                    kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(lamb), activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(lamb)))
    model.add(Dropout(0.6))
    model.add(Dense(1, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(lamb), activation='sigmoid',
                    name='score'))
    return model


def create_model_with_attention(segment_size=32, lamb=0.01, feat_size=4096):
    """ model with temporal attention """

    # attention model
    input_shape = (segment_size, feat_size)
    inputs = Input(input_shape, name="input_features")

    x = Dense(512, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(1)(x)
    x = Flatten()(x)
    x_score = Activation('softmax', name='score')(x)
    # x_soft = K.expand_dims(x_score, axis=-1)
    x_soft = Reshape((segment_size, 1))(x_score)
    y = multiply([inputs, x_soft])

    y = Lambda(
        lambda z: K.sum(z, axis=-2)
    )(y)

    # classification model
    y = Dense(512, kernel_initializer='glorot_normal',
              kernel_regularizer=l2(lamb), activation='relu')(y)
    y = Dropout(0.6)(y)
    y = Dense(32, kernel_initializer='glorot_normal',
              kernel_regularizer=l2(lamb))(y)
    y = Dropout(0.6)(y)
    y = Dense(1, kernel_initializer='glorot_normal',
              kernel_regularizer=l2(lamb), activation='sigmoid')(y)

    model = Model(inputs=inputs, outputs=y)

    return model


def get_compiled_model(segment_size=32, feat_size=4096, lamb=0.01,
                       model_type="c3d"):
    adagrad = Adagrad(lr=lr, epsilon=1e-08)  # optimizer
    model = None
    _loss = None

    if model_type == "c3d":  # normal c3d model
        model = create_model(lamb, feat_size)
        _loss = custom_loss
    elif model_type == 'c3d-attn':
        model = create_model_with_attention(
            segment_size, lamb, feat_size
        )
        _loss = custom_loss_attn
    elif model_type == '3d':
        model = create_model_3d(lamb)
        _loss = custom_loss

    model.compile(loss=_loss, optimizer=adagrad)
    return model
