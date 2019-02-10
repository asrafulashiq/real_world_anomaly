from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout, Input, multiply, Lambda
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adagrad
from keras.layers import Activation, Flatten, Reshape
from tcn import TCN
from keras import activations
from keras.layers import Concatenate
from keras.losses import binary_crossentropy
import numpy as np


# default hyper-parameters
batch_size = 60
segment_size = 32
lambda1 = 8e-5
lambda2 = 8e-5

num_iters = 20000
lr = 0.001


def custom_multi_class_loss(y_true, y_pred):
    """ multi class custom loss
    Arguments:
        y_true -- size: [[None, 32, 6]]
    """
    # gt_score = y_true[:, :, 0]
    pred_score = y_pred[:, :, 0]

    # class_agnostic_score = K.max(pred_score, axis=1)

    class_agnostic_score = K.constant([])
    for i in range(batch_size):
        a = K.max(pred_score[i])
        class_agnostic_score = K.concatenate(
            [class_agnostic_score, [a]]
        )

    # class_agnostic_gt = K.max(gt_score, axis=1)
    class_agnostic_gt = K.constant(np.tile(
        np.array([1, 0], dtype=np.float),
        batch_size // 2
    ))

    class_gt_score = y_true[:, :, 1:]
    class_pred_score = y_pred[:, :, 1:]

    loss1 = binary_crossentropy(
        class_agnostic_gt, class_agnostic_score
    )

    loss_cat = K.constant([])

    # work on the segments where abnormality greater than 0.5
    for i in range(0, batch_size, 2):
        batch_pred = class_pred_score[i]  # shape : (32, 5)
        wh = K.tf.greater(K.flatten(pred_score[i]), 0.5)  # shape : (32,)
        abn_batch_pred = K.tf.boolean_mask(batch_pred, wh)  # shape : (?, 5)
        batch_gt = class_gt_score[i]
        abn_batch_gt = K.tf.boolean_mask(batch_gt, wh)

        _cat_loss = K.categorical_crossentropy(
            abn_batch_gt, abn_batch_pred
        )
        loss_cat = K.concatenate([loss_cat, _cat_loss])

    total_loss = loss1 + K.mean(loss_cat)

    return total_loss


def tmp_custom_multi_class_loss(y_true, y_pred):
    """ multi class custom loss
    Arguments:
        y_true -- size: [[None, 32, 6]]
    """
    # gt_score = y_true
    pred_score = y_pred

    # class_agnostic_score = K.max(pred_score, axis=-2)
    #class_agnostic_gt = K.max(gt_score, axis=1)


    tmp = K.constant([])
    for i in range(batch_size):
        a = K.max(K.flatten(pred_score[i]))
        tmp = K.concatenate([tmp, [a]])

    class_agnostic_gt = K.constant(np.tile(
        np.array([1, 0], dtype=np.float),
        batch_size // 2
    ))

    # loss1 = binary_crossentropy(
    #     class_agnostic_gt, K.flatten(class_agnostic_score)
    # )

    loss1 = binary_crossentropy(class_agnostic_gt, tmp)

    total_loss = loss1  # + K.mean(loss_cat)

    return total_loss


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

        # tmp_abn = K.max(y_batch_abn)
        # tmp_n = K.max(y_batch_nor)
        # tmp = K.constant([])
        # tmp = K.concatenate([tmp, [tmp_abn]])
        # tmp = K.concatenate([tmp, [tmp_n]])
        # loss_1 = binary_crossentropy(K.constant([1,0]), tmp)

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


def create_model_tcn(segment_size=32, feat_size=4096):
    """model with temporal convolutional layer"""
    input_shape = (segment_size, feat_size)
    inputs = Input(input_shape, name='input')

    t1 = TCN(nb_filters=128, kernel_size=3, dropout_rate=0.5,
             dilations=[1, 2], name='t1')(inputs)
    # t2 = TCN(nb_filters=128, kernel_size=3, dropout_rate=0.3,
    #          dilations=[1, 2, 4, 8], name='t2')(t1)
    # t3 = TCN(nb_filters=128, kernel_size=3, dropout_rate=0.3,
    #          dilations=[1, 2, 4, 8], name='t3')(t2)

    t4 = TCN(nb_filters=1, kernel_size=2, dilations=[1, 2],
             dropout_rate=0.5, name='t4', use_skip_connections=False)(t1)

    # out = Activation('sigmoid', name='score')(t1)
    model = Model(inputs=inputs, outputs=t4)
    model.layers[-1].activation = activations.sigmoid

    return model


def create_multi_class_model(segment_size=32, feat_size=4096, num_classes=5):
    """ multi class network"""
    input_shape = (segment_size, feat_size)
    inputs = Input(input_shape, name='input')

    # normal / abnormal class model
    t1 = TCN(nb_filters=128, kernel_size=3, dropout_rate=0.5,
             dilations=[1, 2], name='t1')(inputs)
    t2 = TCN(nb_filters=1, kernel_size=2, dilations=[1, 2], dropout_rate=0.5,
             name='abnormality', use_skip_connections=False)(t1)

    model_abnormality = Model(inputs=inputs, outputs=t2)
    model_abnormality.layers[-1].activation = activations.sigmoid

    # class specific model
    tc2 = TCN(nb_filters=128, kernel_size=2, dilations=[1, 2],
              dropout_rate=0.5, name='tc2')(inputs)
    tc3 = TCN(nb_filters=num_classes, kernel_size=2, dilations=[1],
              use_skip_connections=False, name='cls_out')(tc2)
    model_class = Model(inputs, tc3)
    model_class.layers[-1].activation = activations.softmax

    out = Concatenate(name='output')([
        model_abnormality.output, model_class.output
    ])

    model = Model(inputs=inputs, outputs=out)

    return model


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
    elif model_type == 'tcn':
        # model = create_model_tcn(segment_size=segment_size,
        #                          feat_size=feat_size)
        # _loss = tmp_custom_multi_class_loss #custom_loss
        model = create_multi_class_model(num_classes=5)
        _loss = custom_multi_class_loss

    model.compile(loss=_loss, optimizer=adagrad)
    return model
