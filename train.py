from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.optimizers import Adagrad
import os
import logging
from tqdm import tqdm
import argparse
import datetime
from utils import load_dataset_batch, save_model


# set logging
logging.basicConfig()
log = logging.getLogger("train")
log.setLevel(logging.INFO)

# create file handler which logs even debug messages
now = datetime.datetime.now()
log_file = now.strftime("%m_%d")
fh = logging.FileHandler('logs/logging_'+log_file+'.log')
fh.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
fh.setFormatter(formatter)
log.addHandler(fh)


""" default hyper-parameters """

batch_size = 60
segment_size = 32
lambda1 = 0.00008
lambda2 = 0.00008
num_iters = 20000


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


def create_model():
    """ define model """

    log.debug("Create Model")
    model = Sequential()
    model.add(Dense(512, input_dim=4096, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.001), activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(32, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1, kernel_initializer='glorot_normal',
                    kernel_regularizer=l2(0.001), activation='sigmoid'))
    log.info("model created")
    return model


def train(abnormal_list_path, normal_list_path, output_dir,
          model_path, weight_path, num_iters=20000):
    """start training"""
    model = create_model()
    adagrad = Adagrad(lr=0.01, epsilon=1e-08)  # optimizer
    model.compile(loss=custom_loss, optimizer=adagrad)
    save_model(model, model_path)

    log.info("Iteration started")
    losses = []

    for cur_iter in tqdm(range(num_iters)):
        # get one batch
        inputs, labels = load_dataset_batch(abnormal_list_path,
                                            normal_list_path)

        # train on a batch
        batch_loss = model.train_on_batch(inputs, labels)
        losses.append(batch_loss)

        if cur_iter % 20 == 0:  # logging
            log.info(
                f"{cur_iter}: loss : {batch_loss}")

        if cur_iter % 1000 == 0:  # save weight
            weight_path = os.path.join(output_dir, 'weights-' + str(cur_iter) + '.h5')
            save_model(model, weight_path=weight_path)
            log.debug(f'save model for iter {iter}')


def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="Training anomaly detection")
    parser.add_argument('--iter', type=int, default=20000,
                        help='total iteration')
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma separated list of GPU devices to use")
    args = parser.parse_args()
    log.info(args)

    num_iters = args.iter
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # define all path
    _HOME = os.environ['HOME']
    abnormal_list_path = _HOME + \
        '/dataset/UCF_crime/custom_split/Custom_train_split_mini_abnormal.txt'
    normal_list_path = _HOME + \
        '/dataset/UCF_crime/custom_split/Custom_train_split_mini_normal.txt'

    output_dir = 'model/trained_model'
    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(abnormal_list_path), \
        f'{abnormal_list_path} does not exist'
    assert os.path.exists(normal_list_path), \
        f'{normal_list_path} does not exist'

    model_path = os.path.join(output_dir, 'model.json')
    weight_path = os.path.join(output_dir, 'weights.h5')

    # start training
    train(abnormal_list_path, normal_list_path, output_dir,
          model_path, weight_path, num_iters)


if __name__ == "__main__":
    main()