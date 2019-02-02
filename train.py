import os
import logging
from tqdm import tqdm
import argparse
import datetime
from utils import save_model
from dataloader import load_dataset_batch
from dataloader import load_dataset_batch_with_segment
from models import get_compiled_model


lambda3 = 0.01
batch_size = 60
segment_size = 32

# set logging
logging.basicConfig()
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def train(abnormal_list_path, normal_list_path, output_dir,
          model_path, weight_path, num_iters=20000, model_type="c3d"):
    """start training"""
    _load = load_dataset_batch
    if model_type == "c3d":
        feat_size = 4096
    elif model_type == "c3d-attn":
        feat_size = 4096
        _load = load_dataset_batch_with_segment
    else:
        feat_size = 512

    model = get_compiled_model(
        segment_size, feat_size, lambda3, model_type
    )

    log.info(model.summary())

    log.info(f"saving model in {model_path}")
    save_model(model, model_path)

    log.info("Iteration started")
    losses = []

    for cur_iter in tqdm(range(num_iters)):
        # get one batch
        inputs, labels = _load(abnormal_list_path,
                               normal_list_path,
                               feat_size=feat_size)

        # train on a batch
        batch_loss = model.train_on_batch(inputs, labels)
        losses.append(batch_loss)

        if cur_iter % 20 == 0:  # logging
            log.info(
                f"{cur_iter}: loss : {batch_loss}")

        if cur_iter % 1000 == 0:  # save weight
            weight_path = os.path.join(output_dir, 'weights-' +
                                       str(cur_iter) + '.h5')
            save_model(model, weight_path=weight_path)
            log.debug(f'save model for iter {iter}')


def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="Training anomaly detection")
    parser.add_argument('--iter', type=int, default=20000,
                        help='total iteration')
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--model-type", '-m', dest='model_type',
                        type=str, default="c3d",
                        help="Extract C3D features?")
    parser.add_argument("--mini", type=str, default="false",
                        help="Whether to use mini data")
    args = parser.parse_args()
    log.info(args)

    num_iters = args.iter
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    # define all path
    _HOME = os.environ['HOME']
    if args.mini == "true":
        flag_mini = "_mini"
    else:
        flag_mini = ""

    if args.model_type == 'c3d':
        output_dir = 'model/trained_model'+flag_mini+'/C3D'
        flag_split = ""
    elif args.model_type == '3d':
        output_dir = 'model/trained_model'+flag_mini+'/3D'
        flag_split = "_3d"
    elif args.model_type == 'c3d-attn':
        output_dir = 'model/trained_model'+flag_mini+'/C3D_attn'
        flag_split = ""

    # create file handler which logs even debug messages
    now = datetime.datetime.now()
    log_file = now.strftime("%m_%d")
    fh = logging.FileHandler(
        'logs/logging_' + log_file +
        '_' + args.model_type + flag_mini + '.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)

    abnormal_list_path = _HOME + \
        '/dataset/UCF_crime/custom_split' + flag_split +\
        '/Custom_train_split' + flag_mini + '_abnormal.txt'
    normal_list_path = _HOME + \
        '/dataset/UCF_crime/custom_split' + flag_split +\
        '/Custom_train_split' + flag_mini + '_normal.txt'

    os.makedirs(output_dir, exist_ok=True)

    assert os.path.exists(abnormal_list_path), \
        f'{abnormal_list_path} does not exist'
    assert os.path.exists(normal_list_path), \
        f'{normal_list_path} does not exist'

    model_path = os.path.join(output_dir, 'model.json')
    weight_path = os.path.join(output_dir, 'weights.h5')

    # start training
    train(abnormal_list_path, normal_list_path, output_dir,
          model_path, weight_path, num_iters, args.model_type)


if __name__ == "__main__":
    main()
