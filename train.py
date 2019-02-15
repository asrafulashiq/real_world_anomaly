import os
import logging
from tqdm import tqdm
import argparse
import datetime
from utils.utils import save_model
from IO.dataloader import load_dataset_batch, load_valid_batch
from IO.dataloader import load_dataset_batch_with_segment
from models import get_compiled_model
import numpy as np
import pickle


lambda3 = 0.01
batch_size = 60
segment_size = 32

# set logging
logging.basicConfig()
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def train(abnormal_list_path, normal_list_path, output_dir,
          model_path, weight_path, num_iters=20000, model_type="c3d",
          validation=False):
    """start training"""
    _load = load_dataset_batch
    if model_type == "c3d":
        feat_size = 4096
    elif model_type == "tcn":
        feat_size = 4096
        _load = load_dataset_batch_with_segment
    elif model_type == "c3d-attn":
        feat_size = 4096
        _load = load_dataset_batch_with_segment
    else:
        feat_size = 512

    model = get_compiled_model(
        segment_size, feat_size, lambda3, model_type
    )

    # import pdb
    # pdb.set_trace()

    log.info(model.summary())

    log.info(f"saving model in {model_path}")
    save_model(model, model_path)

    log.info("Iteration started")
    losses = []

    if validation:
        from keras import Model
        from utils import valid_res
        valid_list_file = (
            '/media/ash/New Volume/dataset/UCF_crime/'
            'custom_split/Custom_test_split_mini.txt')
        tmp_ann_file = './Temporal_Anomaly_Annotation.txt'
        valid_gen = load_valid_batch(
            valid_list_file,
            tmp_ann_file
        )
        model_valid = Model(
            inputs=model.input,
            outputs=[
                model.output,
                model.get_layer(name='score').output
            ]
        )
        with open("./frame_num.pkl", "rb") as fp:
            dict_frame = pickle.load(fp)

    for cur_iter in tqdm(range(num_iters)):
        # get one batch
        inputs, labels = _load(abnormal_list_path,
                               normal_list_path,
                               feat_size=feat_size)

        # train on a batch
        batch_loss = model.train_on_batch(inputs, labels)
        losses.append(batch_loss)

        if (cur_iter+1) % 20 == 0:  # logging
            log.info(
                f"{cur_iter+1}: loss : {batch_loss}")

        if (cur_iter+1) % 1000 == 0:  # save weight
            weight_path = os.path.join(output_dir, 'weights-' +
                                       str(cur_iter) + '.h5')
            save_model(model, weight_path=weight_path)
            log.debug(f'save model at iteration {cur_iter+1}')

        if (cur_iter+1) % 1000 == 0 and validation:  # validation
            valid_batch = 50

            list_gt_ind = []
            list_gt_pred = []
            list_vid_path = []
            for i_valid, (vid_file, gt_ind, val_x) in enumerate(valid_gen):
                val_x = np.expand_dims(val_x, axis=0)
                pred_all, pred_x = model_valid.predict_on_batch(val_x)
                pred_x = pred_x.squeeze()

                list_gt_ind.append(gt_ind)
                list_gt_pred.append(pred_x)
                list_vid_path.append(vid_file)
                if i_valid == valid_batch - 1:
                    break
            auc_score = valid_res(
                list_gt_ind,
                list_gt_pred,
                list_vid_path,
                dict_frame,
                seg=32
            )
            log.info("\nValidation result")
            log.info("  AUC: {}\n".format(auc_score))


def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="Training anomaly detection")
    parser.add_argument('--iter', type=int, default=20000,
                        help='total iteration')
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--model-type", '-m', dest='model_type',
                        type=str, default="c3d-attn",
                        help="Extract C3D features?")
    parser.add_argument("--mini", type=str, default="true",
                        help="Whether to use mini data")
    parser.add_argument("--valid", action='store_true')
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
        flag_split = "_C3D"
    elif args.model_type == '3d':
        output_dir = 'model/trained_model'+flag_mini+'/3D'
        flag_split = "_3d"
    elif args.model_type == 'c3d-attn':
        output_dir = 'model/trained_model'+flag_mini+'/C3D_attn'
        flag_split = "_C3D"
    elif args.model_type == 'tcn':
        output_dir = 'model/trained_model'+flag_mini+'/tcn'
        flag_split = '_C3D'

    # create file handler which logs even debug messages
    now = datetime.datetime.now()
    log_file = now.strftime("%m_%d_%H")
    log.info(f"log time: {now}")
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

    log.info(f"model output path {output_dir}")

    # import pdb; pdb.set_trace()

    # start training
    train(abnormal_list_path, normal_list_path, output_dir,
          model_path, weight_path, num_iters, args.model_type,
          validation=args.valid)


if __name__ == "__main__":
    main()
