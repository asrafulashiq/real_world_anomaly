import os
import logging
from tqdm import tqdm
import argparse
import datetime
from utils import save_model
from dataloader import load_dataset_batch, load_valid_batch
from dataloader import load_dataset_batch_with_segment, tmp_load
from models import get_compiled_model
import numpy as np
import pickle

from global_var import segment_size, lambda3, batch_size, feat_size

# set logging
logging.basicConfig()
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def train(abnormal_list_path, normal_list_path, output_dir,
          model_path, weight_path, num_iters=20000, model_type="c3d",
          validation=False, pretrain=False):
    """start training"""
    _load = load_dataset_batch
    if model_type == "c3d":
        # feat_size = 4096
        pass
    elif model_type == "tcn":
        # feat_size = 4096
        _load = load_dataset_batch_with_segment
    elif model_type == "c3d-attn":
        # feat_size = 4096
        _load = tmp_load
        cur_loader = _load(abnormal_list_path,
                           normal_list_path,
                           batch_size=batch_size,
                           segment_size=segment_size,
                           feat_size=feat_size)
    # else:
        # feat_size = 512

    if pretrain:
        import glob
        from utils import load_model
        from models import custom_loss_attn, custom_loss_l1
        from keras.optimizers import Adam
        from global_var import lr

        list_weights = glob.glob(output_dir + '/weights*')
        weight_path = max(list_weights, key=os.path.getctime)
        model = load_model(model_path, weight_path=weight_path)
        _loss = [custom_loss_attn, custom_loss_l1]
        _loss_weights = [1.0, 1e-4]
        optim = Adam(lr=lr)
        model.compile(optim, _loss, loss_weights=_loss_weights)

    else:
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
        from utils_attn import validation_result
        valid_list_file = (
            '/media/ash/New Volume/dataset/UCF_crime/'
            'custom_split_C3D/Custom_test_split_mini.txt')
        tmp_ann_file = './Temporal_Anomaly_Annotation.txt'

        import matplotlib.pyplot as plt
        # from utils import get_frames_32_seg, get_num_frame
        from pathlib import Path
        from collections import defaultdict, deque
        all_lines = defaultdict(lambda: deque(maxlen=10))

    for cur_iter in tqdm(range(num_iters)):
        # get one batch
        if model_type == 'tcn' or model_type == 'c3d-attn':
            inputs, labels, paths, enc = next(cur_loader)
        else:
            inputs, labels = _load(abnormal_list_path,
                                   normal_list_path,
                                   feat_size=feat_size)
        # train on a batch
        batch_loss = model.train_on_batch(inputs, labels)
        losses.append(batch_loss)

        if (cur_iter+1) % 10 == 0:  # logging
            log.info(
                f"{cur_iter+1}: loss : {batch_loss}")

        if (cur_iter+1) % 1000 == 0:  # save weight
            weight_path = os.path.join(output_dir, 'weights-' +
                                       str(cur_iter) + '.h5')
            save_model(model, weight_path=weight_path)
            log.debug(f'save model at iteration {cur_iter+1}')

        if (cur_iter+1) % 50 == 0 and validation:  # validation

            _all_pred = model.predict_on_batch(inputs)
            all_pred = _all_pred[1]
            for _i in range(0, batch_size, 2):
                _pred = all_pred[_i]
                abn_score = np.squeeze(_pred)
                # cls_score = enc.inverse_transform(
                    # np.argmax(_pred[:, 1:], axis=-1))
                vid_path = paths[_i]
                all_lines[Path(vid_path).stem].append(abn_score)

        if (cur_iter+1) % 200 == 0 and validation:  # validation
            total_images = len(all_lines)
            im_per_row = 4
            total_row = int(np.ceil(total_images / im_per_row))
            fig, _ = plt.subplots(total_row, im_per_row)
            fig.set_size_inches(10, 20)
            axes = fig.axes

            for cnt, k in enumerate(sorted(list(all_lines.keys()))):
                num_lines = len(all_lines[k])
                alpha_values = np.linspace(0.3, 1, num_lines)
                ax = axes[cnt]
                for line_ctr in range(num_lines):
                    ax.plot(all_lines[k][line_ctr],
                            alpha=alpha_values[line_ctr])
                _title = k.split("_")[0]  #+ '_' + all_lines[k][-1][1][0][:3]  \
                    # + '' + str(all_lines[k][-1][1][1])
                ax.set_title(_title)
                ax.set_ylim(0, 1.1)

            fig.tight_layout()
            fig.savefig('./tmp/attn.pdf')
            # fig.savefig('./tmp/'+str(cur_iter+1)+'.pdf')
        if validation and (cur_iter+1) % 1000 == 0:
            plot_path = './results/plots/C3D_attn/valid_plot_' + \
                        str(cur_iter+1) + '.pdf'
            validation_result(model, valid_list_file, tmp_ann_file,
                              log=log, plot=True, plot_path=plot_path)


def main():

    # parse arguments
    parser = argparse.ArgumentParser(description="Training anomaly detection")
    parser.add_argument('--iter', type=int, default=70000,
                        help='total iteration')
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--model-type", '-m', dest='model_type',
                        type=str, default="c3d-attn",
                        help="Extract C3D features?")
    parser.add_argument("--mini", type=str, default="true",
                        help="Whether to use mini data")
    parser.add_argument("--valid", action='store_false')
    parser.add_argument("--pretrain", action='store_true')
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
          validation=args.valid, pretrain=args.pretrain)


if __name__ == "__main__":
    main()
