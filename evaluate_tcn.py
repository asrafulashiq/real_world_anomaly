""" Evaluate AUC for testing videos """

from pathlib import Path
import pickle
import os
import re
from utils import get_num_frame, get_frames_32_seg
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import datetime
import re
from sklearn.preprocessing import LabelEncoder


np.random.seed(0)

parser = argparse.ArgumentParser(description="Training anomaly detection")
parser.add_argument("--pred", type=str,
                    default="./results/predictions/tcn_mini/",
                    help="path to save predictions")
parser.add_argument("--plot", action='store_false',
                    help='whether to save plots')
parser.add_argument("--plot-path", dest='plot_path', type=str,
                    default="./results/plots/tcn/",
                    help="path to save all plots as pdf file")

args = parser.parse_args()


_HOME = os.environ['HOME']
PRED_PATH = Path(args.pred)
DATA_HOME = Path(_HOME + '/dataset/UCF_crime/')
ANOM_DIR = DATA_HOME / "Anomaly-Videos"
TEST_NORM_DIR = DATA_HOME / "Testing_Normal_Videos_Anomaly"
SEG = 32
TEMP_ANN_FILE = './Temporal_Anomaly_Annotation.txt'
df_temp_ann = pd.read_csv(
    TEMP_ANN_FILE,
    delimiter=" ",
    header=None,
    skipinitialspace=True
)

#
encoder = LabelEncoder()
_categories = np.array(df_temp_ann.iloc[:, 1])
encoder.fit(_categories)


# prog = re.compile('[^a-zA-Z]')

pred_path_list = sorted(list(PRED_PATH.iterdir()))
print('total test files: ', len(pred_path_list))


if args.plot:
    from matplotlib.backends.backend_pdf import PdfPages
    print('setting up plotting')

    PLOT_PATH = Path(args.plot_path)
    PLOT_PATH.mkdir(exist_ok=True, parents=True)
    pdf_path = PLOT_PATH / ('plots_' + PRED_PATH.parts[-1] + '_'
                            + str(datetime.datetime.now()) + '.pdf')
    pdf_pages = PdfPages(pdf_path)
    nb_plot_per_page = 10
    total_pages = int(np.ceil(len(pred_path_list)/nb_plot_per_page))
    grid_size = (nb_plot_per_page//2, 2)
    # fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
    all_axes = []
    all_figs = []
    for _i in range(total_pages):
        fig, _ = plt.subplots(*grid_size)
        fig.set_size_inches(8.27, 11.69)
        all_figs.append(fig)
        all_axes += fig.axes
    assert len(all_axes) == total_pages * \
        nb_plot_per_page, f'{len(all_axes)} != {total_pages * nb_plot_per_page}'


def get_ind_from_pd(df):
    # print(df.head())
    indices = list(df.iloc[0, 2:])
    ret_ind = []
    for k in range(0, len(indices)-1, 2):
        if indices[k] == -1:
            continue
        ret_ind.append((indices[k], indices[k+1]))
    return ret_ind


all_score_pred = np.array([])
all_score_gt = np.array([])

norm_score_pred = np.array([])
norm_score_gt = np.array([])
abn_score_pred = np.array([])
abn_score_gt = np.array([])

full_video_gt = []
full_video_pred = []

cls_video_gt = []
cls_video_full = []

for i, pred_file in tqdm(enumerate(pred_path_list)):
    if pred_file.suffix != '.pkl':
        continue
    with pred_file.open('rb') as fp:
        _pred_all = pickle.load(fp)

    _pred = _pred_all[:, 0]
    _pred_cls = _pred_all[:, 1:]
    _pred_cls = np.argmax(_pred_cls, axis=-1)

    full_video_pred.append(np.max(_pred))

    # search this pred_file video
    vid_name = pred_file.stem  # remove suffix from file name

    if str(vid_name).startswith('Normal'):  # normal video
        vid_path = TEST_NORM_DIR / vid_name
        assert vid_path.exists()
        cls_video_gt.append('Normal')
    else:  # anomaly video
        anom_type = re.split('[^a-zA-Z]', vid_name)[0]
        vid_path = ANOM_DIR / anom_type / vid_name
        assert vid_path.exists()
        cls_video_gt.append(anom_type)

    # get frame number from video
    num_frames = get_num_frame(vid_path)
    indices = get_frames_32_seg(num_frames, SEG)

    # import pdb
    # pdb.set_trace()

    # get prediction for each frame
    score_pred = np.zeros(num_frames)
    for counter, ind in enumerate(indices):
        start_ind = ind[0]
        end_ind = ind[1]
        score_pred[start_ind:end_ind+1] = _pred[counter]

    all_score_pred = np.concatenate(
        (all_score_pred, score_pred)
    )

    # import pdb; pdb.set_trace()

    # print(pred_file)
    # get gt score for each frame
    score_gt = np.zeros(num_frames)
    x_row = df_temp_ann.loc[df_temp_ann[0] == vid_name]
    gt_ind = get_ind_from_pd(x_row)

    for counter, ind in enumerate(gt_ind):
        start_ind = ind[0]
        end_ind = ind[1]
        score_gt[start_ind:end_ind+1] = 1
    all_score_gt = np.concatenate(
        (all_score_gt, score_gt)
    )

    if args.plot:
        ax = all_axes[i]
        ax.set_ylim(0, 1.2)
        ax.plot(score_pred, color='g', linewidth=2)
        ax.plot(score_gt, color='r', linestyle='dashed')
        ax.set_title(vid_name)

    if len(gt_ind) != 0:  # abnormal video
        # _auc = roc_auc_score(score_gt, score_pred)
        # print(_auc)
        # print()
        abn_score_gt = np.concatenate((abn_score_gt, score_gt))
        abn_score_pred = np.concatenate((abn_score_pred, score_pred))

        full_video_gt.append(1)
    else:
        # norm_score_gt = np.concatenate((norm_score_gt, score_gt))
        norm_score_pred = np.concatenate((norm_score_pred, score_pred))
        full_video_gt.append(0)


fpr, tpr, thresholds = roc_curve(all_score_gt, all_score_pred)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc}")
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange',
#          label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

# precision recall
# pr, re, _ = precision_recall_curve(all_score_gt, all_score_pred)
# average_precision = average_precision_score(all_score_gt, all_score_pred)

# _fig, _ax = plt.subplots()
# _ax.step(re, pr, color='b', alpha=0.2, where='post')
# _ax.set_xlabel('Recall')
# _ax.set_ylabel('Precision')
# _ax.set_ylim([0.0, 1.05])
# _ax.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
#           average_precision))
# _fig.show()
# print(f' AP: {average_precision}')


# optimal_idx = np.argmax(tpr - fpr)
# optimal_threshold = thresholds[optimal_idx]
optimal_threshold = 0.5
print("Threshold :", optimal_threshold)


# get false alarm for normal
false_alarm = sum(norm_score_pred > optimal_threshold) / len(norm_score_pred)
print("Normal video:")
print(f" False alarm for +ve: {false_alarm*100}")

# get TPR for abnormal
p_ind = (abn_score_gt == 1.)
recall = sum(abn_score_pred[p_ind] > optimal_threshold) / sum(p_ind)
prec = sum(abn_score_pred[p_ind] > optimal_threshold) / \
    sum(abn_score_pred > optimal_threshold)
print("Abnormal video:")
print(f" precision: {prec*100}\n recall: {recall*100}")

if args.plot:
    for fig in all_figs:
        fig.tight_layout()
        pdf_pages.savefig(fig)
    pdf_pages.close()
    print(f'{pdf_path} saved')


# ## full video classification
full_video_pred = [int(i > 0.5) for i in full_video_pred]
accuracy_full = accuracy_score(
    full_video_gt, full_video_pred)
print(f" full video accuracy : {accuracy_full}")