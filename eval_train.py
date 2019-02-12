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
from matplotlib.backends.backend_pdf import PdfPages


np.random.seed(0)
SEG = 32

_HOME = os.environ['HOME']
pred_path = Path('/media/ash/New Volume/dataset/UCF_crime/custom_split_C3D/Custom_train_split_mini_abnormal.txt')

with pred_path.open("r") as fp:
    pred_path_list = sorted([line.rstrip() for line in fp])

print('setting up plotting')

PLOT_PATH = Path('./results/plots_train/')
PLOT_PATH.mkdir(exist_ok=True, parents=True)
pdf_path = PLOT_PATH / ('plots_' + str(datetime.datetime.now()) + '.pdf')
pdf_pages = PdfPages(pdf_path)
nb_plot_per_page = 18
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


full_video_gt = []
full_video_pred = []

for i, pred_file in tqdm(enumerate(pred_path_list)):
    pred_file = Path(pred_file)
    if pred_file.suffix != '.pkl':
        continue
    with pred_file.open('rb') as fp:
        _pred_all = pickle.load(fp)

    if len(_pred_all.shape) > 1:
        _pred = _pred_all[:, 0]
        _pred_cls = _pred_all[:, 1:]
        _pred_cls = np.argmax(_pred_cls, axis=-1)

    full_video_pred.append(np.max(_pred))

    # search this pred_file video
    vid_name = pred_file.stem  # remove suffix from file name

    if str(vid_name).startswith('Normal'):  # normal video
        full_video_gt.append(0)
    else:  # anomaly video
        full_video_gt.append(1)

    # get frame number from video
    num_frames = get_num_frame(pred_file)
    indices = get_frames_32_seg(num_frames, SEG)

    # import pdb
    # pdb.set_trace()

    # get prediction for each frame
    score_pred = np.zeros(num_frames)
    for counter, ind in enumerate(indices):
        start_ind = ind[0]
        end_ind = ind[1]
        score_pred[start_ind:end_ind+1] = _pred[counter]


    ax = all_axes[i]
    ax.set_ylim(0, 1.2)
    ax.plot(score_pred, color='g', linewidth=2)
    ax.set_title(vid_name)


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
