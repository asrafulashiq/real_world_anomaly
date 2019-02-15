from utils import get_frames_seg
from pathlib import Path
import pickle
from keras import Model, Input
import numpy as np
from global_var import segment_size, feat_size
from sklearn.metrics import roc_curve, auc
from dataloader import load_valid_batch


SEG = segment_size


def validation_result(model, valid_list_file, tmp_ann_file,
                      log=None, plot=True,
                      plot_path='tmp/val.pdf'):

    all_score_pred = []
    all_score_gt = []

    valid_gen = load_valid_batch(
        valid_list_file,
        tmp_ann_file
    )

    norm_score_pred = np.array([])
    norm_score_gt = np.array([])
    abn_score_pred = np.array([])
    abn_score_gt = np.array([])

    vid_level_gt = []
    vid_level_pred = []

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        print('setting up plotting')

        pdf_path = Path(plot_path)
        pdf_path.parent.mkdir(exist_ok=True, parents=True)
        pdf_pages = PdfPages(pdf_path)
        nb_plot_per_page = 10
        total_pages = int(np.ceil(40./nb_plot_per_page))
        grid_size = (nb_plot_per_page//2, 2)
        # fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        all_axes = []
        all_figs = []
        for _i in range(total_pages):
            fig, _ = plt.subplots(*grid_size)
            fig.set_size_inches(8.27, 11.69)
            all_figs.append(fig)
            all_axes += fig.axes

    # create abn model
    abn_layer = model.get_layer(name='abnormality')
    test_input = Input(batch_shape=(segment_size, feat_size))
    abn_out = abn_layer(test_input)
    abn_model = Model(test_input, abn_out)

    dict_frame_num = pickle.load(open("./frame_num.pkl", "rb"))

    # create attention layer
    attn_model = Model(model.input, model.get_layer('attention').output)

    i = 0
    for vid_name, gt_ind, _input in valid_gen:
        inp = np.expand_dims(_input, axis=0)

        out_atttn = attn_model.predict_on_batch(inp).squeeze()
        out_abn = abn_model.predict_on_batch(_input).squeeze()
        out_all = model.predict_on_batch(inp)[0].squeeze()

        _pred = out_atttn * out_abn

        num_frames = dict_frame_num[vid_name]
        indices = get_frames_seg(num_frames, SEG)

        # get prediction for each frame
        score_pred = np.zeros(num_frames)
        for counter, ind in enumerate(indices):
            start_ind = ind[0]
            end_ind = ind[1]
            score_pred[start_ind:end_ind+1] = _pred[counter]

        all_score_pred.extend(list(score_pred))

        score_gt = np.zeros(num_frames)
        for counter, ind in enumerate(gt_ind):
            start_ind = ind[0]
            end_ind = ind[1]
            score_gt[start_ind:end_ind+1] = 1
        all_score_gt.extend(list(score_gt))

        if len(gt_ind) != 0:  # abnormal video
            abn_score_gt = np.concatenate((abn_score_gt, score_gt))
            abn_score_pred = np.concatenate((abn_score_pred, score_pred))
            vid_level_gt.append(1)
        else:
            # norm_score_gt = np.concatenate((norm_score_gt, score_gt))
            norm_score_pred = np.concatenate((norm_score_pred, score_pred))
            vid_level_gt.append(0)
        vid_level_pred.append(out_all)

        if plot:
            ax = all_axes[i]
            ax.set_ylim(0, 1.2)
            ax.plot(score_pred, color='g', linewidth=2)
            ax.plot(score_gt, color='r', linestyle='dashed')
            ax.set_title(vid_name)
        i += 1

    all_score_gt = np.array(all_score_gt)
    all_score_gt = np.array(all_score_gt)
    fpr, tpr, thresholds = roc_curve(all_score_gt, all_score_pred)
    roc_auc = auc(fpr, tpr)

    if plot:
        for fig in all_figs:
            fig.tight_layout()
            pdf_pages.savefig(fig)
        pdf_pages.close()
        print(f'{pdf_path} saved')

    if log:
        log.info("")
        log.info("")
        log.info("VALIDATION RESULT")
        log.info(f"AUC : {roc_auc}")
