import os
import numpy as np
import cv2
from multiprocessing import Pool
import argparse
# import skvideo.io
# import scipy.misc
from pathlib import Path
# import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm


def ToImg(flow, bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow += bound
    flow *= (255/float(2*bound))

    # return bgr
    return flow.astype(np.uint8)


def save_flows(flows, image, save_dir, num, bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param image: raw image
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id, which belongs one of the extracted frames
    :param bound: set the bi-bound to flow images
    :return: return 0
    '''
    # rescale to 0~255 with the bound setting
    flow_x = ToImg(flows[..., 0], bound)
    flow_y = ToImg(flows[..., 1], bound)
    save_dir.mkdir(parents=True, exist_ok=True)

    # save the flows
    save_x = str(save_dir / 'flow_x_{:05d}.png'.format(num))
    save_y = str(save_dir / 'flow_y_{:05d}.png'.format(num))
    # cv2.imwrite(save_x, flow_x)
    # cv2.imwrite(save_y, flow_y)
    imageio.imwrite(save_x, flow_x)
    imageio.imwrite(save_y, flow_y)
    # return 0
    # plt.imshow(flow_x, 'gray')
    # plt.show()


def dense_flow(augs):
    '''
    To extract dense_flow images
    :param augs:the detailed augments:
        video_name: the video name which is like: 'v_xxxxxxx',if different ,please have a modify.
        save_dir: the destination path's final direction name.
        step: num of frames between each two extracted frames
        bound: bi-bound parameter
    :return: no returns
    '''
    video_name, save_dir, step, bound = augs
    video_path = str(video_name)

    try:
        videocapture = cv2.VideoCapture(video_path)
    except Exception:
        print('{} read error! '.format(video_name))
        return 0
    print(video_name)

    frame_num = 1
    _, prev_image = videocapture.read()
    prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)

    dtvl1 = cv2.createOptFlow_DualTVL1()
    while videocapture.isOpened():

        ret, image = videocapture.read()  # videocapture[num0]

        if not ret:
            break

        next_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # print(frame_num)

        if frame_num % 50 == 0:
            print(f"frame {frame_num}")

        if not dtvl1.getUseInitialFlow():
            flowDTVL1 = dtvl1.calc(prev_image, next_frame, None)
            dtvl1.setUseInitialFlow(True)
        else:
            flowDTVL1 = dtvl1.calc(prev_image, next_frame, flowDTVL1)

        # # this is to save flows and img.
        save_flows(flowDTVL1.copy(), image, save_dir, frame_num, bound)

        prev_image = next_frame
        frame_num += 1

    videocapture.release()
    cv2.destroyAllWindows()
    print(f'{video_path} captured')


def parse_args():
    parser = argparse.ArgumentParser(
        description="densely extract the video frames and optical flows")
    parser.add_argument('--dataset', default='ucf-crime', type=str,
                        help='set the dataset name, to find the data path')
    parser.add_argument(
        '--data_root', default='/home/ash/dataset/UCF_crime', type=str)
    # parser.add_argument('--new_dir', default='flows', type=str)
    parser.add_argument('--num_workers', default=4, type=int,
                        help='num of workers to act multi-process')
    parser.add_argument('--step', default=1, type=int, help='gap frames')
    parser.add_argument('--bound', default=15, type=int,
                        help='set the maximum of optical flow')
    # parser.add_argument('--s_', default=0, type=int, help='start id')
    # parser.add_argument('--e_', default=13320, type=int, help='end id')
    parser.add_argument('--mode', default='run', type=str,
                        help='set \'run\' if debug done, otherwise, set debug')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    data_root = Path(args.data_root)
    videos_root = os.path.join(data_root, 'videos')

    # specify the augments
    num_workers = args.num_workers
    mode = args.mode
    step = args.step
    bound = args.bound

    do_overwrite = True
    _HOME = os.environ['HOME']
    PARENT_FOLDER = Path(_HOME) / 'dataset' / 'UCF_crime'
    ANOM_FOLDER = PARENT_FOLDER / 'Anomaly-Videos'
    assert PARENT_FOLDER.exists()
    # extract anomaly folder
    FEAT_PARENT_FOLDER = PARENT_FOLDER / 'flow'
    FEAT_ANOM_FOLDER = FEAT_PARENT_FOLDER / 'Anomaly-Videos'
    FEAT_ANOM_FOLDER.mkdir(parents=True, exist_ok=True)

    video_list = []
    save_dir = []

    for anom in ANOM_FOLDER.iterdir():
        anom_type = anom.name
        anom_type_folder = FEAT_ANOM_FOLDER

        # create feature folder for this type
        for vid_file in sorted(anom.iterdir()):
            # vid_file_name = vid_file.name
            feat_path = anom_type_folder / vid_file.stem

            if not do_overwrite and feat_path.exists():
                print("{} exists".format(feat_path))
                continue
            video_list.append(vid_file)
            save_dir.append(feat_path)

    # extract normal folder
    normal_test_train = ['Training-Normal-Videos',
                         'Testing_Normal_Videos_Anomaly']

    for normal_folder in normal_test_train:
        feat_normal_fldr = FEAT_PARENT_FOLDER / normal_folder
        feat_normal_fldr.mkdir(parents=True, exist_ok=True)

        normal = PARENT_FOLDER / normal_folder

        for vid_file in sorted(normal.iterdir()):
            # vid_file_name = vid_file.name

            feat_path = feat_normal_fldr / vid_file.stem
            if not do_overwrite and feat_path.exists():
                print("{} exists".format(feat_path))
                continue
            video_list.append(vid_file)
            save_dir.append(feat_path)

    len_videos = len(video_list)
    # pool = Pool(num_workers)
    if mode == 'run':
        # pool.map(dense_flow, list(zip(video_list, save_dir, [
        #             step]*len(video_list), [bound]*len(video_list))))
        for aug in tqdm(list(zip(video_list, save_dir,
                                [step]*len(video_list), [bound]*len(video_list)))):
            dense_flow(aug)

    else:  # mode=='debug
        dense_flow((video_list[0], save_dir[0], step, bound))
