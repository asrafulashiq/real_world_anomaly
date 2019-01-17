import cv2
import os
import csv
import pathlib2
from pathlib2 import Path

def get_num_frame(vid_file):
    """get the number of frames in a video

    Arguments:
        vid_file {string} -- video file name
    """
    assert os.path.exists(vid_file), "file (%s) not found".format(vid_file)
    cap = cv2.VideoCapture(vid_file)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length


def get_csv_line_video(vid_file, clip_length=16):
    """create a line for csv file to be fed to ldb

    Arguments:
        vid_file {string} --

    Keyword Arguments:
        clip_length {int} --  (default: {16})
    """
    if type(vid_file) == pathlib2.PosixPath:
        vid_file = str(vid_file)
    assert os.path.exists(vid_file), "file (%s) not found".format(vid_file)
    num_frames = get_num_frame(vid_file)

    strt_frm_list = range(0, num_frames, clip_length)
    strt_frm_list.pop()
    line_l = zip([os.path.abspath(vid_file)]*len(strt_frm_list), strt_frm_list)
    return line_l


def write_csv_for_lmdb(vid_files, csv_name,clip_length=16):
    """create csv file for lmdb
    """
    if type(vid_files) != list:
        vid_files = [vid_files]
    lines = []
    for vid_file in vid_files:
        each_line = get_csv_line_video(vid_file, clip_length=clip_length)
        lines.extend(each_line)
    with open(csv_name, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['org_video', 'label', 'start_frm', 'video_id'])
        for i, line in enumerate(lines):
            writer.writerow([line[0], 0, line[1], i])

if __name__ == "__main__":
    vid_file = '/media/ash/New Volume/dataset/UCF_crime/Anomaly-Videos/Abuse/Abuse001_x264.mp4'
    l = get_num_frame(vid_file)
    print(l)


    vid_files = ['/media/ash/New Volume/dataset/UCF_crime/Anomaly-Videos/Abuse/Abuse001_x264.mp4',
                 '/media/ash/New Volume/dataset/UCF_crime/Anomaly-Videos/Abuse/Abuse002_x264.mp4']
    csv_file = 'tmp.csv'
    write_csv_for_lmdb(vid_files, csv_file)
