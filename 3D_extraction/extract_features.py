import os
from helper import *
from pathlib2 import Path
import platform
import subprocess

if platform.node() == 'ash-ubuntu':
	PARENT_FOLDER = Path('/media/ash/New Volume/dataset/UCF_crime/')
else:
	PARENT_FOLDER = Path('~/dataset/UCF_crime')
	import sys
	sys.path.append("/home/islama6a/local/pytorch/build")

ANOM_FOLDER = PARENT_FOLDER / 'Anomaly-Videos'

assert PARENT_FOLDER.exists()

CMD_1 = '''
python data/create_video_db.py \
--list_file=tmp.csv \
--output_file=tmp_lmdb_data \
--use_list=1 --use_video_id=1 --use_start_frame=1
'''

CMD_2_tmp = '''
python  tools/extract_features.py \
--test_data=tmp_lmdb_data \
--model_name=r2plus1d --model_depth=18 --clip_length_rgb=16 \
--gpus=2 \
--batch_size=4 \
--load_model_path=./model/r2.5d_d18_l16.pkl \
--output_path=%s \
--features=softmax,final_avg,video_id \
--sanity_check=0 --get_video_id=1 --use_local_file=1 --num_labels=1
'''

# import pdb; pdb.set_trace()

# extract anomaly folder
FEAT_PARENT_FOLDER = PARENT_FOLDER / '3D_features'
FEAT_ANOM_FOLDER = FEAT_PARENT_FOLDER / 'Anomaly-Videos'

FEAT_ANOM_FOLDER.mkdir(parents=True, exist_ok=True)

for anom in ANOM_FOLDER.iterdir():
    anom_type = anom.name

    # create feature folder for this type

    for vid_file in anom.iterdir():
		vid_file_name = vid_file.name

		# create temporary csv file tmp.csv
		csvfile = 'tmp.csv'
		write_csv_for_lmdb(vid_file, csvfile)

		# extract lmdb file format
		# TODO:
		subprocess.check_output(CMD_1, shell=True)
		# extract features
		feat_path = FEAT_ANOM_FOLDER / (vid_file_name+'.pkl')
		# TODO
		CMD_2 = CMD_2_tmp.format(feat_path.__str__())
		subprocess.check_output(CMD_2, shell=True)
