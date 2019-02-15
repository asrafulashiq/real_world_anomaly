import os
import shutil
from helper import *
from pathlib2 import Path
import platform
import subprocess

# overwrite already existed .pkl file
do_overwrite = False


if platform.node() == 'ash-ubuntu':
	PARENT_FOLDER = Path('/media/ash/New Volume/dataset/UCF_crime/')
else:
	PARENT_FOLDER = Path('/home/islama6a/local/UCF_crime')
	import sys
	sys.path.append("/home/islama6a/local/pytorch/build")

ANOM_FOLDER = PARENT_FOLDER / 'Anomaly-Videos'

assert PARENT_FOLDER.exists()

if os.path.exists('tmp_lmdb_data'):
	shutil.rmtree('tmp_lmdb_data')

CMD_1 = '''
python data/create_video_db.py \
--list_file=tmp.csv \
--output_file=tmp_lmdb_data \
--use_list=1 --use_video_id=1 --use_start_frame=1
'''

model_name = "c3d"
model_path = "./model/c3d_l16.pkl"

CMD_2_tmp = '''
python  tools/extract_features.py \
--test_data=tmp_lmdb_data \
--model_name={model_name} --model_depth=18 --clip_length_rgb=16 \
--gpus=0 \
--batch_size=4 \
--load_model_path={load_model_path} \
--output_path={output_path} \
--features=fc6 \
--sanity_check=0 --get_video_id=1 --use_local_file=1 --num_labels=1 && \
rm -rf tmp_lmdb_data && \
rm tmp.csv
'''

# import pdb; pdb.set_trace()

# extract anomaly folder
FEAT_PARENT_FOLDER = PARENT_FOLDER / 'C3D_features'
FEAT_ANOM_FOLDER = FEAT_PARENT_FOLDER / 'Anomaly-Videos'

FEAT_ANOM_FOLDER.mkdir(parents=True, exist_ok=True)


for anom in ANOM_FOLDER.iterdir():
    anom_type = anom.name
    anom_type_folder = FEAT_ANOM_FOLDER #/ anom_type
    #anom_type_folder.mkdir(exist_ok=True, parents=True)

	# create feature folder for this type

    for vid_file in sorted(anom.iterdir()):
		vid_file_name = vid_file.name

		feat_path = anom_type_folder  / (vid_file_name+'.pkl')

		if not do_overwrite and feat_path.exists():
			print("{} exists".format(feat_path))
			continue

		# create temporary csv file tmp.csv
		csvfile = 'tmp.csv'
		write_csv_for_lmdb(vid_file, csvfile)

		# extract lmdb file format
		subprocess.check_output(CMD_1, shell=True)
		# extract features
		CMD_2 = CMD_2_tmp.format(model_name=model_name,load_model_path=model_path,
								output_path=str(feat_path))
		subprocess.check_output(CMD_2, shell=True)
		# shutil.rmtree('tmp_lmdb_data')
		# asdasdasdas


# extract normal folder
normal_test_train = ['Training-Normal-Videos',
                       'Testing_Normal_Videos_Anomaly']

for normal_folder in normal_test_train:
	feat_normal_fldr = FEAT_PARENT_FOLDER / normal_folder
	feat_normal_fldr.mkdir(parents=True, exist_ok=True)

	normal = PARENT_FOLDER / normal_folder

	for vid_file in sorted(normal.iterdir()):
		vid_file_name = vid_file.name

		feat_path = feat_normal_fldr / (vid_file_name+'.pkl')
		if not do_overwrite and feat_path.exists():
			print("{} exists".format(feat_path))
			continue

		# create temporary csv file tmp.csv
		csvfile = 'tmp.csv'
		write_csv_for_lmdb(vid_file, csvfile)

		# extract lmdb file format
		subprocess.check_output(CMD_1, shell=True)
		CMD_2 = CMD_2_tmp.format(model_name=model_name, load_model_path=model_path,
                                 output_path=str(feat_path))
		subprocess.check_output(CMD_2, shell=True)
		# shutil.rmtree('tmp_lmdb_data')
