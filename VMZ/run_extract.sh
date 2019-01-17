python  tools/extract_features.py \
--test_data=tmp_lmdb_data \
--model_name=r2plus1d --model_depth=18 --clip_length_rgb=16 \
--gpus=2 \
--batch_size=4 \
--load_model_path=./model/r2.5d_d18_l16.pkl \
--output_path=my_features.pkl \
--features=softmax,final_avg,video_id \
--sanity_check=0 --get_video_id=1 --use_local_file=1 --num_labels=1