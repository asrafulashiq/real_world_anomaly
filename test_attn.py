import logging
import argparse
import glob
import os
from utils import load_model
from dataloader import load_one_video
from pathlib import Path
import pickle
from keras import Model
import numpy as np


# set logging
logging.basicConfig()
log = logging.getLogger("test")
log.setLevel(logging.INFO)


# def main():
parser = argparse.ArgumentParser(description="Testing anomaly detection")
parser.add_argument('--weight', type=str,
                    default="./model/trained_model/C3D_attn/",
                    help="model weight path")
parser.add_argument('--model', type=str,
                    default="./model/trained_model/C3D_attn/model.json",
                    help="model.json path")
parser.add_argument("--mini", type=str, default="false",
                    help="Whether to use mini data")
parser.add_argument("--pred", type=str,
                    default="./results/predictions/C3D_attn/",
                    help="path to save predictions")
parser.add_argument("--model-type", dest='model_type',
                    type=str, default="c3d-attn",
                    help="Extract C3D features?")
args = parser.parse_args()
log.info(args)

if args.model_type == 'c3d' or args.model_type == 'c3d-attn':
    flag_split = ""
elif args.model_type == '3d':
    flag_split = "_3d"

if args.mini == "true":
    flag_mini = "_mini"
else:
    flag_mini = ""

test_list = os.environ['HOME']+'/dataset/UCF_crime/' +\
    'custom_split'+flag_split+'/Custom_test_split' + flag_mini + '.txt'

pred_path = Path(args.pred)
pred_path.mkdir(parents=True, exist_ok=True)

assert os.path.exists(test_list), \
    "test list file does not exist"
model_path = args.model
weight_path = args.weight
if os.path.isdir(args.model):
    model_path = os.path.join(args.model, 'model.json')
else:
    model_path = args.model
assert os.path.exists(model_path)

if os.path.isdir(args.weight):
    list_weights = glob.glob(args.weight + 'weights*')
    # get latest weight path
    weight_path = max(list_weights, key=os.path.getctime)
else:
    assert os.path.exists(args.weight)
    weight_path = args.weight

log.info(f"Weight path: {weight_path}")

model = load_model(model_path, weight_path=weight_path)

# model.output = model.get_layer(name='score')

pred_model = Model(inputs=model.input, outputs=[
    model.output, model.get_layer(name='score').output
])


for vid_name, _input in load_one_video(test_list, log=log):
    inp = np.expand_dims(_input, axis=0)
    pred_all, pred = pred_model.predict(inp)
    # import pdb; pdb.set_trace()
    fname = pred_path / (vid_name + ".pkl")
    with fname.open("wb") as fp:
        pickle.dump(pred.squeeze(), fp, protocol=pickle.HIGHEST_PROTOCOL)


# if __name__ == "__main__":
#     main()
