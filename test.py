import logging
import argparse
import glob
import os
from utils import load_model
from dataloader import load_one_video
from pathlib import Path
import pickle

# set logging
logging.basicConfig()
log = logging.getLogger("test")
log.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Testing anomaly detection")
    parser.add_argument('--weight', type=str, default=None,
                        help="model weight path")
    parser.add_argument('--model', type=str, default=None,
                        help="model.json path")
    args = parser.parse_args()
    log.info(args)

    test_list = os.environ['HOME']+'/dataset/UCF_crime/' +\
        'custom_split/Custom_test_split.txt'
    pred_path = Path('./results/predictions/')
    pred_path.mkdir(parents=True, exist_ok=True)

    assert os.path.exists(test_list), \
        "test list file does not exist"
    model_path = args.model
    weight_path = args.weight
    if model_path is None:
        model_path = "./model/trained_model/model.json"
    if weight_path is None:
        list_weights = glob.glob('./model/trained_model/weights*')
        # get latest weight path
        weight_path = max(list_weights, key=os.path.getctime)

    log.info(f"Weight path: {weight_path}")

    # model = load_model(model_path, weight_path=weight_path)
    import utils
    model = load_model(model_path)
    model = utils.load_weights_from_mat(model, './weights_L1L2.mat')

    for vid_name, input in load_one_video(test_list, log=log):
        pred = model.predict_on_batch(input)
        # import pdb; pdb.set_trace()
        fname = pred_path / (vid_name + ".pkl")
        with fname.open("wb") as fp:
            pickle.dump(pred, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
