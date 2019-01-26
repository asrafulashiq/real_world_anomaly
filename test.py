import logging
import argparse
import glob
import os
from utils import load_model


# set logging
logging.basicConfig()
log = logging.getLogger("test")
log.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Testing anomaly detection")
    parser.add_argument('weight', type=str, default=None,
                        help="model weight path")
    parser.add_argument('model', type=str, default=None,
                        help="model.json path")
    args = parser.parse_args()
    logging.info(args)

    model_path = args.model
    weight_path = args.weight
    if model_path is None:
        model_path = "./model/trained_model/model.json"
    if weight_path is None:
        list_weights = glob.glob('./model/trained_model/weights*')
        # get latest weight path
        weight_path = max(list_weights, key=os.path.getctime)

    model = load_model(model_path, weight_path=weight_path)
