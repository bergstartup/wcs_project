#!/usr/bin/python3
import os
import sys
import json
import yaml

from train import train_model, train_test_split
from predict import predict

def train_m(dataset_path="./data"):
    train_model(dataset_path=dataset_path)
    return "Train model function called and returned"

def train_t_spl(dataset_path="./data"):
    train_test_split(dataset_path=dataset_path)
    return "Train_test_split function called and returned"

def pred(model_path="./model", dataset_path="./data"):
    predict(model_path=model_path, dataset_path=dataset_path)
    return "Predict function called and returned"


if __name__ == "__main__":
    command = sys.argv[1]
    params = []
    if command == "train_m":
        params.append(json.loads(os.environ["DATASET_PATH"]))
        result = train_m(*params)

    elif command == "train_t_spl":
        params.append(json.loads(os.environ["DATASET_PATH"]))
        result = train_t_spl(*params)
        
    elif command == "pred":
        params.append(json.loads(os.environ["MODEL_PATH"]))
        params.append(json.loads(os.environ["DATASET_PATH"]))
        result = pred(*params)

    # Print the result with the YAML package
    # print(yaml.dump({ "resultOutput": result }))
