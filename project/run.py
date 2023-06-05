#!/usr/bin/python3
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn import linear_model, preprocessing
import pickle
import numpy as np
import sys
import json
import yaml
from visualize import visualize
from train import train_model, train_test_split
from preprocess import preprocess
from predict import predict

def visual(dataset_path="./data", output_path="./output/images", cmd="both"):
    visualize(dataset_path=dataset_path, output_path=output_path, cmd=cmd)
    return "Visualization function called and returned"

def train_m(dataset_path="./data"):
    train_model(dataset_path=dataset_path)
    return "Train model function called and returned"

def train_t_spl(dataset_path="./data"):
    train_test_split(dataset_path=dataset_path)
    return "Train_test_split function called and returned"

def prepr(dataset_path="./data"):
    preprocess(dataset_path=dataset_path)
    return "Preprocess function called and returned"

def pred(dataset_path="./data", output_path="./output"):
    predict(dataset_path=dataset_path, output_path=output_path)
    return "Predict function called and returned"


if __name__ == "__main__":
    command = sys.argv[1]
    params = []
    if command == "visual":
        params.append(json.loads(os.environ["DATASET_PATH"]))
        params.append(json.loads(os.environ["OUTPUT_PATH"])+"output/images")
        params.append(json.loads(os.environ["CMD"]))
        result = visual(*params)

    elif command == "train_m":
        params.append(json.loads(os.environ["DATASET_PATH"]))
        result = train_m(*params)

    elif command == "train_t_spl":
        params.append(json.loads(os.environ["DATASET_PATH"]))
        result = train_t_spl(*params)

    elif command == "prepr":
        params.append(json.loads(os.environ["DATASET_PATH"]))
        result = prepr(*params)

    elif command == "pred":
        params.append(json.loads(os.environ["DATASET_PATH"]))
        params.append(json.loads(os.environ["OUTPUT_PATH"])+"output")
        result = pred(*params)

    # Print the result with the YAML package
    print(yaml.dump({ "resultOutput": result }))