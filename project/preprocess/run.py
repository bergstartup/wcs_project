#!/usr/bin/python3
import os
import sys
import json
import yaml

from preprocess import preprocess

def prepr(dataset_path="./data"):
    preprocess(dataset_path=dataset_path)
    return "Preprocess function called and returned"


if __name__ == "__main__":
    command = sys.argv[1]
    params = []
    
    if command == "prepr":
        params.append(json.loads(os.environ["DATASET_PATH"]))
        result = prepr(*params)

    # Print the result with the YAML package
    print(yaml.dump({ "resultOutput": result }))