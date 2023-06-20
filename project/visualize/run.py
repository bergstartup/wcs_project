#!/usr/bin/python3
import os
import sys
import json
import yaml

from visualize import visualize

def visual(dataset_path="./data", output_path="./output/images", cmd="both"):
    visualize(dataset_path=dataset_path, output_path=output_path, cmd=cmd)
    return "Visualization function called and returned"

if __name__ == "__main__":
    command = sys.argv[1]
    params = []
    if command == "visual":
        params.append(json.loads(os.environ["DATASET_PATH"]))
        params.append("/result/output/images")
        params.append(json.loads(os.environ["CMD"]))
        result = visual(*params)

    # Print the result with the YAML package
    print(yaml.dump({ "resultOutput": result }))