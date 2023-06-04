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

def pred(dataset_path="./data", output_path="./output"):
    predict(dataset_path=dataset_path, output_path=output_path)


if __name__ == "__main__":
    