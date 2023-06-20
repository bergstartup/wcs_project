import pandas as pd
from sklearn import linear_model
import pickle

# based on the kaggle notebooks:
# 1. https://www.kaggle.com/code/adnanshikh/listen-to-secrets-in-your-data/notebook
# 2. https://www.kaggle.com/code/thanakr/first-project-store-sales/notebook

class TrainTestSplit:
    def train_test_split(self, dataset_path, result_path):
        # read the inputs from the pickle file
        self.inputs = pd.read_pickle(f"{dataset_path}/inputs.pkl")

        self.inputs = self.inputs.sort_index()
        self.y_train = self.inputs.loc[
            "2013-01-01":"2017-08-15", "sales"
        ]  # set y_train
        self.x_train = self.inputs.loc["2013-01-01":"2017-08-15"].drop(
            ["sales", "id"], axis=1
        )  # set x_train
        self.x_train = self.x_train.reset_index()
        self.x_train = self.x_train.set_index(["date", "store_nbr", "family"])

        self.x_test = self.inputs.loc["2017-08-16":]
        self.x_test.drop(["sales", "id"], axis=1, inplace=True)
        self.x_test = self.x_test.reset_index()
        self.x_test = self.x_test.set_index(["date", "store_nbr", "family"])

        # store the test and train data in pickle files
        self.x_train.to_pickle(f"{result_path}/x_train.pkl")
        self.x_test.to_pickle(f"{result_path}/x_test.pkl")
        self.y_train.to_pickle(f"{result_path}/y_train.pkl")

        # Also copy/paste the sample necessary by the prediction
        sample = pd.read_pickle(f"{dataset_path}/sample.pkl")
        sample.to_pickle(f"{result_path}/sample.pkl")


class TrainModel:
    def train_model(self, dataset_path, result_path):
        # read the inputs from the pickle file
        self.x_train = pd.read_pickle(f"{dataset_path}/x_train.pkl")
        self.y_train = pd.read_pickle(f"{dataset_path}/y_train.pkl")

        self.ln = linear_model.LinearRegression()  # create linear regression model
        self.ln.fit(self.x_train, self.y_train)  # fit model to x_train and y_train

        # store the model in pickle file
        pickle.dump(self.ln, open(f"{result_path}/lnmodel.pkl", "wb"))


def train_test_split(dataset_path="./data", result_path="/result"):
    train_test_split = TrainTestSplit()
    train_test_split.train_test_split(dataset_path, result_path)


def train_model(dataset_path="./data", result_path="/result"):
    train_model = TrainModel()
    train_model.train_model(dataset_path, result_path)


# train_test_split("./data")
# train_model("./data")