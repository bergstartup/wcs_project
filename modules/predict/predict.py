import pandas as pd
import pickle
import os

# based on the kaggle notebooks:
# 1. https://www.kaggle.com/code/adnanshikh/listen-to-secrets-in-your-data/notebook
# 2. https://www.kaggle.com/code/thanakr/first-project-store-sales/notebook

class Predict:
    def predict(self, dataset_path, output_path):
        # read the inputs from the pickle file
        self.x_test = pd.read_pickle(f"{dataset_path}/x_test.pkl")
        self.sample = pd.read_pickle(f"{dataset_path}/sample.pkl")

        # load the model from pickle file
        self.ln = pickle.load(open(f"{dataset_path}/lnmodel.pkl", "rb"))

        self.y_pred = self.ln.predict(self.x_test)  # predict y_pred
        self.sample["sales"] = self.y_pred  # set sales column to y_pred
        self.sample.to_csv(
            os.path.join(output_path, "submission.csv"), index=False
        )  # save the submission file

def predict(dataset_path="../data", output_path="../output"):
    predict = Predict()
    predict.predict(dataset_path, output_path)

# predict("../data", "../output")