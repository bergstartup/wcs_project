import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# based on the kaggle notebooks:
# 1. https://www.kaggle.com/code/adnanshikh/listen-to-secrets-in-your-data/notebook
# 2. https://www.kaggle.com/code/thanakr/first-project-store-sales/notebook


# a class for preprocessing data
class Preprocess:
    def read_data(self, dataset_path):
        self.train_data = pd.read_csv(f"{dataset_path}/train.csv")
        self.store_data = pd.read_csv(f"{dataset_path}/stores.csv")
        self.holiday_data = pd.read_csv(
            f"{dataset_path}/holidays_events.csv",
            index_col="date",
            parse_dates=["date"],
        )  # Set the 'date' column as the index and parse it as dates
        self.oil_data = pd.read_csv(f"{dataset_path}/oil.csv", parse_dates=["date"])
        self.transaction_data = pd.read_csv(f"{dataset_path}/transactions.csv")
        self.sample = pd.read_csv(f"{dataset_path}/sample_submission.csv")
        self.test_data = pd.read_csv(f"{dataset_path}/test.csv")

    def preprocess_data(self, dataset_path):
        self.process_train = self.train_data.copy()
        self.process_train["date"] = pd.to_datetime(
            self.process_train["date"]
        )  # Convert the 'date' column to datetime format
        self.process_train = self.process_train.set_index("date")
        self.process_train = self.process_train.drop("id", axis=1)
        self.process_train[["store_nbr", "family"]].astype(
            "category"
        )  # Convert the 'store_nbr' and 'family' columns to the category data type

        self.preprocess_daily_sales()
        self.preprocess_location_data()
        self.preprocess_total_sales()
        self.preprocess_transactions()
        self.preprocess_holiday_data()

        # store process_train data in a pickle file
        self.process_train.to_pickle(f"{dataset_path}/process_train.pkl")
        self.holiday_data.to_pickle(f"{dataset_path}/holiday_data.pkl")
        self.inputs.to_pickle(f"{dataset_path}/inputs.pkl")
        self.oil_data.to_pickle(f"{dataset_path}/oil_data.pkl")
        self.sample.to_pickle(f"{dataset_path}/sample.pkl")

    def preprocess_daily_sales(self):
        daily_sale_dict = {}
        for (
            i
        ) in (
            self.process_train.store_nbr.unique()
        ):  # Iterate over unique store numbers in the processed train data
            daily_sale = self.process_train[self.process_train["store_nbr"] == i]
            daily_sale_dict[i] = daily_sale

        self.daily_store_sale_dict = {}
        for (
            i
        ) in (
            daily_sale_dict.keys()
        ):  # Iterate over the keys (store numbers) in the daily_sale_dict dictionary
            self.daily_store_sale_dict[i] = (
                daily_sale_dict[i]
                .groupby(["date", "store_nbr"])
                .sales.sum()
                .to_frame()  # Group the daily sale data by date and store number, and calculate the sum of sales
            )

        del daily_sale_dict

        for i in self.daily_store_sale_dict.keys():
            self.daily_store_sale_dict[i] = self.daily_store_sale_dict[i].droplevel(1)

    def preprocess_location_data(self):
        self.test_data["sales"] = 0
        self.test_data = self.test_data[
            ["id", "date", "store_nbr", "family", "sales", "onpromotion"]
        ]

        self.total_sales = self.process_train.sales.groupby("date").sum()

        self.process_train["id"] = self.process_train.reset_index().index
        self.process_train = self.process_train.reset_index()
        self.process_train = self.process_train[
            ["id", "date", "store_nbr", "family", "sales", "onpromotion"]
        ]

        # Merge the processed train data and test data
        merged_train = pd.concat([self.process_train, self.test_data])
        merged_train = merged_train.set_index(["date", "store_nbr", "family"])

        del self.test_data

        store_location = self.store_data.drop(
            ["state", "type", "cluster"], axis=1
        )  # the city column is the same scale as holiday location.
        store_location = store_location.set_index("store_nbr")
        store_location = pd.get_dummies(store_location, prefix="store_loc_")

        self.inputs = merged_train.reset_index().merge(
            store_location,
            how="outer",
            left_on="store_nbr",
            right_on=store_location.index,
        )  # merge the store location data with the merged train data

        del store_location
        del merged_train

    def preprocess_total_sales(self):
        total_sales_to_scale = pd.DataFrame(
            index=pd.date_range(
                start="2013-01-01", end="2017-08-31"
            )  # Create a date range from 2013-01-01 to 2017-08-31
        )
        total_sales_to_scale = total_sales_to_scale.merge(
            self.total_sales,
            how="left",
            left_index=True,
            right_index=True,  # Merge the total sales data with the total sales to scale data
        )
        total_sales_to_scale = total_sales_to_scale.rename(
            columns={"sales": "national_sales"}
        )

        mmScale = preprocessing.MinMaxScaler()  # Scale the national sales data
        mmScale.fit(
            total_sales_to_scale["national_sales"].to_numpy().reshape(-1, 1)
        )  # Reshape the national sales data to a 2D array
        total_sales_to_scale["scaled_nat_sales"] = mmScale.transform(
            total_sales_to_scale["national_sales"]
            .to_numpy()
            .reshape(-1, 1)  # Reshape the national sales data to a 2D array
        )

        lags = [16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28]
        for lag in lags:
            total_sales_to_scale[
                "nat_scaled_sales_lag{}".format(
                    lag
                )  # Create lagged features for the scaled national sales data
            ] = total_sales_to_scale["scaled_nat_sales"].shift(lag)
        total_sales_to_scale = total_sales_to_scale.drop(
            ["national_sales", "scaled_nat_sales"],
            axis=1,  # Drop the national sales and scaled national sales columns
        )

        self.inputs["date"] = pd.to_datetime(
            self.inputs["date"]
        )  # Convert the 'date' column to datetime format
        self.inputs = self.inputs.merge(
            total_sales_to_scale.reset_index(),
            how="left",
            left_on="date",
            right_on="index",
        )  # Merge the total sales to scale data with the inputs data
        del total_sales_to_scale

        self.inputs.drop(["index"], axis=1, inplace=True)  # Drop the index column

    def preprocess_transactions(self):
        lags = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14]
        for lag in lags:
            self.inputs["store_fam_sales_lag_{}".format(lag)] = self.inputs[
                "sales"
            ].shift(
                lag
            )  # Create lagged features for the sales data

        self.transactions = self.transaction_data.copy()
        self.transactions = self.transactions.set_index("date")
        self.transactions.index = pd.to_datetime(
            self.transactions.index
        )  # Convert the index to datetime format
        self.transactions.reset_index()

        store_nbr = range(1, 55)
        dates = pd.date_range(
            "2013-01-01", "2017-08-31"
        )  # Create a date range from 2013-01-01 to 2017-08-31
        mul_index = pd.MultiIndex.from_product(
            [dates, store_nbr], names=["date", "store_nbr"]
        )  # Create a multi-index from the date range and store numbers
        df = pd.DataFrame(index=mul_index)
        df.reset_index()

        df_transaction = df.reset_index().merge(
            self.transactions.reset_index(),
            how="left",
            left_on=["date", "store_nbr"],
            right_on=["date", "store_nbr"],
        )  # Merge the transactions data with the multi-index data
        df_transaction.fillna(0, inplace=True)
        lags = [21, 22, 28]
        for lag in lags:
            df_transaction["trans_lag_{}".format(lag)] = df_transaction[
                "transactions"
            ].shift(
                lag
            )  # Create lagged features for the transactions data
        df_transaction = df_transaction.drop("transactions", axis=1)
        df_transaction = df_transaction.fillna(0)
        self.inputs = self.inputs.merge(
            df_transaction,
            how="left",
            left_on=["date", "store_nbr"],
            right_on=["date", "store_nbr"],
        )  # Merge the transactions data with the inputs data

        del df_transaction

    def preprocess_holiday_data(self):
        # pre-processing holiday data
        ny_date = pd.to_datetime(
            [
                "2012-01-01",
                "2013-01-01",
                "2014-01-01",
                "2015-01-01",
                "2016-01-01",
                "2017-01-01",
                "2018-01-01",
            ]
        )  # Create a date range from 2012-01-01 to 2018-01-01
        cm_date = pd.to_datetime(
            [
                "2012-12-25",
                "2013-12-25",
                "2014-12-25",
                "2015-12-25",
                "2016-12-25",
                "2017-12-25",
                "2018-12-25",
            ]
        )  # Create a date range from 2012-12-25 to 2018-12-25

        for date in ny_date:
            self.holiday_data.loc[date] = [
                "Holiday",
                "National",
                "Ecuador",
                "New Year day",
                "False",
            ]  # Add New Year day to the holiday data
        for date in cm_date:
            self.holiday_data.loc[date] = [
                "Holiday",
                "National",
                "Ecuador",
                "Christmas day",
                "False",
            ]  # Add Christmas day to the holiday data
        self.holiday_data = self.holiday_data.sort_index()

        self.calendar = pd.DataFrame(
            index=pd.date_range("2013-01-01", "2017-08-31")
        )  # Create a date range from 2013-01-01 to 2017-08-31
        self.calendar = self.calendar.join(self.holiday_data).fillna(0)
        self.calendar["dow"] = (
            self.calendar.index.dayofweek + 1
        )  # Create a day of week column
        self.calendar["workday"] = True  # Create a workday column
        self.calendar.loc[
            self.calendar["dow"] > 5, "workday"
        ] = False  # make work_day false for sat and sun (6/7 in dow)
        self.calendar.loc[
            (self.calendar["type"] == "Holiday")
            & (self.calendar["locale"].str.contains("National")),
            "workday",
        ] = False  # make work_day false for national holidays
        self.calendar.loc[
            (self.calendar["type"] == "Additional")
            & (self.calendar["locale"].str.contains("National")),
            "workday",
        ] = False  # make work_day false for additional national holidays
        self.calendar.loc[
            (self.calendar["type"] == "Bridge")
            & (self.calendar["locale"].str.contains("National")),
            "workday",
        ] = False  # make work_day false for bridge national holidays
        self.calendar.loc[
            (self.calendar["type"] == "Transfer")
            & (self.calendar["locale"].str.contains("National")),
            "workday",
        ] = False  # make work_day false for transfer national holidays
        self.calendar.loc[
            self.calendar["type"] == "Work Day", "workday"
        ] = True  # make work_day true for work days
        self.calendar.where(
            self.calendar["transferred"] == True
        ).dropna()  # drop rows where transferred is true
        self.calendar.loc[
            (self.calendar["transferred"] == True), "workday"
        ] = True  # make work_day true for transferred holidays
        self.calendar.where(
            self.calendar["description"].str.contains("futbol")
        ).dropna()  # drop rows where description contains futbol
        self.calendar["is_football"] = 0  # create a is_football column
        self.calendar["is_eq"] = 0  # create a is_eq column
        self.calendar.loc[
            (self.calendar["is_football"] == 0)
            & (self.calendar["description"].str.contains("futbol")),
            "is_football",
        ] = 1  # make is_football true for futbol
        self.calendar.loc[
            (self.calendar["is_eq"] == 0)
            & (self.calendar["description"].str.contains("Terremoto")),
            "is_eq",
        ] = 1  # make is_eq true for Terremoto
        self.calendar.loc[
            self.calendar["is_football"] == 1, "description"
        ] = "football"  # make description football for is_football true
        self.calendar.loc[
            self.calendar["is_eq"] == 1, "description"
        ] = "earthquake"  # make description earthquake for is_eq true
        self.calendar["workday"] = self.calendar["workday"].map(
            {False: 0, True: 1}
        )  # map workday to 0 and 1
        self.calendar["transferred"] = self.calendar["transferred"].map(
            {"False": 0, False: 0, True: 1}
        )  # map transferred to 0 and 1
        self.calendar["is_ny"] = 0  # create a is_ny column
        self.calendar["is_christmas"] = 0  # create a is_christmas column
        self.calendar["is_shopping"] = 0  # create a is_shopping column
        self.calendar.loc[
            self.calendar["description"] == "New Year day", "is_ny"
        ] = 1  # make is_ny true for New Year day
        self.calendar.loc[
            self.calendar["description"] == "Christmas day", "is_christmas"
        ] = 1  # make is_christmas true for Christmas day
        self.calendar.loc[
            self.calendar["description"] == "Black Friday", "is_shopping"
        ] = 1  # make is_shopping true for Black Friday
        self.calendar.loc[
            self.calendar["description"] == "Cyber Monday", "is_shopping"
        ] = 1  # make is_shopping true for Cyber Monday
        self.calendar = self.calendar.drop(
            ["type", "locale"], axis=1
        )  # drop type and locale
        locale_dummy = pd.get_dummies(
            self.calendar["locale_name"], prefix="holiday_"
        )  # create dummies for locale_name
        self.calendar = locale_dummy.join(
            self.calendar, how="left"
        )  # join locale_name dummies to calendar
        self.calendar = self.calendar.drop("locale_name", axis=1)
        del locale_dummy

        calendar_checkpoint = self.calendar
        calendar_checkpoint = calendar_checkpoint.drop(
            "description", axis=1
        )  # don't need description anymore we have dummied them all
        calendar_checkpoint = calendar_checkpoint[
            ~calendar_checkpoint.index.duplicated(keep="first")
        ]  # drop duplicate rows
        calendar_checkpoint = calendar_checkpoint.iloc[:, 1:-1]
        self.inputs = self.inputs.merge(
            calendar_checkpoint,
            how="left",
            left_on=["date"],
            right_on=calendar_checkpoint.index,
        )  # join calendar_checkpoint to inputs

        pd.set_option("display.max_rows", None)
        pd.reset_option(
            "display.max_rows", "display.max_columns"
        )  # reset display options
        self.inputs.dropna(inplace=True)
        self.inputs = self.inputs.set_index("date")  # set index to date


class Visualize:
    def set_config(self):
        ## Set Plot Parameters
        sns.set(color_codes=True)
        plt.style.use("seaborn-whitegrid")
        plt.rc("figure", autolayout=True, figsize=(11, 5))
        plt.rc(
            "axes",
            labelweight="bold",
            labelsize="large",
            titleweight="bold",
            titlesize=14,
            titlepad=10,
        )
        self.plot_params = dict(
            color="0.75",
            style=".-",
            markeredgecolor="0.25",
            markerfacecolor="0.25",
            legend=False,
        )

    def visualize_sales_vs_holiday(self, dataset_path, output_path):
        # the required files should be from the pickle file
        self.process_train = pd.read_pickle(f"{dataset_path}/process_train.pkl")
        self.holiday_data = pd.read_pickle(f"{dataset_path}/holiday_data.pkl")
        self.inputs = pd.read_pickle(f"{dataset_path}/inputs.pkl")

        # calculate avg sales of each day from the input data by grouping by date
        avg_sales = self.process_train.groupby("date")["sales"].mean().to_frame()

        # Visualize National Holidays vs Avg Sales
        NRHolidays = self.holiday_data.loc[self.holiday_data["locale"] != "Local", :]
        NRHolidays_avg_sales = avg_sales.reset_index().merge(
            NRHolidays, on="date", how="left"
        )
        x_cor = NRHolidays_avg_sales.loc[
            NRHolidays_avg_sales["type"].notna(), "date"
        ].values
        y_cor = NRHolidays_avg_sales.loc[
            NRHolidays_avg_sales["type"].notna(), "sales"
        ].values
        _ = avg_sales["sales"].plot(**self.plot_params)
        _ = plt.plot_date(x_cor, y_cor, color="C3", label="National / Regional Holiday")
        _ = plt.xlabel("Date")
        _ = plt.ylabel("Avg. Sales")
        _ = plt.title("Avg. Sales At National and Regional Holidays Only")
        _ = plt.legend()

        # save the image to the images folder
        plt.savefig(
            os.path.join(
                output_path,
                "avg_sales_at_national_and_regional_holidays_only.png",
            )
        )

        # clear the plot
        plt.clf()

        # Visualize Local Holidays vs Avg Sales
        LHolidays = self.holiday_data.loc[self.holiday_data["locale"] == "Local", :]
        LHolidays_avg_sales = avg_sales.reset_index().merge(
            LHolidays, on="date", how="left"
        )
        x_cor = LHolidays_avg_sales.loc[
            LHolidays_avg_sales["type"].notna(), "date"
        ].values
        y_cor = LHolidays_avg_sales.loc[
            LHolidays_avg_sales["type"].notna(), "sales"
        ].values
        _ = avg_sales["sales"].plot(**self.plot_params)
        _ = plt.plot_date(x_cor, y_cor, color="C3", label="Local Holiday")
        _ = plt.xlabel("Date")
        _ = plt.ylabel("Avg. Sales")
        _ = plt.title("Avg. Sales At Local Holidays Only")
        _ = plt.legend()

        # save the image to the images folder
        plt.savefig(
            os.path.join(
                output_path,
                "avg_sales_at_local_holidays_only.png",
            )
        )

        # clear the plot
        plt.clf()

        # Visualize Work Days vs Avg Sales
        workday_avg_sales = (
            self.inputs[self.inputs["workday"] == True].groupby("date")["sales"].mean()
        )
        # remove the last test data added to the data as they are not complete
        workday_avg_sales = workday_avg_sales[:-12]

        holiday_avg_sales = (
            self.inputs[self.inputs["workday"] == False].groupby("date")["sales"].mean()
        )
        # remove the last test data added to the data as they are not complete
        holiday_avg_sales = holiday_avg_sales[:-4]

        _ = workday_avg_sales.plot(
            color="0.4", style=".", legend=True, label="Workday Avg. Sales"
        )
        _ = holiday_avg_sales.plot(
            color="red", style=".", legend=True, label="Holiday Avg. Sales"
        )
        plt.xlabel("Date")
        plt.ylabel("Avg. Sales")
        plt.title("Avg. Sales At Workdays and Holidays")
        # add a caption
        plt.figtext(
            0.5,
            0.01,
            "Sales are higher at workdays than holidays.",
            wrap=True,
            horizontalalignment="center",
            fontsize=12,
        )

        # save the image to the images folder
        plt.savefig(
            os.path.join(
                output_path,
                "avg_sales_at_workdays_and_holidays.png",
            )
        )

        # clear the plot
        plt.clf()

    def visualize_sales_vs_oil(self, dataset_path, output_path):
        # the required files should be from the pickle file
        self.process_train = pd.read_pickle(f"{dataset_path}/process_train.pkl")
        self.oil_data = pd.read_pickle(f"{dataset_path}/oil_data.pkl")

        # merge oil data with the input data
        # convert the date column to datetime in oil data
        self.oil_data["date"] = pd.to_datetime(self.oil_data["date"])
        self.oil_data.set_index("date", inplace=True)

        self.process_copy = self.process_train.reset_index().set_index("date").copy()
        self.process_copy = pd.merge(
            left=self.process_copy,
            right=self.oil_data,
            left_index=True,
            right_index=True,
            how="left",
        )
        self.process_copy.rename(columns={"dcoilwtico": "oil_price"}, inplace=True)
        self.process_copy["oil_price"] = (
            self.process_copy["oil_price"]
            .fillna(method="ffill")
            .fillna(method="bfill")
            .astype(float)
        )  # fill the missing values with the previous value

        # calculate avg sales of each day from the input data by grouping by date
        avg_sales = (
            self.process_copy.groupby(["date", "oil_price"])["sales"]
            .mean()
            .reset_index()
        )

        # # plotting the avg sales vs oil price
        sns.regplot(
            data=avg_sales,
            x="oil_price",
            y="sales",
            scatter_kws={"color": "0.4"},
            line_kws={"color": "red", "linewidth": 3},
        )
        plt.xlabel("Oil Price")
        plt.ylabel("Avg. Sales")
        # add caption
        plt.figtext(
            0.5,
            0.01,
            "Sales are increasing with the decrease in oil price.",
            wrap=True,
            horizontalalignment="center",
            fontsize=12,
        )
        plt.title("Avg. Sales vs Oil Price")

        # save the image to the images folder
        plt.savefig(
            os.path.join(
                output_path,
                "avg_sales_vs_oil_price.png",
            )
        )

        # clear the plot
        plt.clf()

        # check if the sales price is affected by the oil price
        print(
            "Correlation between sales and oil price: ",
            avg_sales["sales"].corr(avg_sales["oil_price"]),
        )
        print("Sales are increasing with the decrease in oil price.")


class TrainTestSplit:
    def train_test_split(self, dataset_path):
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
        self.x_train.to_pickle(f"{dataset_path}/x_train.pkl")
        self.x_test.to_pickle(f"{dataset_path}/x_test.pkl")
        self.y_train.to_pickle(f"{dataset_path}/y_train.pkl")

class TrainModel:
    def train_model(self, dataset_path):
        # read the inputs from the pickle file
        self.x_train = pd.read_pickle(f"{dataset_path}/x_train.pkl")
        self.y_train = pd.read_pickle(f"{dataset_path}/y_train.pkl")

        self.ln = linear_model.LinearRegression()  # create linear regression model
        self.ln.fit(self.x_train, self.y_train)  # fit model to x_train and y_train
        
        # store the model in pickle file
        pickle.dump(self.ln, open(f"{dataset_path}/lnmodel.pkl", "wb"))

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


def preprocess(dataset_path="./data"):
    preprocess = Preprocess()
    preprocess.read_data(dataset_path)
    preprocess.preprocess_data(dataset_path)


def visualize(dataset_path="./data", output_path="./output/images", cmd="both"):
    visualize = Visualize()
    visualize.set_config()

    if cmd == "holiday":
        visualize.visualize_sales_vs_holiday(dataset_path, output_path)

    elif cmd == "oil":
        visualize.visualize_sales_vs_oil(dataset_path, output_path)

    elif cmd == "both":
        visualize.visualize_sales_vs_holiday(dataset_path, output_path)
        visualize.visualize_sales_vs_oil(dataset_path, output_path)

def train_test_split(dataset_path="./data"):
    train_test_split = TrainTestSplit()
    train_test_split.train_test_split(dataset_path)
    
def train_model(dataset_path="./data"):
    train_model = TrainModel()
    train_model.train_model(dataset_path)
    
def predict(dataset_path="./data", output_path="./output"):
    predict = Predict()
    predict.predict(dataset_path, output_path)

preprocess("./data")
visualize("./data", "./output/images", "both")
train_test_split("./data")
train_model("./data")
predict("./data", "./output")
