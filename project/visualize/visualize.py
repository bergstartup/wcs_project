import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# based on the kaggle notebooks:
# 1. https://www.kaggle.com/code/adnanshikh/listen-to-secrets-in-your-data/notebook
# 2. https://www.kaggle.com/code/thanakr/first-project-store-sales/notebook

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
            "Sales are higher at holidays than on workdays.",
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


def visualize(dataset_path="./data", output_path="./output/images", cmd="both"):
    visualize = Visualize()
    visualize.set_config()

    os.makedirs(output_path)
    if cmd == "holiday":
        visualize.visualize_sales_vs_holiday(dataset_path, output_path)

    elif cmd == "oil":
        visualize.visualize_sales_vs_oil(dataset_path, output_path)

    elif cmd == "both":
        visualize.visualize_sales_vs_holiday(dataset_path, output_path)
        visualize.visualize_sales_vs_oil(dataset_path, output_path)


# visualize("./data", "./output/images", "both")