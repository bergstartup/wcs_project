import numpy as np
import pandas as pd
from sklearn import preprocessing, linear_model
import os
import gc
import warnings


# a class for preprocessing data
class Preprocess:
    # a function to read data
    def read_data(self):
        self.holiday_data = pd.read_csv(
            "./data/holidays_events.csv", index_col="date", parse_dates=["date"]
        )
        self.oil_data = pd.read_csv("./data/oil.csv")
        self.sample = pd.read_csv("./data/sample_submission.csv")
        self.store_data = pd.read_csv("./data/stores.csv")
        self.test_data = pd.read_csv("./data/test.csv")
        self.train_data = pd.read_csv("./data/train.csv")
        self.transaction_data = pd.read_csv("./data/transactions.csv")

    # a function to preprocess data
    def preprocess_data(self):
        self.process_train = self.train_data.copy()
        self.process_train["date"] = pd.to_datetime(self.process_train["date"])
        self.process_train = self.process_train.set_index("date")
        self.process_train = self.process_train.drop("id", axis=1)
        self.process_train[["store_nbr", "family"]].astype("category")

        # pre-processing daily sales data
        daily_sale_dict = {}
        for i in self.process_train.store_nbr.unique():
            daily_sale = self.process_train[self.process_train["store_nbr"] == i]
            daily_sale_dict[i] = daily_sale

        self.daily_store_sale_dict = {}
        for i in daily_sale_dict.keys():
            self.daily_store_sale_dict[i] = (
                daily_sale_dict[i]
                .groupby(["date", "store_nbr"])
                .sales.sum()
                .to_frame()
            )

        del daily_sale_dict

        for i in self.daily_store_sale_dict.keys():
            self.daily_store_sale_dict[i] = self.daily_store_sale_dict[i].droplevel(1)

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

        merged_train = pd.concat([self.process_train, self.test_data])
        merged_train = merged_train.set_index(["date", "store_nbr", "family"])

        del self.test_data
        
        # pre-processing location data
        store_location = self.store_data.drop(['state','type','cluster'],axis=1) # the city column is the same scale as holiday location.
        store_location = store_location.set_index('store_nbr')
        store_location = pd.get_dummies(store_location,prefix='store_loc_')

        self.inputs = merged_train.reset_index().merge(store_location,how='outer',left_on='store_nbr',right_on=store_location.index)

        del store_location
        del merged_train
        
        # pre-processing total sales data
        # self.total_sales = self.process_train.sales.groupby("date").sum()
        total_sales_to_scale = pd.DataFrame(index=pd.date_range(start='2013-01-01',end='2017-08-31'))
        total_sales_to_scale = total_sales_to_scale.merge(self.total_sales,how='left',left_index=True,right_index=True)
        total_sales_to_scale = total_sales_to_scale.rename(columns={'sales':'national_sales'})

        mmScale = preprocessing.MinMaxScaler()
        mmScale.fit(total_sales_to_scale['national_sales'].to_numpy().reshape(-1,1))
        total_sales_to_scale['scaled_nat_sales'] = mmScale.transform(total_sales_to_scale['national_sales'].to_numpy().reshape(-1,1))

        lags= [16,17,18,19,20,21,22,23,24,27,28]
        for lag in lags:
            total_sales_to_scale['nat_scaled_sales_lag{}'.format(lag)] = total_sales_to_scale['scaled_nat_sales'].shift(lag)
        total_sales_to_scale = total_sales_to_scale.drop(['national_sales','scaled_nat_sales'],axis=1) 
        total_sales_to_scale.reset_index().tail() #reset index for ease of merge
        
        self.inputs['date'] = pd.to_datetime(self.inputs['date'])
        self.inputs = self.inputs.merge(total_sales_to_scale.reset_index(),how='left',left_on='date',right_on='index')
        del total_sales_to_scale
        
        self.inputs.drop(['index'],axis=1,inplace=True) #column named 'index' in dt format don't need it anymore

        # pre-processing transaction data
        lags = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14]
        for lag in lags:
            self.inputs['store_fam_sales_lag_{}'.format(lag)] = self.inputs['sales'].shift(lag)

        self.transactions = self.transaction_data.copy()
        self.transactions = self.transactions.set_index('date')
        self.transactions.index = pd.to_datetime(self.transactions.index)
        self.transactions.reset_index()
        
        store_nbr = range(1,55)
        dates = pd.date_range('2013-01-01','2017-08-31')
        mul_index = pd.MultiIndex.from_product([dates,store_nbr],names=['date','store_nbr'])
        df = pd.DataFrame(index=mul_index)
        df.reset_index()
        
        df_transaction = df.reset_index().merge(self.transactions.reset_index(),
                                        how='left',
                                        left_on=['date','store_nbr'],
                                        right_on=['date','store_nbr']
                                       )
        df_transaction.fillna(0, inplace=True)
        lags = [21,22,28]
        for lag in lags:
            df_transaction['trans_lag_{}'.format(lag)] = df_transaction['transactions'].shift(lag)
        df_transaction = df_transaction.drop('transactions',axis=1)
        df_transaction = df_transaction.fillna(0)
        self.inputs = self.inputs.merge(df_transaction, how='left', left_on = ['date','store_nbr'],right_on = ['date','store_nbr'])
        
        # pre-processing holiday data
        self.holiday_data.locale_name.value_counts().head()
        ny_dic = {'type': 'Holiday','locale':'National','locale_name':'Ecuador','description': 'New Year Day','transferred':'False'}
        ny_date = pd.to_datetime(['2012-01-01','2013-01-01','2014-01-01','2015-01-01','2016-01-01','2017-01-01','2018-01-01'])
        cm_dic = {'type': 'Holiday','locale':'National','locale_name':'Ecuador','description': 'Christmas Day','transferred':'False'}
        cm_date = pd.to_datetime(['2012-12-25','2013-12-25','2014-12-25','2015-12-25','2016-12-25','2017-12-25','2018-12-25'])
        
        for date in ny_date:
            self.holiday_data.loc[date] = ['Holiday','National', 'Ecuador', 'New Year day','False']
        for date in cm_date:
            self.holiday_data.loc[date] = ['Holiday','National', 'Ecuador', 'Christmas day','False']
        self.holiday_data = self.holiday_data.sort_index()
        
        calendar = pd.DataFrame(index = pd.date_range('2013-01-01','2017-08-31'))
        calendar = calendar.join(self.holiday_data).fillna(0)
        calendar['dow'] = calendar.index.dayofweek+1
        calendar['workday'] = True
        calendar.loc[calendar['dow']>5 , 'workday'] = False #make work_day false for sat and sun (6/7 in dow)
        calendar.loc[(calendar['type']=='Holiday') & (calendar['locale'].str.contains('National')), 'workday'] = False
        calendar.loc[(calendar['type']=='Additional') & (calendar['locale'].str.contains('National')), 'workday'] = False
        calendar.loc[(calendar['type']=='Bridge') & (calendar['locale'].str.contains('National')), 'workday'] = False
        calendar.loc[(calendar['type']=='Transfer') & (calendar['locale'].str.contains('National')), 'workday'] = False
        calendar.loc[calendar['type']=='Work Day' , 'workday'] = True
        calendar.where(calendar['transferred'] == True).dropna()
        calendar.loc[(calendar['transferred'] == True), 'workday'] = True
        calendar.where(calendar['description'].str.contains('futbol')).dropna()
        calendar['is_football'] = 0
        calendar['is_eq'] = 0
        calendar.loc[(calendar['is_football'] == 0) & (calendar['description'].str.contains('futbol')), 'is_football'] = 1
        calendar.loc[(calendar['is_eq'] == 0) & (calendar['description'].str.contains('Terremoto')), 'is_eq'] = 1
        calendar.loc[calendar['is_football']==1,'description'] = 'football'
        calendar.loc[calendar['is_eq']==1,'description'] = 'earthquake'
        calendar['workday'] = calendar['workday'].map({False:0,True:1})
        calendar['transferred'] = calendar['transferred'].map({'False':0,False:0,True:1})
        calendar['is_ny'] = 0
        calendar['is_christmas'] = 0
        calendar['is_shopping'] = 0
        calendar.loc[calendar['description'] == 'New Year day', 'is_ny'] = 1
        calendar.loc[calendar['description'] == 'Christmas day', 'is_christmas'] = 1
        calendar.loc[calendar['description'] == 'Black Friday', 'is_shopping'] = 1
        calendar.loc[calendar['description'] == 'Cyber Monday' , 'is_shopping'] = 1
        calendar = calendar.drop(['type','locale'], axis=1)
        locale_dummy = pd.get_dummies(calendar['locale_name'],prefix='holiday_')
        calendar = locale_dummy.join(calendar,how='left')
        calendar = calendar.drop('locale_name',axis=1)
        del locale_dummy
        
        calendar_checkpoint = calendar
        calendar_checkpoint = calendar_checkpoint.drop('description',axis = 1) #don't need description anymore we have dummied them all
        calendar_checkpoint = calendar_checkpoint[~calendar_checkpoint.index.duplicated(keep='first')] 
        calendar_checkpoint = calendar_checkpoint.iloc[:,1:-1]
        calendar_checkpoint.reset_index().tail()
        self.inputs = self.inputs.merge(calendar_checkpoint,how='left',left_on=['date'],right_on=calendar_checkpoint.index)
        
        pd.set_option('display.max_rows',None)
        pd.reset_option('display.max_rows','display.max_columns')
        self.inputs.dropna(inplace = True)
        self.inputs = self.inputs.set_index('date')
        
    def train_test_split(self):
        self.inputs = self.inputs.sort_index()
        self.y_train = self.inputs.loc['2013-01-01':'2017-08-15', 'sales']
        self.x_train = self.inputs.loc['2013-01-01':'2017-08-15'].drop(['sales','id'],axis=1)
        self.x_train = self.x_train.reset_index()
        self.x_train = self.x_train.set_index(['date','store_nbr','family'])
        
        self.x_test = self.inputs.loc['2017-08-16': ]
        self.test_id = self.x_test['id'] #Keep for later
        self.x_test.drop(['sales','id'],axis = 1,inplace = True)
        self.x_test = self.x_test.reset_index()
        self.x_test = self.x_test.set_index(['date','store_nbr','family'])
        
    def train_model(self):
        self.ln = linear_model.LinearRegression()
        self.ln.fit(self.x_train,self.y_train)
        
    def predict(self):
        self.y_pred = self.ln.predict(self.x_test)
        self.sample['sales'] = self.y_pred
        self.sample.to_csv('submission.csv', index = False) 
                    
        
preprocess = Preprocess()
preprocess.read_data()
preprocess.preprocess_data()
preprocess.train_test_split()
preprocess.train_model()
preprocess.predict()
gc.collect()
