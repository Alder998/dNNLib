"""Class to Process the datasets for time series and/or other scopes"""

import pandas as pd

class Dataset:

    def __init__(self):
        pass

    # Process Dataset for Time Series Models
    def processDatasetForTimeSeries (self, dataInDataFrameFormat, date_column, target_column, lag_series,
                                     date_column_format="%Y-%m-%d", frequency="1h", classification=False):

        # 0.0. Check the compliance of the time-frequency
        if frequency not in ["1mo", "1d", "1h", "15min", "1min"]:
            raise Exception("Frequency not implemented! Available: '1mo', '1d', '15min', '1min'.")
        # 0.1. Convert to date column to datetime
        dataInDataFrameFormat[date_column] = pd.to_datetime(dataInDataFrameFormat[date_column], format=date_column_format)

        # 0.2 First, add the time params from the date column ("year","quarter","month","day","day_of_week","hour","minute")
        if "year" not in dataInDataFrameFormat.columns:
            dataInDataFrameFormat["year"] = dataInDataFrameFormat[date_column].dt.year
        if "quarter" not in dataInDataFrameFormat.columns:
            dataInDataFrameFormat["quarter"] = dataInDataFrameFormat[date_column].dt.quarter
        if "month" not in dataInDataFrameFormat.columns:
            dataInDataFrameFormat["month"] = dataInDataFrameFormat[date_column].dt.month
        if "day" not in dataInDataFrameFormat.columns:
            dataInDataFrameFormat["day"] = dataInDataFrameFormat[date_column].dt.day
        if "day_of_week" not in dataInDataFrameFormat.columns:
            dataInDataFrameFormat["day_of_week"] = dataInDataFrameFormat[date_column].dt.day_of_week
        if "hour" not in dataInDataFrameFormat.columns:
            dataInDataFrameFormat["hour"] = dataInDataFrameFormat[date_column].dt.hour
        if "minute" not in dataInDataFrameFormat.columns:
            dataInDataFrameFormat["minute"] = dataInDataFrameFormat[date_column].dt.minute

        # 0.1. Sort by Date column
        dataInDataFrameFormat = dataInDataFrameFormat.sort_values(by=date_column, ascending=True).reset_index(drop=True)

        # 1. Add lags to the target variable, according specified by user
        if len(lag_series) != 0:
            for lag_number in lag_series:
                for tc in target_column:
                    dataInDataFrameFormat[tc + "_" + str(lag_number) + "_lag"] = dataInDataFrameFormat[tc].shift(lag_number)

        # 2. Make the target variable categorical (only for classification problems)
        if classification:
            dataInDataFrameFormat, categories_map= self.processDatasetForClassification(dataInDataFrameFormat=dataInDataFrameFormat, target_column=target_column)
            return dataInDataFrameFormat.dropna().reset_index(drop=True), categories_map

        else:
            return dataInDataFrameFormat.dropna().reset_index(drop=True)

    # Process Dataset for Classification Task
    def processDatasetForClassification (self, dataInDataFrameFormat, target_column):

        print("DATASET - Processing Dataset for Multiclass Classification Tasks...")
        # 0. The only stuff to do here is to make the target variable(s) categorical
        # 0.1. For each one of the target columns, create the dict to map categorical variable into classes
        categories_map = {}
        for ccol in target_column:
            categories_map[ccol] = {}
            categories = dataInDataFrameFormat[ccol].unique()
            for i, cat in enumerate(categories):
                # 0.2. Create the categories mapping
                categories_map[ccol][cat] = i
                # 0.3. Now, transform the categories into numbers
                dataInDataFrameFormat.loc[dataInDataFrameFormat[ccol] == cat, ccol] = i
            # 0.4. Force integer (otherwise it will give error "unrecognized type: object")
            dataInDataFrameFormat[ccol] = dataInDataFrameFormat[ccol].astype(int)

        return dataInDataFrameFormat, categories_map