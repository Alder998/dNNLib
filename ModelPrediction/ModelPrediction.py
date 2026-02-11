"""Class to handle the model prediction according to the problem"""
import numpy as np
import pandas as pd
from VectorModule import VectorModule as v

class ModelPrediction:

    def __init__(self, model):
        self.model = model
        pass

    # Function to create the steps-ahead dataFrame
    def createFutureDataFrame(self, dataInDataFrameFormat, date_column, frequency):

        if date_column == "index":
            date_idx = dataInDataFrameFormat.index
        else:
            date_idx = dataInDataFrameFormat[date_column]
        future_dataframe = pd.DataFrame(pd.date_range(start=date_idx.max(), periods=self.model["time_window"], freq=frequency)).set_axis(["Date"], axis=1)

        # 1. Create the columns params
        if "year" in self.model["params"]:
            future_dataframe["year"] = future_dataframe["Date"].dt.year
        if "month" in self.model["params"]:
            future_dataframe["month"] = future_dataframe["Date"].dt.month
        if "day" in self.model["params"]:
            future_dataframe["day"] = future_dataframe["Date"].dt.day
        if "day_of_week" in self.model["params"]:
            future_dataframe["day_of_week"] = future_dataframe["Date"].dt.dayofweek
        if "hour" in self.model["params"]:
            future_dataframe["hour"] = future_dataframe["Date"].dt.hour
        if "minute" in self.model["params"]:
            future_dataframe["minute"] = future_dataframe["Date"].dt.minute

        # 2. Drop Duplicated columns to allow re-creation
        future_dataframe = future_dataframe.loc[:, ~future_dataframe.columns.duplicated()]

        # 2.1. Standardize the features
        if (self.model["feature_scaler"][0] if isinstance(self.model["feature_scaler"], list) else self.model["feature_scaler"]) is not None:
            future_dataframe[self.model["params"]] = self.model["feature_scaler"].fit_transform(future_dataframe[self.model["params"]])

        # 3. Process the future dataframe with the batch size
        input_data = np.array(future_dataframe[self.model["params"]])
        batch_size_LSTM = int(input_data.shape[0] / self.model["time_window"])
        fabt = []
        for i in range(batch_size_LSTM):
            fab = input_data[self.model["time_window"] * i: self.model["time_window"] * (i + 1), :]
            fabt.append(fab)
        input_data = np.stack(fabt, axis=0)

        return future_dataframe, input_data

    def predictTimeSeriesWithTrainedModel (self, dataInDataFrameFormat, steps_ahead, frequency, date_column="index"):

        # 0. Create the future DataFrame
        future_dataframe, input_data = self.createFutureDataFrame(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                  date_column=date_column,
                                                                  frequency=frequency)
        # 1. Enable the model to predict a given number of steps ahead
        if steps_ahead < self.model["time_window"]:
            # 2. Predict with stored data
            prediction = self.model["model"].predict(input_data)
            prediction_dataFrame = pd.DataFrame(np.squeeze(prediction, axis=0)).set_axis([self.model["var_to_predict"]],axis=1).set_index(future_dataframe["Date"])
            # 2.1. De-standardize
            if self.model["target_scaler"] is not None:
                prediction_dataFrame[self.model["var_to_predict"]] = self.model["target_scaler"].inverse_transform(pd.DataFrame(prediction_dataFrame[self.model["var_to_predict"]]))
            prediction_dataFrame = prediction_dataFrame[0:steps_ahead]
        elif steps_ahead == self.model["time_window"]:
            # 2. Predict with stored data
            prediction = self.model["model"].predict(input_data)
            prediction_dataFrame = pd.DataFrame(np.squeeze(prediction, axis=0)).set_axis([self.model["var_to_predict"]],axis=1).set_index(future_dataframe["Date"])
            # 2.1. De-standardize
            if self.model["target_scaler"] is not None:
                prediction_dataFrame[self.model["var_to_predict"]] = self.model["target_scaler"].inverse_transform(pd.DataFrame(prediction_dataFrame[self.model["var_to_predict"]]))
        elif steps_ahead > self.model["time_window"]:
            full_chuncks = int(steps_ahead / self.model["time_window"]) if steps_ahead % self.model["time_window"] == 0 else int(steps_ahead / self.model["time_window"]) + 1
            steps_remaining = steps_ahead % self.model["time_window"]
            full_predictions = []
            for chunk in range(full_chuncks):
                prediction = self.model["model"].predict(input_data)
                # 2.1. Transform to dataFrame
                prediction_dataFrame_chunk = pd.DataFrame(np.squeeze(prediction, axis=0)).set_axis([self.model["var_to_predict"]], axis=1).set_index(future_dataframe["Date"])
                # 2.1.1. De-standardize
                if self.model["target_scaler"] is not None:
                    prediction_dataFrame_chunk[self.model["var_to_predict"]] = self.model["target_scaler"].inverse_transform(pd.DataFrame(prediction_dataFrame_chunk[self.model["var_to_predict"]]))
                # 2.2. Update Input data
                future_dataframe, input_data = self.createFutureDataFrame(dataInDataFrameFormat=future_dataframe, date_column="Date", frequency=frequency)
                # 2.3. Append to the full prediction DataFrame
                full_predictions.append(prediction_dataFrame_chunk)
            prediction_dataFrame = pd.concat([df for df in full_predictions], axis=0)
            # 2.4. Truncate for not full batch steps
            if steps_ahead % self.model["time_window"] != 0:
                prediction_dataFrame = prediction_dataFrame.iloc[:-(self.model["time_window"] - steps_remaining)]
        else:
            raise Exception("Error in computing prediction steps!")

        return prediction_dataFrame

    # Function to predict with geospatial model
    def predictGeoSpatialWithTrainedModel (self, dataInDataFrameFormat, steps_ahead, frequency, date_column="index"):

        # 0. For each coordinate, create future dataFrame
        unique_spaceVar = self.model["space_variables"].astype(str).agg('_'.join, axis=1) if len(self.model["space_variables"].columns) > 1 else self.model["space_variables"]
        timeSpaceDataFrame = []
        print("PREDICTION - Creating Geospatial Prediction Dataset...")
        for coord in unique_spaceVar.unique():
            future_dataframe, input_data = self.createFutureDataFrame(dataInDataFrameFormat=dataInDataFrameFormat, date_column=date_column,
                                                 frequency=frequency)
            future_dataframe["space_unique"] = coord
            timeSpaceDataFrame.append(future_dataframe)
        timeSpaceDataFrame = pd.concat([df for df in timeSpaceDataFrame], axis=0).reset_index(drop=True)

        # 1. Separate space columns
        for i, space_col_name in enumerate(self.model["space_variables"].columns):
            timeSpaceDataFrame[space_col_name] = timeSpaceDataFrame["space_unique"].str.split("_").str[i]
        timeSpaceDataFrame = timeSpaceDataFrame.drop(columns="space_unique")
        timeSpaceDataFrame[self.model["var_to_predict"]] = 0

        # Transform into array
        features_array, target_array, feature_scaler, target_scaler = v.VectorModule(modelStructure=self.model["modelStructure"]).processDataForGeospatialModel(dataInDataFrameFormat=timeSpaceDataFrame,
                                                                                     feature_variables=self.model["params"],
                                                                                     target_variables=self.model["var_to_predict"],
                                                                                     test_size=None,
                                                                                     time_window=self.model["time_window"],
                                                                                     space_variables=self.model["space_variables"].columns,
                                                                                     standardize=False,
                                                                                     split_method="random",
                                                                                     seasonal_splits=0,
                                                                                     prediction=True)
        model_prediction = self.model["model"].predict(features_array.transpose(0, 1, 3, 2))

        print("INFO - Model Prediction Shape: ", model_prediction.shape)

        return model_prediction