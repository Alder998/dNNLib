"""Class to handle the model prediction according to the problem"""
import numpy as np
import pandas as pd
from VectorModule import VectorModule as v
from Dataset import Dataset as data

class ModelPrediction:

    def __init__(self, model):
        self.model = model
        pass

    # Function to create the steps-ahead dataFrame
    def createFutureDataFrame(self, dataInDataFrameFormat, date_column, frequency, lag_series=[], scope="prediction", scaler=None):

        if date_column == "index":
            date_idx = dataInDataFrameFormat.index
        else:
            date_idx = dataInDataFrameFormat[date_column]
        if scope == "prediction":
            future_dataframe = pd.DataFrame(pd.date_range(start=date_idx.max(), periods=self.model["time_window"], freq=frequency)).set_axis(["Date"], axis=1)
        elif scope == "multiple-prediction":
            # Re-create lags (the prediction data you are passing will not work otherwise)
            dataInDataFrameFormat = data.Dataset().processDatasetForTimeSeries(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                                date_column=date_column,
                                                                                target_column=self.model["var_to_predict"],
                                                                                date_column_format="%Y-%m-%d %H:%M:%S",
                                                                                frequency=frequency,
                                                                                lag_series=lag_series)
            future_dataframe = pd.DataFrame(pd.date_range(start=date_idx.max(), periods=self.model["time_window"], freq=frequency)).set_axis(["Date"], axis=1)
        elif scope == "confidence":
            future_dataframe = pd.DataFrame(pd.date_range(start=pd.Series(date_idx).head(self.model["time_window"])[self.model["time_window"]-1],
                                                          periods=self.model["time_window"],
                                                          freq=frequency)).set_axis(["Date"], axis=1)
        else:
            raise Exception("Scope: " + str(scope) + " not implemented!")

        # 1. Create the columns params
        if "year" in self.model["params"]:
            future_dataframe["year"] = future_dataframe["Date"].dt.year
        if "quarter" in self.model["params"]:
            future_dataframe["quarter"] = future_dataframe["Date"].dt.quarter
        if "month" in self.model["params"]:
            future_dataframe["month"] = future_dataframe["Date"].dt.month
        if "season" in self.model["params"]:
            future_dataframe.loc[future_dataframe["month"].isin([1, 2, 3]), "season"] = 1
            future_dataframe.loc[future_dataframe["month"].isin([4, 5, 6]), "season"] = 2
            future_dataframe.loc[future_dataframe["month"].isin([7, 8, 9]), "season"] = 3
            future_dataframe.loc[future_dataframe["month"].isin([10, 11, 12]), "season"] = 4
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

        # 4. Process dataFrame for lags
        if len(lag_series) != 0:
            for tc in self.model["var_to_predict"]:
                df_for_lags = dataInDataFrameFormat[[tc + "_" + str(lag_number) + "_lag" for lag_number in lag_series]]
                df_for_lags = df_for_lags[-self.model["time_window"]:].reset_index(drop=True)
                # 5. Add lags
                future_dataframe = pd.concat([future_dataframe, df_for_lags], axis=1)

        # 2.1. Standardize the features to align them with the model prediction
        if (self.model["feature_scaler"][0] if isinstance(self.model["feature_scaler"], list) else self.model["feature_scaler"]) is not None:
            future_dataframe[self.model["params"]] = scaler.fit_transform(future_dataframe[self.model["params"]])

        # 3. Process the future dataframe with the batch size
        input_data = np.array(future_dataframe[self.model["params"]])
        batch_size_LSTM = int(input_data.shape[0] / self.model["time_window"])
        fabt = []
        for i in range(batch_size_LSTM):
            fab = input_data[self.model["time_window"] * i: self.model["time_window"] * (i + 1), :]
            fabt.append(fab)
        input_data = np.stack(fabt, axis=0)

        return future_dataframe, input_data

    def generateConfidenceBars (self, dataInDataFrameFormat, frequency, date_column="index", target_division=1, lag_series=[]):

        print("INFO -- Generating confidence area...")
        future_dataframe, input_data = self.createFutureDataFrame(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                  date_column=date_column,
                                                                  frequency=frequency,
                                                                  lag_series=lag_series,
                                                                  scope="confidence",
                                                                  scaler=self.model["feature_scaler"])
        # 0. predict
        prediction = self.model["model"].predict(input_data)
        prediction_dataFrame = pd.DataFrame(np.squeeze(prediction, axis=0)).set_axis(self.model["var_to_predict"],axis=1).set_index(future_dataframe["Date"])

        # 0.1. Apply the inverse-transform if user chose standardization
        if self.model["target_scaler"] is not None:
            prediction_dataFrame[self.model["var_to_predict"]] = self.model["target_scaler"].inverse_transform(pd.DataFrame(prediction_dataFrame[self.model["var_to_predict"]]))

        # 1. Get the actual data from the dataset
        if date_column == "index":
            actual_data = dataInDataFrameFormat[dataInDataFrameFormat.index.isin(prediction_dataFrame.index)][self.model["var_to_predict"]]/target_division
        else:
            actual_data = dataInDataFrameFormat[dataInDataFrameFormat[date_column].isin(prediction_dataFrame.index)].set_index(date_column)[self.model["var_to_predict"]]/target_division
        # 2. Create the residuals (absolute values) - Apply the target division as well - iterate for each one of the target variable
        residuals = prediction_dataFrame - actual_data
        scores = np.abs(residuals.values)
        # 3. Generate the q areas (confidence) at 95%
        conficence_area = np.quantile(scores, 0.95, axis=0)

        # 3.1. Expected shape is going to be (var_to_predict,)
        return conficence_area

    def predictTimeSeriesWithTrainedModel (self, dataInDataFrameFormat, steps_ahead, frequency, date_column="index",
                                           confidence_area=True, target_division=1, lag_series=[]):

        # 0. Create the future DataFrame
        future_dataframe, input_data = self.createFutureDataFrame(dataInDataFrameFormat=dataInDataFrameFormat,
                                                                  date_column=date_column,
                                                                  frequency=frequency,
                                                                  lag_series=lag_series,
                                                                  scaler=self.model["feature_scaler"])
        # 0.1. Create Confidence bars
        if confidence_area:
            conficence_area = self.generateConfidenceBars(dataInDataFrameFormat=dataInDataFrameFormat,
                                                          date_column=date_column,
                                                          frequency=frequency,
                                                          target_division=target_division,
                                                          lag_series=lag_series)
        else:
            conficence_area = 0

        # 1. Enable the model to predict a given number of steps ahead
        if steps_ahead < self.model["time_window"]:
            # 2. Predict with stored data
            prediction = self.model["model"].predict(input_data)
            prediction_dataFrame = pd.DataFrame(np.squeeze(prediction, axis=0)).set_axis(self.model["var_to_predict"],axis=1).set_index(future_dataframe["Date"])
            # 2.0.1. Multiply for target divider, if required
            prediction_dataFrame = prediction_dataFrame * target_division
            # 2.1. De-standardize
            if self.model["target_scaler"] is not None:
                prediction_dataFrame[self.model["var_to_predict"]] = self.model["target_scaler"].inverse_transform(pd.DataFrame(prediction_dataFrame[self.model["var_to_predict"]]))
            prediction_dataFrame = prediction_dataFrame[0:steps_ahead]
        elif steps_ahead == self.model["time_window"]:
            # 2. Predict with stored data
            prediction = self.model["model"].predict(input_data)
            prediction_dataFrame = pd.DataFrame(np.squeeze(prediction, axis=0)).set_axis(self.model["var_to_predict"],axis=1).set_index(future_dataframe["Date"])
            # 2.0.1. Multiply for target divider, if required
            prediction_dataFrame = prediction_dataFrame * target_division
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
                prediction_dataFrame_chunk = pd.DataFrame(np.squeeze(prediction, axis=0)).set_axis(self.model["var_to_predict"], axis=1).set_index(future_dataframe["Date"])
                # 2.1.1. De-standardize
                if self.model["target_scaler"] is not None:
                    prediction_dataFrame_chunk[self.model["var_to_predict"]] = self.model["target_scaler"].inverse_transform(pd.DataFrame(prediction_dataFrame_chunk[self.model["var_to_predict"]]))
                # 2.2. Update Input data
                future_dataframe, input_data = self.createFutureDataFrame(dataInDataFrameFormat=future_dataframe, date_column="Date", frequency=frequency, scaler=self.model["feature_scaler"])
                # 2.3. Append to the full prediction DataFrame
                full_predictions.append(prediction_dataFrame_chunk)
            prediction_dataFrame = pd.concat([df for df in full_predictions], axis=0)
            # 2.3.1. Multiply for target divider, if required
            prediction_dataFrame = prediction_dataFrame * target_division
            # 2.4. Truncate for not full batch steps
            if steps_ahead % self.model["time_window"] != 0:
                prediction_dataFrame = prediction_dataFrame.iloc[:-(self.model["time_window"] - steps_remaining)]
        else:
            raise Exception("Error in computing prediction steps!")

        # 3. Add the confidence area
        upper_prediction = prediction_dataFrame + (conficence_area * target_division)
        lower_prediction = prediction_dataFrame - (conficence_area * target_division)

        return prediction_dataFrame, upper_prediction, lower_prediction

    # Utils function to prepare dataframe for geo-space prediction
    def prepareDataForGeoSpacePrediction (self, dataInDataFrameFormat, date_column, frequency, lag_series, target_division, scope="prediction"):

        # 0. For each coordinate, create future dataFrame
        unique_spaceVar = self.model["space_variables"].astype(str).agg('_'.join, axis=1) if len(self.model["space_variables"].columns) > 1 else self.model["space_variables"]
        timeSpaceDataFrame = []
        print("PREDICTION - Creating Geospatial Prediction Dataset...")
        for i, coord in enumerate(unique_spaceVar.unique()):
            future_dataframe, input_data = self.createFutureDataFrame(dataInDataFrameFormat=dataInDataFrameFormat, date_column=date_column,
                                                 frequency=frequency, lag_series=lag_series, scope=scope, scaler=self.model["feature_scaler"][i])
            future_dataframe["space_unique"] = coord
            timeSpaceDataFrame.append(future_dataframe)
        timeSpaceDataFrame = pd.concat([df for df in timeSpaceDataFrame], axis=0).reset_index(drop=True)

        # 1. Separate space columns
        for i, space_col_name in enumerate(self.model["space_variables"].columns):
            timeSpaceDataFrame[space_col_name] = timeSpaceDataFrame["space_unique"].str.split("_").str[i]
        timeSpaceDataFrame = timeSpaceDataFrame.drop(columns="space_unique")
        timeSpaceDataFrame[self.model["var_to_predict"]] = 0

        # Transform into array
        features_array, target_array, feature_scaler, target_scaler = v.VectorModule(modelStructure=self.model["modelStructure"]).processDataForGeospatialModel(
                                                                                     dataInDataFrameFormat=timeSpaceDataFrame,
                                                                                     feature_variables=self.model["params"],
                                                                                     target_variables=self.model["var_to_predict"],
                                                                                     test_size=None,
                                                                                     time_window=self.model["time_window"],
                                                                                     space_variables=self.model["space_variables"].columns,
                                                                                     standardize=False,
                                                                                     split_method="random",
                                                                                     seasonal_splits=0,
                                                                                     prediction=True,
                                                                                     target_division=target_division)

        return features_array, target_array, feature_scaler, target_scaler, unique_spaceVar, timeSpaceDataFrame

    # Utils function to predict geo-space returning DataFrame, saving precious 6 lines of code
    def predictGeoSpaceDataFrame (self, features_array, unique_spaceVar, timeSpaceDataFrame, target_division, steps_ahead):
        # Use stored model for prediction
        model_prediction = self.model["model"].predict(features_array.transpose(0, 1, 3, 2))
        # Adapt the model to handle multiple prediction for multiple variables
        prediction_dfs = []
        for i, variable_pred in enumerate(self.model["var_to_predict"]):

            # Create a DataFrame + populate Axis with time and space variables
            model_prediction_df = pd.DataFrame(np.squeeze(model_prediction[:, :, :, i], axis=0)).T

            model_prediction_df = model_prediction_df.set_index(pd.Series(unique_spaceVar.unique()))
            model_prediction_df = model_prediction_df.set_axis(pd.Series(timeSpaceDataFrame["Date"].unique()), axis=1)
            # Truncates in case of LESS time steps for prediction than time window
            if steps_ahead < self.model["time_window"]:
                model_prediction_df = model_prediction_df[model_prediction_df.columns[:steps_ahead]]

            # Create Latitude + Longitude columns + "pile" the coordinates on axis=0
            dataPiled = []
            for singleValue in model_prediction_df.index:
                dfc = model_prediction_df[model_prediction_df.index == singleValue].T
                dfc["singleCoord"] = singleValue
                dfc = dfc.reset_index()
                dfc = dfc.rename(columns={singleValue: variable_pred}).rename(columns={"index": "date"})
                dataPiled.append(dfc)
            dataPiled = pd.concat([df for df in dataPiled], axis=0).reset_index(drop=True)

            dataPiled["latitude"] = dataPiled["singleCoord"].str.split("_").str[0].astype(float)
            dataPiled["longitude"] = dataPiled["singleCoord"].str.split("_").str[1].astype(float)
            dataPiled = dataPiled.drop(columns=["singleCoord"])
            dataPiled = dataPiled.reset_index(drop=True)
            # Re-order
            dataPiled = dataPiled[["date", "latitude", "longitude", variable_pred]]
            # Multiply for target division param to preserve the variable scale
            dataPiled[variable_pred] = dataPiled[variable_pred] * target_division
            # Append
            prediction_dfs.append(dataPiled)

        # De-Standardize with the loaded scaler from Model Training
        df_prediction = []
        for df in prediction_dfs:
            df = df.set_index(["date", "latitude", "longitude"])
            df_prediction.append(df)
        df_prediction = pd.concat([dt for dt in df_prediction], axis=1)
        df_prediction = df_prediction.reset_index()
        # De-Standardize (for de-standardization, a shape of (time steps, variables) is needed
        df_prediction["unique_spaceVar"] = df_prediction["latitude"].astype(str) + "_" + df_prediction["longitude"].astype(str)

        if (self.model["target_scaler"][0] if isinstance(self.model["target_scaler"], list) else self.model["target_scaler"]) is not None:
            for k, coord_scaler in enumerate(self.model["target_scaler"]):
                mask = df_prediction["unique_spaceVar"] == unique_spaceVar[k]
                df_prediction.loc[mask, self.model["var_to_predict"]] = (coord_scaler.inverse_transform(df_prediction.loc[mask, self.model["var_to_predict"]]))
        df_prediction = df_prediction.drop(columns=["unique_spaceVar"])

        return df_prediction

    # Function to predict with geospatial model
    def predictGeoSpatialWithTrainedModel (self, dataInDataFrameFormat, steps_ahead, frequency, date_column="index", target_division=1, lag_series=[]):

        # Prepare data for prediction
        features_array, target_array, feature_scaler, target_scaler, unique_spaceVar, timeSpaceDataFrame = self.prepareDataForGeoSpacePrediction(
                                                                                                            dataInDataFrameFormat=dataInDataFrameFormat,
                                                                                                            date_column=date_column,
                                                                                                            frequency=frequency,
                                                                                                            lag_series=lag_series,
                                                                                                            target_division=target_division)

        # Case for time steps == or < time_window
        if (steps_ahead == self.model["time_window"]) | (steps_ahead < self.model["time_window"]):
            model_prediction_df = self.predictGeoSpaceDataFrame(features_array=features_array,
                                                                unique_spaceVar=unique_spaceVar,
                                                                timeSpaceDataFrame=timeSpaceDataFrame,
                                                                target_division=target_division,
                                                                steps_ahead=steps_ahead)
            return model_prediction_df
        # Most difficult case: when prediction steps ahead > time_window
        elif steps_ahead > self.model["time_window"]:
            # define how many times one-batch prediction the model has to perform
            prediction_chunks = (int(steps_ahead / self.model["time_window"])) if steps_ahead % self.model["time_window"] == 0 else (int(steps_ahead / self.model["time_window"]) + 1) # If 1.1 -> 1 + 1 = 2 chuncks
            # Perform the first prediction
            aggregated_model_prediction = []
            model_prediction_df = self.predictGeoSpaceDataFrame(features_array=features_array,
                                                                unique_spaceVar=unique_spaceVar,
                                                                timeSpaceDataFrame=timeSpaceDataFrame,
                                                                target_division=target_division,
                                                                steps_ahead=steps_ahead)
            aggregated_model_prediction.append(model_prediction_df)
            # Then, perform the calculation again upon the prediction dataFrame
            for ch in range(prediction_chunks-1):
                marginal_steps_ahead = np.abs((self.model["time_window"] * (ch+1)) - steps_ahead)
                features_array, target_array, feature_scaler, target_scaler, unique_spaceVar, timeSpaceDataFrame = self.prepareDataForGeoSpacePrediction(
                                                                                                                    dataInDataFrameFormat=model_prediction_df,
                                                                                                                    date_column=date_column,
                                                                                                                    frequency=frequency,
                                                                                                                    lag_series=lag_series,
                                                                                                                    target_division=target_division,
                                                                                                                    scope="multiple-prediction")
                model_prediction_df = self.predictGeoSpaceDataFrame(features_array=features_array, unique_spaceVar=unique_spaceVar,
                                                                    timeSpaceDataFrame=timeSpaceDataFrame, target_division=target_division,
                                                                    steps_ahead=marginal_steps_ahead)
                aggregated_model_prediction.append(model_prediction_df)
            aggregated_model_prediction = pd.concat([df for df in aggregated_model_prediction], axis=0).reset_index(drop=True)
            return aggregated_model_prediction
        else:
            raise Exception("Prediction steps and time window are not compatible! Time window: " + str(self.model["time_window"]) + " - Steps ahead: " + str(steps_ahead))