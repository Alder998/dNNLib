"""Class to handle the model prediction according to the problem"""
import numpy as np
import pandas as pd

class ModelPrediction:

    def __init__(self, model):
        self.model = model
        pass

    def predictTimeSeriesWithTrainedModel (self, dataInDataFrameFormat, steps_ahead, frequency, date_column="index"):

        # 0. Create the future DataFrame
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
        if "hour" in self.model["params"]:
            future_dataframe["hour"] = future_dataframe["Date"].dt.hour
        if "minute" in self.model["params"]:
            future_dataframe["minute"] = future_dataframe["Date"].dt.minute

        # 3. Process the future dataframe with the batch size
        input_data = np.array(future_dataframe[self.model["params"]])
        batch_size_LSTM = int(input_data.shape[0] / self.model["time_window"])
        fabt = []
        for i in range(batch_size_LSTM):
            fab = input_data[self.model["time_window"] * i: self.model["time_window"] * (i + 1), :]
            fabt.append(fab)
        input_data = np.stack(fabt, axis=0)

        # 3.1. Enable the model to predict a given number of steps ahead
        if steps_ahead < self.model["time_window"]:
            # 4. Predict with stored data
            prediction = self.model["model"].predict(input_data)
            prediction_dataFrame = pd.DataFrame(np.squeeze(prediction, axis=0)).set_axis([self.model["var_to_predict"]],axis=1).set_index(future_dataframe["Date"])
            prediction_dataFrame = prediction_dataFrame[0:steps_ahead]
        elif steps_ahead == self.model["time_window"]:
            # 4. Predict with stored data
            prediction = self.model["model"].predict(input_data)
            prediction_dataFrame = pd.DataFrame(np.squeeze(prediction, axis=0)).set_axis([self.model["var_to_predict"]],axis=1).set_index(future_dataframe["Date"])

        return prediction_dataFrame