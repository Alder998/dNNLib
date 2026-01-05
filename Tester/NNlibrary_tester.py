"""Simple tester"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ModelArch import ModelArch as arch
from ModelTraining import ModelTraining as train
from ModelEvaluation import ModelEvaluation as eval

# 0.0. get some data (2025 15 min-power production in Italy)
dataset = pd.read_excel(r"C:\\Users\\alder\\Downloads\\Export-DownloadCenterFile-20260103-165403.xlsx")
time_window = 96
var_to_predict = "Photovoltaic"  # 'Wind', 'Geothermal', 'Hydro', 'Photovoltaic', 'Biomass', 'Thermal', 'Self-consumption'
# 0.1. Clean data
ren_prod_italy = []
for category in dataset["Primary Source"].unique():
    dfc = dataset[["Date","Actual Generation"]][dataset["Primary Source"] == category].reset_index(drop=True)
    dfc = dfc.rename(columns={"Actual Generation":category})
    ren_prod_italy.append(dfc.set_index("Date"))
ren_prod_italy = pd.concat([df for df in ren_prod_italy], axis=1)
ren_prod_italy["year"] = ren_prod_italy.index.year
ren_prod_italy["month"] = ren_prod_italy.index.month
ren_prod_italy["day"] = ren_prod_italy.index.day
ren_prod_italy["hour"] = ren_prod_italy.index.hour
ren_prod_italy["minute"] = ren_prod_italy.index.minute

# 0. Build the model
model = arch.ModelArch(modelStructure={"LSTM": [64, 64, 64, 64], "FF": [500, 500]}).createRegressionModelArchitecture(dropout_FF=0.2)

# 1. Compile and train
trained_model = train.ModelTraining(model=model).trainModel(dataInDataFrameFormat=ren_prod_italy,
                                                            feature_variables=["year", "month", "day", "hour", "minute"],
                                                            target_variables=var_to_predict,
                                                            split_method="time-series",
                                                            time_window=time_window,
                                                            test_size=0.30,
                                                            batch_size=32, validation_split=0.2, epochs=30)

# 2. Evaluate the model
evaluation = eval.ModelEvaluation(model=trained_model).evaluateModelPerformance()

# 3. Predict
future_dataframe = pd.DataFrame(pd.date_range(start=ren_prod_italy.index.max(), periods=time_window, freq="15min")).set_axis(["Date"], axis=1)
future_dataframe["year"] = future_dataframe["Date"].dt.year
future_dataframe["month"] = future_dataframe["Date"].dt.month
future_dataframe["day"] = future_dataframe["Date"].dt.day
future_dataframe["hour"] = future_dataframe["Date"].dt.hour
future_dataframe["minute"] = future_dataframe["Date"].dt.minute

input_data = np.array(future_dataframe[["year", "month", "day", "hour", "minute"]])
batch_size_LSTM = int(input_data.shape[0] / time_window)
fabt = []
for i in range(batch_size_LSTM):
    fab = input_data[time_window * i: time_window * (i + 1), :]
    fabt.append(fab)
input_data = np.stack(fabt, axis=0)

prediction = trained_model["model"].predict(input_data)
prediction = pd.DataFrame(np.squeeze(prediction, axis=0)).set_axis([var_to_predict], axis=1).set_index(future_dataframe["Date"])

# 4. Plot the prediction
plt.figure(figsize = (15, 5))
plt.plot(ren_prod_italy.sort_values(by="Date", ascending=True)[var_to_predict][-480:])
plt.plot(prediction[var_to_predict], color="red", linestyle="dashed")
plt.show()