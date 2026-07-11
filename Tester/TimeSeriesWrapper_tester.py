"""Tester for Time Series Wrapper"""

from Wrappers import TimeSeriesWrapper as ts
import pandas as pd

# 0. Data ( "Solar_Actual Aggregated", "Wind Onshore_Actual Aggregated", "Hydro Run-of-river and poundage_Actual Aggregated")
data = pd.read_excel("C:\\Users\\alder\\Downloads\\time_series_weather_Milano.xlsx")

# 1. Model
ts.TimeSeriesWrapper(modelStructure={"MultiSeasonConv1DGated": {"layers": [16], "cycles": [48],
                                                                "use_layer_norm": True, "baseline_kernel": None,
                                                                "use_cross_cycle_attention": True, "use_seasonal_memory": False},
                                      "LSTM": {"layers": [128, 64], "activation": "tanh", "dropout": 0.1},
                                      "FF": {"layers": [200, 200], "activation": "relu", "dropout": 0.1}},
                     feature_variables=["year","season","month","day","hour"],
                     time_window=48,
                     target_variables=["temperature"],
                     date_column="date",
                     frequency="1h",
                     lags=[]
                     ).trainPredictAndSaveTimeSeriesModel(data=data,
                                                          date_column_format="%Y-%m-%dT%H:%M",
                                                          loss="MSE",  # "MSE" | "peak-aware-MSE"
                                                          split_method="time-series",  # "time-series" | "seasonal-time-series" | "random"
                                                          shuffle=False,
                                                          seasonal_splits=300, # splitting chunks on time-series
                                                          standardize=True, # Scale the data
                                                          scaler="std",  # if standardize = True - "std" | "min-max"
                                                          epochs=50,
                                                          prediction_steps_ahead=48,
                                                          save_dir="D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models",
                                                          model_save_name="ts-1",
                                                          target_division=1, # whether to divide the target variable
                                                          plot=True, # whether to plot the prediction ahead
                                                          plot_save_dir=r"C:\Users\alder\Downloads\temp_prediction_1h.png")