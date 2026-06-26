"""Tester for Time Series Wrapper"""

from Wrappers import TimeSeriesWrapper as ts
import pandas as pd

# 0. Data ( "Solar_Actual Aggregated", "Wind Onshore_Actual Aggregated", "Hydro Run-of-river and poundage_Actual Aggregated")
data = pd.read_excel("C:\\Users\\alder\\Downloads\\time_series_weather_Milano.xlsx")

# 1. Model
ts.TimeSeriesWrapper(modelStructure={"MultiSeasonConv1DGated": {"layers": [16], "cycles": [960],
                                                                "use_layer_norm": True, "baseline_kernel": None,
                                                                "use_cross_cycle_attention": True, "use_seasonal_memory": False},
                                      "LSTM": {"layers": [128, 64], "activation": "tanh", "dropout": 0.1},
                                      "FF": {"layers": [200, 200], "activation": "relu", "dropout": 0.1}},
                     feature_variables=["year","month","day","hour"],
                     time_window=48,
                     target_variables=["temperature", "windSpeed"],
                     date_column="date",
                     frequency="1h",
                     lags=[]
                     ).trainPredictAndSaveTimeSeriesModel(data=data,
                                                          date_column_format="%Y-%m-%dT%H:%M",
                                                          loss="MSE",  # "MSE" | "peak-aware-MSE"
                                                          split_method="time-series",  # "time-series" | "seasonal-time-series" | "random"
                                                          seasonal_splits=30,
                                                          epochs=3,
                                                          prediction_steps_ahead=48,
                                                          target_division=1,
                                                          scaler="std",  # "std" | "min-max"
                                                          standardize=True,
                                                          plot=True,
                                                          plot_save_dir=r"C:\Users\alder\Downloads\temp_prediction_1h.png")