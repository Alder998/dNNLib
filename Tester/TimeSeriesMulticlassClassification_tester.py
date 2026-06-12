"""Easy tester for a Multi-Class Classification Time-Series tester"""

from Wrappers import TimeSeriesMClassificationWrapper as tc
import pandas as pd

# 0. Data (Weather Forecast)
data = pd.read_excel("C:\\Users\\alder\\Downloads\\time_series_weather.xlsx")

# 1. Model
tc.TimeSeriesMClassificationWrapper(modelStructure={"MultiSeasonConv1DGated": {"layers": [16], "cycles": [192],
                                                    "use_layer_norm": True, "baseline_kernel": None,
                                                    "use_cross_cycle_attention": True, "use_seasonal_memory": False},
                                      "LSTM": {"layers": [128, 64], "activation": "tanh", "dropout": 0.0},
                                      "FF": {"layers": [200, 200], "activation": "relu", "dropout": 0.0}},
                     feature_variables=["day","day_of_week","hour","minute"],
                     time_window=192,
                     target_variables=["precipitation_binary"],
                     date_column="date",
                     frequency="1h",
                     lags=[]
                     ).trainPredictAndSaveTimeSeriesModel(data=data,
                                                          date_column_format="%Y-%m-%d %H:%M:%S",
                                                          loss="MSE",  # "MSE" | "peak-aware-MSE"
                                                          split_method="time-series",  # "time-series" | "seasonal-time-series" | "random"
                                                          seasonal_splits=6,
                                                          epochs=50,
                                                          target_division=100)