"""Tester for Time Series Wrapper"""

from Wrappers import TimeSeriesWrapper as ts
import pandas as pd

# 0. Data ( "Solar_Actual Aggregated", "Wind Onshore_Actual Aggregated", "Hydro Run-of-river and poundage_Actual Aggregated")
data = pd.read_excel("C:\\Users\\alder\\Downloads\\time_series_weather2.xlsx")

# 1. Model
ts.TimeSeriesWrapper(modelStructure={"MultiSeasonConv1DGated": {"layers": [16], "cycles": [48],
                                                                "use_layer_norm": True, "baseline_kernel": None,
                                                                "use_cross_cycle_attention": True, "use_seasonal_memory": False},
                                      "LSTM": {"layers": [128, 64], "activation": "tanh", "dropout": 0.0},
                                      "FF": {"layers": [200, 200], "activation": "relu", "dropout": 0.0}},
                     feature_variables=["year","month","day","day_of_week","hour","minute"],
                     time_window=48,
                     target_variables=["temperature"],
                     date_column="date",
                     frequency="1h",
                     lags=[]
                     ).trainPredictAndSaveTimeSeriesModel(data=data,
                                                          date_column_format="%Y-%m-%dT%H:%M",
                                                          loss="MSE",  # "MSE" | "peak-aware-MSE"
                                                          split_method="seasonal-time-series",  # "time-series" | "seasonal-time-series" | "random"
                                                          seasonal_splits=36,
                                                          epochs=50,
                                                          prediction_steps_ahead=48,
                                                          target_division=1,
                                                          plot=True,
                                                          plot_save_dir=r"C:\Users\alder\Downloads\temp_prediction_1h.png")