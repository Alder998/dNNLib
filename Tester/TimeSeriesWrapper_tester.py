"""Tester for Time Series Wrapper"""

from Wrappers import TimeSeriesWrapper as ts
import pandas as pd

# 0. Data
zone = "Italia Sicilia"
data = pd.read_csv("C:\\Users\\alder\\Downloads\\energy_market_data_italia_1p.csv")
data["Date"] = pd.to_datetime(data["Date"])
data["Date"] = data["Date"].dt.tz_localize(None)
data = data[(data["zone"]==zone) & (data["lat"]==data[data["zone"] == zone]["lat"].reset_index(drop=True)[0]) &
            (data["lon"]==data[data["zone"] == zone]["lon"].reset_index(drop=True)[0])].sort_values(by="Date", ascending=True).reset_index(drop=True)
data = data.drop(columns = ["Unnamed: 0", "zone", "short_code", "code", "map_code", "lat", "lon"])

# 1. Model
ts.TimeSeriesWrapper(modelStructure={"MultiSeasonConv1DGated": {"layers": [16, 16, 16], "cycles": [16, 48, 96],
                                                                "use_layer_norm": True, "baseline_kernel": None,
                                                                "use_cross_cycle_attention": False, "use_seasonal_memory": False},
                                      "LSTM": {"layers": [128, 64], "activation": "tanh", "dropout": 0.0},
                                      "FF": {"layers": [200, 200], "activation": "relu", "dropout": 0.0}},
                     feature_variables=["month","day","day_of_week","hour","minute"],
                     time_window=96,
                     target_variables=["Actual Load", "DAM Price"],
                     date_column="Date",
                     frequency="15min",
                     lags=[96, 192]
                     ).trainPredictAndSaveTimeSeriesModel(data=data,
                                                          date_column_format="%Y-%m-%d %H:%M:%S",
                                                          loss="MSE",  # "MSE" | "peak-aware-MSE"
                                                          split_method="time-series",  # "time-series" | "seasonal-time-series" | "random"
                                                          seasonal_splits=12,
                                                          epochs=10,
                                                          prediction_steps_ahead=96,
                                                          plot=False,
                                                          plot_save_dir=r"C:\Users\alder\Downloads\temp_prediction_1h.png",
                                                          target_division=1000)