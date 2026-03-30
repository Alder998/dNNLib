"""Tester for Time Series Wrapper"""

from Wrappers import TimeSeriesWrapper as ts
import pandas as pd

# 0. Data
data = pd.read_csv("C:\\Users\\alder\\Downloads\\energy_market_data_italia_1p.xlsx")
data["Date"] = pd.to_datetime(data["Date"])
data["Date"] = data["Date"].dt.tz_localize(None)
data = data[(data["zone"]=="Italia Nord") & (data["lat"]==data["lat"][0]) & (data["lon"]==data["lon"][0])].sort_values(by="Date", ascending=True).reset_index(drop=True)
data = data.drop(columns = ["Unnamed: 0", "zone", "short_code", "code", "map_code", "lat", "lon"])

#data = pd.read_csv(r"C:\Users\alder\Downloads\1mo_weather.csv")
#data = data[["date","year","month","day","hour","temperature"]][(data["latitude"] == data["latitude"].unique()[0]) & (data["longitude"] == data["longitude"].unique()[0])].reset_index(drop=True)
#data["date"] = pd.to_datetime(data["date"])

# 1. Model
ts.TimeSeriesWrapper(modelStructure={"MultiSeasonConv1DGated": {"layers": [16], "cycles": [672],
                                                                "use_layer_norm": True, "baseline_kernel": None,
                                                                "use_cross_cycle_attention": False, "use_seasonal_memory": False},
                                      "LSTM": {"layers": [128, 64], "activation": "tanh", "dropout": 0.1},
                                      "FF": {"layers": [200, 200], "activation": "relu", "dropout": 0.1}},
                     feature_variables=["month","day","day_of_week","hour","minute"],
                     time_window=192,
                     target_variables="Actual Load",
                     date_column="Date",
                     frequency="15min").trainPredictAndSaveTimeSeriesModel(data=data,
                                                                        loss="MSE",  # "MSE" | "peak-aware-MSE"
                                                                        split_method="time-series",  # "time-series" | "seasonal-time-series" | "random"
                                                                        seasonal_splits=12,
                                                                        epochs=200,
                                                                        prediction_steps_ahead=192,
                                                                        plot=True,
                                                                        plot_save_dir=r"C:\Users\alder\Downloads\temp_prediction_1h.png",
                                                                        target_division=10000)