"""Tester for Time Series Wrapper"""

from Wrappers import TimeSeriesWrapper as ts
import pandas as pd

# 0. Data
#data = pd.read_excel("C:\\Users\\alder\\Downloads\\aggregate_load_by_bidding_zone.xlsx")
#data = data[data["Bidding_zone"]=="IT-NORD"].sort_values(by="Date", ascending=True).reset_index(drop=True)

data = pd.read_csv(r"C:\Users\alder\Downloads\1mo_weather.csv")
data = data[["date","year","month","day","hour","temperature"]][(data["latitude"] == data["latitude"].unique()[0]) & (data["longitude"] == data["longitude"].unique()[0])].reset_index(drop=True)
data["date"] = pd.to_datetime(data["date"])

# 1. Model
ts.TimeSeriesWrapper(modelStructure={"MultiSeasonConv1DGated": {"layers": [16, 16, 16, 16, 16, 16], "cycles": [6, 12, 24, 48, 240, 480],
                                                                "use_layer_norm": True, "baseline_kernel": None,
                                                                "use_cross_cycle_attention": False, "use_seasonal_memory": False},
                                      "LSTM": {"layers": [128, 64], "activation": "tanh", "dropout": 0.0},
                                      "FF": {"layers": [200, 200], "activation": "relu", "dropout": 0.0}},
                     feature_variables=["month","day","hour"],
                     time_window=48,
                     target_variables="temperature",
                     date_column="date",
                     frequency="1h").trainPredictAndSaveTimeSeriesModel(data=data,
                                                                        loss="MSE",  # "MSE" | "peak-aware-MSE"
                                                                        split_method="time-series",  # "time-series" | "seasonal-time-series" | "random"
                                                                        seasonal_splits=12,
                                                                        epochs=200,
                                                                        prediction_steps_ahead=48,
                                                                        plot=True,
                                                                        plot_save_dir=r"C:\Users\alder\Downloads\temp_prediction_1h.png")