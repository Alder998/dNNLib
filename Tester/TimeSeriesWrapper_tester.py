"""Tester for Time Series Wrapper"""

from Wrappers import TimeSeriesWrapper as ts
import pandas as pd

# 0. Data
data = pd.read_excel("C:\\Users\\alder\\Downloads\\aggregate_load_by_bidding_zone.xlsx")
data = data[data["Bidding_zone"]=="IT-NORD"].sort_values(by="Date", ascending=True).reset_index(drop=True)

# 1. Model
ts.TimeSeriesWrapper(modelStructure={"MultiSeasonConv1DGated": {"layers": [16, 16, 16, 16, 16], "cycles": [96, 192, 288, 450, 672],
                                                                "use_layer_norm": True, "baseline_kernel": None,
                                                                "use_cross_cycle_attention": False, "use_seasonal_memory": False},
                                      "LSTM": {"layers": [128, 64], "activation": "tanh", "dropout": 0.0},
                                      "FF": {"layers": [200, 200], "activation": "relu"}},
                     feature_variables=["year","month","day","day_of_week","hour","minute"],
                     time_window=96,
                     target_variables="Actual Load",
                     date_column="Date",
                     frequency="15min").trainPredictAndSaveTimeSeriesModel(data=data,
                                                                           loss="MSE",  # "MSE" | "peak-aware-MSE"
                                                                           split_method="time-series",  # "time-series" | "seasonal-time-series" | "random"
                                                                           epochs=200,
                                                                           prediction_steps_ahead=96,
                                                                           plot=True,
                                                                           plot_save_dir=r"C:\Users\alder\Downloads\load_prediction_15min.png")