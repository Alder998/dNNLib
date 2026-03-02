"""Tester for Time Series Wrapper"""

from Wrappers import TimeSeriesWrapper as ts
from Dataset import Dataset as data

# 0. Data
data = data.Dataset().getItalyEnergyProductionDataset(freq="15min")

# 1. Model
ts.TimeSeriesWrapper(modelStructure={"MultiSeasonConv1DGated": {"layers": [16, 16], "cycles": [96, 672],
                                                                "mix_units": 32, "use_layer_norm": True, "baseline_kernel": 192,
                                                                "use_cross_cycle_attention": True, "use_seasonal_memory": False},
                                      "LSTM": {"layers": [128, 64], "activation": "tanh", "dropout": 0.0},
                                      "FF": {"layers": [200, 200], "activation": "relu"}},
                     feature_variables=["year","month","day","hour","minute"],
                     time_window=96,
                     target_variables="Thermal",
                     date_column="index",
                     frequency="15min").trainPredictAndSaveTimeSeriesModel(data=data,
                                                                           loss="MSE",  # "MSE" | "peak-aware-MSE"
                                                                           epochs=200,
                                                                           prediction_steps_ahead=96,
                                                                           plot=True)