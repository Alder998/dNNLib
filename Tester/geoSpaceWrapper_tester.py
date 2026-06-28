from Wrappers import GeoSpaceModelWrapper as geo
import pandas as pd

data = pd.read_csv(r"C:\Users\alder\Downloads\weather_liguria1.csv")

geo.GeoSpaceModelWrapper(modelStructure={"GraphGRU": {"layers": [64], "activation": "tanh"}},
                         space_variables=["latitude","longitude"],
                         feature_variables=["month","day","hour"],
                         time_window=48,
                         target_variables="temperature",
                         date_column="date",
                         frequency="1h",
                         lags=[]).trainPredictAndSaveGeospaceModel(data=data,
                                                                   epochs=200,
                                                                   standardize=True,
                                                                   scaler="min-max",
                                                                   date_column_format="%Y-%m-%dT%H:%M",
                                                                   split_method="time-series",
                                                                   seasonal_splits=3,
                                                                   prediction_steps_ahead=48,
                                                                   plot=True,
                                                                   save_dir=None, #"D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models",
                                                                   model_save_name=None, #"geospace_GRU_model",
                                                                   target_division=1)