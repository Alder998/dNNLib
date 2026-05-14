from Wrappers import GeoSpaceModelWrapper as geo
import pandas as pd

data = pd.read_csv(r"C:\Users\alder\Downloads\1mo_weather.csv")

geo.GeoSpaceModelWrapper(modelStructure={"GraphGRU": {"layers": [64], "activation": "tanh"}},
                         space_variables=["latitude","longitude"],
                         feature_variables=["month","day","hour"],
                         time_window=96,
                         target_variables="windSpeed",
                         date_column="date",
                         frequency="1h",
                         lags=[]).trainPredictAndSaveGeospaceModel(data=data,
                                                                 epochs=30,
                                                                 prediction_steps_ahead=96,
                                                                 plot=True,
                                                                 save_dir="D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models",
                                                                 model_save_name="geospace_GRU_model",
                                                                 target_division=1)