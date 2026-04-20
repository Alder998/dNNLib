from Wrappers import GeoSpaceModelWrapper as geo
import pandas as pd

data = pd.read_csv(r"C:\Users\alder\Downloads\energy_market_data_italia_1p.xlsx")

geo.GeoSpaceModelWrapper(modelStructure={"GraphGRU": {"layers": [64], "activation": "tanh"}},
                         space_variables=["lat","lon"],
                         feature_variables=["day","hour","minute"],
                         time_window=288,
                         target_variables="DAM Price",
                         date_column="Date",
                         frequency="15min",
                         lags=[24, 48, 72]).trainPredictAndSaveGeospaceModel(data=data,
                                                                   epochs=10,
                                                                   prediction_steps_ahead=20,
                                                                   plot=True,
                                                                   save_dir="D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models",
                                                                   model_save_name="geospace_GRU_model",
                                                                   target_division=100)