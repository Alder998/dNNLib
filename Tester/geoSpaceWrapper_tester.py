from Wrappers import GeoSpaceModelWrapper as geo
import pandas as pd

data = pd.read_csv(r"C:\Users\alder\Downloads\energy_market_data_1p.xlsx")
data = data[data["map_code"].isin(['NO-NO1','NO-NO2','NO-NO3','NO-NO4','SE-SE1','SE-SE2','SE-SE3','SE-SE4',
                                   'FI','DE','FR','PL','ES'])].reset_index(drop=True)

geo.GeoSpaceModelWrapper(modelStructure={"GraphGRU": {"layers": [64], "activation": "tanh"}},
                         space_variables=["lat","lon"],
                         feature_variables=["day","hour","minute"],
                         time_window=288,
                         target_variables="DAM Price",
                         date_column="Date",
                         frequency="15min").trainPredictAndSaveGeospaceModel(data=data,
                                                                             epochs=100,
                                                                             prediction_steps_ahead=20,
                                                                             plot=True,
                                                                             save_dir="D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models",
                                                                             model_save_name="geospace_GRU_model",
                                                                             target_division=1000)