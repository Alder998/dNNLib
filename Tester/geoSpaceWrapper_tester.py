from Wrappers import GeoSpaceModelWrapper as geo
import pandas as pd

data = pd.read_csv(r"C:\Users\alder\Downloads\italy_load_by_zones_1p.csv")

geo.GeoSpaceModelWrapper(modelStructure={"GraphGRU": {"layers": [64], "activation": "tanh"}},
                         space_variables=["latitude","longitude"],
                         feature_variables=["month","day","hour","minute"],
                         time_window=96,
                         target_variables="DAM Price",
                         date_column="Date",
                         frequency="15min").trainPredictAndSaveGeospaceModel(data=data,
                                                                          epochs=500,
                                                                          prediction_steps_ahead=20,
                                                                          plot=True)