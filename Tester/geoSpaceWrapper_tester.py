from Wrappers import GeoSpaceModelWrapper as geo
import pandas as pd

data = pd.read_csv(r"C:\Users\alder\Downloads\1mo_weather.csv")

geo.GeoSpaceModelWrapper(modelStructure={"GraphGRU": {"layers": [64], "activation": "tanh"}},
                         space_variables=["latitude","longitude"],
                         feature_variables=["month","day","hour"],
                         time_window=96,
                         target_variables="temperature",
                         date_column="date",
                         frequency="1h").trainPredictAndSaveGeospaceModel(data=data,
                                                                          epochs=300,
                                                                          prediction_steps_ahead=20,
                                                                          plot=True)