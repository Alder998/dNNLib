"""tester for geo-spatial model from loaded"""

from ModelLoader import ModelLoader as load
import pandas as pd

weatherData = pd.read_csv(r"C:\Users\alder\Downloads\weather_liguria_1y_100points.csv")

load.ModelLoader().predictGeoSpaceWithLoadedModel(modelPath="D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models\\geospace_GRU_model",
                                                  data=weatherData,
                                                  steps_ahead=20,
                                                  frequency="1h",
                                                  date_column="date",
                                                  target_variable="temperature")