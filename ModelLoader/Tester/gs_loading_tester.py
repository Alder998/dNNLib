"""tester for geo-spatial model from loaded"""

from ModelLoader import ModelLoader as load
from Dataset import Dataset as data

weatherData = data.Dataset().loadWeatherDataset(size="1m")

load.ModelLoader().predictGeoSpaceWithLoadedModel(modelPath="D:\\PythonProjects-Storage\\dNNLib\\Tester\\stored_models\\geospace_GRU_model",
                                                  data=weatherData,
                                                  steps_ahead=20,
                                                  frequency="1h",
                                                  date_column="date",
                                                  target_variable="temperature")