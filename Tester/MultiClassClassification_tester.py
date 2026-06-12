"""
Simple tester for Multi-Class Classification
"""

import pandas as pd
from Wrappers import MultiClassClassificationWrapper as mcc

# 0. Read Data
data = pd.read_excel(r"C:\Users\alder\Downloads\meteo_classification.xlsx").dropna()

# "Temperatura media (°C)", "Temperatura massima (°C)",
# "Temperatura minima (°C)", "Punto di rugiada (°C)",
# "Umidità media (%)", "Umidità massima (%)", "Umidità minima (%)",
# "Velocità del vento media (km/h)", "Velocità del vento media (km/h)",
# "Velocità massima del vento (km/h)", "Pressione media sul livello del mare (mb)",
# "Pioggia (mm)"

# 1. Launch Model
mcc.MultiClassClassificationWrapper(modelStructure={"FF": {"layers": [300, 300, 300], "activation": "relu", "dropout": 0.0}},
                                    target_variables=["Rain_binary"],
                                    feature_variables=["Temperatura media (°C)", "Temperatura massima (°C)",
                                                       "Temperatura minima (°C)",
                                                       "Umidità media (%)", "Umidità massima (%)", "Umidità minima (%)",
                                                       "Pressione media sul livello del mare (mb)"]).train_multiClass_model(data=data, epochs=100, test_size=0.30)