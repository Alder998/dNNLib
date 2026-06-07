"""
Tester for Binary Classification Models
"""

from Wrappers import BinaryClassificationWrapper as bcm
import pandas as pd

# 0. Load data
data = pd.read_excel(r"C:\Users\alder\Downloads\meteo_classification.xlsx").dropna()

# 1. Launch Model
bcm.BinaryClassificationWrapper(modelStructure={"FF": {"layers": [300, 300, 300], "activation": "relu", "dropout": 0.0}},
                                target_variables=["Rain_binary"],
                                feature_variables=["Temperatura media (°C)", "Temperatura massima (°C)",
                                                   "Temperatura minima (°C)",
                                                   "Umidità media (%)", "Umidità massima (%)", "Umidità minima (%)",
                                                   "Pressione media sul livello del mare (mb)"]).train_multiClass_model(data=data, epochs=100, test_size=0.30)