"""
Tester for simple MLP Model
"""
from Wrappers import MLPWrapper as mlp
import pandas as pd

data = pd.read_csv("C:\\Users\\alder\\Downloads\\energy_market_data_italia_1p.csv")
data["Date"] = pd.to_datetime(data["Date"])
data["Date"] = data["Date"].dt.tz_localize(None)
data = data[(data["zone"]=="Italia Sicilia") & (data["lat"]==data[data["zone"] == "Italia Sicilia"]["lat"].reset_index(drop=True)[0]) &
            (data["lon"]==data[data["zone"] == "Italia Sicilia"]["lon"].reset_index(drop=True)[0])].sort_values(by="Date", ascending=True).reset_index(drop=True)

mlp.MLPWrapper(modelStructure = {"FF": {"layers": [300, 300, 300], "activation": "relu", "dropout": 0.0}},
               feature_variables=["Solar_Actual Aggregated"],
               target_variables=["DAM Price"]).train_model(data=data, test_size=0.30, epochs=30)