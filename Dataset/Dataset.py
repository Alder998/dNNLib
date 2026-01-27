"""Class to load some dataset to test NN performance"""

import pandas as pd

class Dataset:

    def __init__(self):
        pass

    # 0. Italy Energy Production Dataset
    def getItalyEnergyProductionDataset (self, freq="15m"):

        # 0.0. get some data (2025 15 min-power production in Italy)
        if freq == "15min":
            dataset = pd.read_excel(r"C:\\Users\\alder\\Downloads\\Export-DownloadCenterFile-20260103-165403.xlsx")
        elif freq == "1h":
            dataset = pd.read_excel(r"C:\\Users\\alder\\Downloads\\2021-2025_hourly_prod.xlsx")
        else:
            raise Exception("Frequency " + str(freq) + " is not supported.")
        # 0.1. Clean data
        energy_prod_italy = []
        for category in dataset["Primary Source"].unique():
            dfc = dataset[["Date", "Actual Generation"]][dataset["Primary Source"] == category].reset_index(drop=True)
            dfc = dfc.rename(columns={"Actual Generation": category})
            energy_prod_italy.append(dfc.set_index("Date"))
        energy_prod_italy = pd.concat([df for df in energy_prod_italy], axis=1)
        energy_prod_italy["year"] = energy_prod_italy.index.year
        energy_prod_italy["month"] = energy_prod_italy.index.month
        energy_prod_italy["day"] = energy_prod_italy.index.day
        energy_prod_italy["day_of_week"] = energy_prod_italy.index.dayofweek
        energy_prod_italy["hour"] = energy_prod_italy.index.hour
        energy_prod_italy["minute"] = energy_prod_italy.index.minute

        return energy_prod_italy

    def loadWeatherDataset (self):

        # 0.0. Load the .csv data
        wdata = pd.read_csv("C:\\Users\\alder\\Downloads\\1m_weather.xlsx")

        # 0.1. Convert the date columns to datetime
        wdata["date"] = pd.to_datetime(wdata["date"])

        # 0.2. Create time variables
        wdata["year"] = wdata["date"].dt.year
        wdata["month"] = wdata["date"].dt.month
        wdata["day"] = wdata["date"].dt.day
        wdata["hour"] = wdata["date"].dt.hour

        return wdata.sort_values(by="date", ascending=True).reset_index(drop=True)