"""Class to load some dataset to test NN performance"""

import pandas as pd

class Dataset:

    def __init__(self):
        pass

    # 0. Italy Energy Production Dataset
    def getItalyEnergyProductionDataset (self):

        # 0.0. get some data (2025 15 min-power production in Italy)
        dataset = pd.read_excel(r"C:\\Users\\alder\\Downloads\\Export-DownloadCenterFile-20260103-165403.xlsx")
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
        energy_prod_italy["hour"] = energy_prod_italy.index.hour
        energy_prod_italy["minute"] = energy_prod_italy.index.minute

        return energy_prod_italy