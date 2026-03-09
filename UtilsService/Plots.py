"""Plotting Utils (maily for geospace representation)"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

class Plots:

    def __init__(self):
        pass

    def plotGeospacePredictionFixedGrid (self, prediction_dataset, variable, date_column="date", colorScale="rainbow", savePath=None):

        print("INFO - Generating Animation...")
        prediction_dataset["date"] = pd.to_datetime(prediction_dataset["date"])

        # Mean
        mean_ts = prediction_dataset.groupby("date")[variable].mean().sort_index()

        # Set the grid
        lats = np.sort(prediction_dataset["latitude"].unique())
        lons = np.sort(prediction_dataset["longitude"].unique())

        lat_to_i = {v: i for i, v in enumerate(lats)}
        lon_to_j = {v: j for j, v in enumerate(lons)}

        time_steps = sorted(prediction_dataset["date"].unique())

        vmin = prediction_dataset[variable].min()
        vmax = prediction_dataset[variable].max()
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Figure with two graphs
        fig, (ax_map, ax_ts) = plt.subplots(
            2, 1, figsize=(8, 10),
            gridspec_kw={"height_ratios": [3, 1]}
        )

        # Heatmap on fist graph
        grid0 = np.full((len(lats), len(lons)), np.nan)
        first = prediction_dataset[prediction_dataset["date"] == time_steps[0]]

        for _, r in first.iterrows():
            grid0[lat_to_i[r["latitude"]], lon_to_j[r["longitude"]]] = r[variable]

        im = ax_map.imshow(
            grid0,
            origin="lower",
            cmap=colorScale,
            norm=norm,
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            aspect="auto"
        )

        ax_map.set_axis_off()
        cbar = plt.colorbar(im, ax=ax_map)
        cbar.set_label(variable)

        # Static Mean Graph
        ax_ts.plot(mean_ts.index, mean_ts.values)
        ax_ts.set_ylabel("Mean")
        ax_ts.set_xlabel("Date")

        # Vertical line to follow Animation
        time_line = ax_ts.axvline(time_steps[0], linestyle="--")

        # Update function
        def update(frame):
            t = time_steps[frame]
            subset = prediction_dataset[prediction_dataset["date"] == t]
            grid = np.full((len(lats), len(lons)), np.nan)
            for _, r in subset.iterrows():
                grid[lat_to_i[r["latitude"]], lon_to_j[r["longitude"]]] = r[variable]
            im.set_data(grid)
            ax_map.set_title(f"{variable}: {t.strftime('%d/%m/%Y %H:%M')}")

            # Update line graph
            time_line.set_xdata([t])

            return im, time_line

        ani = FuncAnimation(fig, update, frames=len(time_steps), interval=300)

        if savePath is not None:
            ani.save(savePath, writer="pillow", fps=5)
        # Show Plot
        plt.show()

    # function to plot time series prediction
    def plotTimeSeriesPrediction(self, dataInDataFrameFormat, prediction_dataset, prediction_dataset_upper,
                                 prediction_dataset_lower,  variable, frequency="1h", date_column="date",savePath=None):

        if date_column != "index":
            dataInDataFrameFormat = dataInDataFrameFormat.set_index(date_column)
        plt.figure(figsize=(15, 5))
        if date_column != "index":
            plt.plot(dataInDataFrameFormat.sort_values(by=date_column, ascending=True)[variable][(-672 if frequency == "15min" else -168):])
        else:
            plt.plot(dataInDataFrameFormat.sort_index(ascending=True)[variable][(-672 if frequency == "15min" else -168):])
        plt.plot(prediction_dataset[variable], color="red", linestyle="dashed")
        plt.fill_between(prediction_dataset_lower.index, prediction_dataset_lower[variable], prediction_dataset_upper[variable], color="red", alpha=0.1)
        if savePath is not None:
            plt.savefig(savePath, dpi=500)
        plt.show()