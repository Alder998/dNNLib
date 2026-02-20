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
        prediction_dataset[date_column] = pd.to_datetime(prediction_dataset[date_column])

        # Mean
        mean_ts = prediction_dataset.groupby(date_column)[variable].mean().sort_index()

        # Set the grid
        lats = np.sort(prediction_dataset["latitude"].unique())
        lons = np.sort(prediction_dataset["longitude"].unique())

        lat_to_i = {v: i for i, v in enumerate(lats)}
        lon_to_j = {v: j for j, v in enumerate(lons)}

        time_steps = sorted(prediction_dataset[date_column].unique())

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
        first = prediction_dataset[prediction_dataset[date_column] == time_steps[0]]

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
            subset = prediction_dataset[prediction_dataset[date_column] == t]
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
            ani.save(savePath,
                     writer="pillow", fps=5)
        # Show Plot
        plt.show()
