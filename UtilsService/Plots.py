"""Plotting Utils (maily for geospace representation)"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

class Plots:

    def __init__(self):
        pass

    # Utils-like functions to interpolate for plots

    def plotGeospacePredictionFixedGrid (self, prediction_dataset, variable, date_column="date",
                                         colorScale="rainbow", space_variables=["latitude", "longitude"],
                                         ncols=2, geojson=None):

        df = prediction_dataset.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        dates = np.sort(df[date_column].unique())

        # -------------------------
        # preparazione dati
        # -------------------------

        spatial = {}
        ts = {}
        for var in variable:
            spatial[var] = (df.groupby(["date", space_variables[0], space_variables[1]])[var].mean().reset_index())
            ts[var] = (df.groupby("date")[var].mean())

        # -------------------------
        # layout
        # -------------------------

        n = len(variable)
        nrows = int(np.ceil(n / ncols))

        fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
        gs = fig.add_gridspec(nrows * 2,ncols,height_ratios=[3, 1] * nrows)

        maps = {}
        cursors = {}
        points = {}
        scatters = {}

        for i, var in enumerate(variable):
            r = (i // ncols) * 2
            c = i % ncols

            ax_map = fig.add_subplot(gs[r, c])
            ax_ts = fig.add_subplot(gs[r + 1, c])

            # Set the map graph
            current = spatial[var][spatial[var].date == dates[0]]
            grid = current.pivot(index=space_variables[0], columns=space_variables[1], values=var)

            lat = grid.index.values
            lon = grid.columns.values
            Z = grid.values
            vmin = spatial[var][var].min()
            vmax = spatial[var][var].max()
            mesh = ax_map.pcolormesh(lon, lat, Z, shading="auto", cmap=colorScale, vmin=vmin, vmax=vmax)

            # Time series
            ax_map.set_title(var)
            ax_ts.plot(ts[var].index, ts[var].values, color="black")
            cursor = ax_ts.axvline(dates[0], color="black", linestyle="dashed")
            point, = ax_ts.plot( dates[0], ts[var].iloc[0], "o", color="black")

            maps[var] = ax_map
            scatters[var] = mesh
            cursors[var] = cursor
            points[var] = point

        # -------------------------
        # animazione
        # -------------------------

        def update(frame):

            date = dates[frame]

            for var in variable:
                current = spatial[var][spatial[var].date == date]

                scatters[var].set_offsets(np.c_[current.longitude,current.latitude])
                scatters[var].set_array(current[var].values)

                maps[var].set_title(f"{var} - {datetime.strftime(pd.to_datetime(date), "%Y-%m-%d %H")}")
                cursors[var].set_xdata([date, date])
                points[var].set_data([date],[ts[var].loc[date]])

            return list(scatters.values())

        anim = FuncAnimation(
            fig,
            update,
            frames=len(dates),
            interval=200
        )

        plt.tight_layout()
        plt.show()

        return anim

    # function to plot time series prediction
    def plotTimeSeriesPrediction(self, dataInDataFrameFormat, prediction_dataset, prediction_dataset_upper,
                                 prediction_dataset_lower,  variable, frequency="1h", date_column="date", savePath=None, target_division=1):

        if date_column != "index":
            dataInDataFrameFormat = dataInDataFrameFormat.set_index(date_column)

        fig, axes = plt.subplots(len(prediction_dataset.columns), 1, figsize=(15, 8))
        if len(variable) == 1:
            axes = [axes]
        for i, var in enumerate(variable):
            if date_column != "index":
                axes[i].plot(dataInDataFrameFormat.sort_values(by=date_column, ascending=True)[var][(-672 if frequency == "15min" else -168):])
            else:
                axes[i].plot(dataInDataFrameFormat.sort_index(ascending=True)[var][(-672 if frequency == "15min" else -168):])
            axes[i].plot(prediction_dataset[var], color="red", linestyle="dashed")
            axes[i].fill_between(prediction_dataset_lower.index, prediction_dataset_lower[var], prediction_dataset_upper[var], color="red", alpha=0.1)
            axes[i].set_title("Prediction: " + str(var))
        # save, layout, other useful stuff to show the graph
        if savePath is not None:
            plt.savefig(savePath, dpi=500)
        plt.tight_layout()
        plt.show()

    # Function to plot having a geopandas geometry element
    #def plotGeoSpaceWithGPGeometry (self, dataInDataFrameFormat, prediction_dataset, variable, date_column="date", colorScale="rainbow", savePath=None):