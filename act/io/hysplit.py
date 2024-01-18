import os
import xarray as xr
import numpy as np

from datetime import datetime
from .csv import read_csv

def read_hysplit(filename):
    """
    Reads an input HYSPLIT trajectory for plotting in ACT.

    Parameters
    ----------
    filename: str
        The input file name.

    Returns
    -------
    ds: xarray Dataset
        The ACT dataset containing the HYSPLIT trajectories
    """

    ds = xr.Dataset({})
    num_lines = 0
    with open(filename, 'r') as filebuf:
        num_grids = int(filebuf.readline().split()[0])
        num_lines += 1
        grid_times = []
        grid_names = []
        forecast_hours = np.zeros(num_grids)
        for i in range(num_grids):
            data = filebuf.readline().split()
            num_lines += 1
            grid_names.append(data[0])
            grid_times.append(
                datetime(year=int(data[1]), month=int(data[2]), day=int(data[3]), hour=int(data[4])))
            forecast_hours[i] = int(data[5])
        ds["forecast_hour"] = xr.DataArray(forecast_hours, dims=["num_grids"])
        ds.attrs["standard_name"] = "Grid forecast hour"
        ds.attrs["units"] = "Hour [UTC]"
        ds["grid_times"] = xr.DataArray(np.array(grid_times), dims=["num_grids"])
        data_line = filebuf.readline().split()
        num_lines += 1
        ds.attrs["trajectory_direction"] = data_line[1]
        ds.attrs["vertical_motion_calculation_method"] = data_line[2]
        num_traj = int(data_line[0])
        traj_times = []
        start_lats = np.zeros(num_traj)
        start_lons = np.zeros(num_traj)
        start_alt = np.zeros(num_traj)
        for i in range(num_traj):
            data = filebuf.readline().split()
            num_lines += 1
            traj_times.append(
                datetime(year=int(data[0]), month=int(data[1]), day=int(data[2]), hour=int(data[3])))
            start_lats[i] = float(data[4])
            start_lons[i] = float(data[5])
            start_alt[i] = float(data[6])

        ds["start_latitude"] = xr.DataArray(start_lats, dims=["num_trajectories"])
        ds["start_latitude"].attrs["long_name"] = "Trajectory start latitude"
        ds["start_latitude"].attrs["units"] = "degree"
        ds["start_longitude"] = xr.DataArray(start_lats, dims=["num_trajectories"])
        ds["start_longitude"].attrs["long_name"] = "Trajectory start longitude"
        ds["start_longitude"].attrs["units"] = "degree"
        ds["start_altitude"] = xr.DataArray(start_alt, dims=["num_trajectories"])
        ds["start_altitude"].attrs["long_name"] = "Trajectory start altitude"
        ds["start_altitude"].attrs["units"] = "degree"

        data = filebuf.readline().split()
        num_lines += 1
        var_list = ["trajectory_number", "grid_number", "year", "month", "day",
                     "hour", "minute", "forecast_hour", "age", "latitude", "longitude", "altitude"]
        for variable in data[1:]:
            var_list.append(variable)
    ds = ds.merge(pd.read_csv(filename, delim_whitespace=True, column_names=var_list, skiprows=num_lines).to_xarray())
    
    return ds
