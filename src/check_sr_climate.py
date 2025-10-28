import os
import xarray as xr
import numpy as np

def is_epsg4326(file_path):
    ds = xr.open_dataset(file_path)
    lat_names = [v for v in ds.variables if "lat" in v.lower()]
    lon_names = [v for v in ds.variables if "lon" in v.lower()]
    if not lat_names or not lon_names:
        ds.close()
        return False
    lat = ds[lat_names[0]].values
    lon = ds[lon_names[0]].values
    lat_ok = (lat.min() >= -90) and (lat.max() <= 90)
    lon_ok = (lon.min() >= -180) and (lon.max() <= 360)
    lat_uniform = np.allclose(np.diff(lat), np.diff(lat)[0])
    lon_uniform = np.allclose(np.diff(lon), np.diff(lon)[0])

    lat_ok = (lat[0] > lat[-1]) & lat_ok & lat_uniform
    lon_ok = (lon[0] < lon[-1]) & lon_ok & lon_uniform
    ds.close()
    return lat_ok and lon_ok


def check_climate_data(config, silent=False, mode='Temp'):
    filename = os.path.join(config['files']['climate_data_dir'], 'Temp_avg.nc' if mode=='Temp' else 'Prec_avg.nc')
    ok = is_epsg4326(filename)
    if silent and not ok:
        print(f'\nWARNING: {filename} does not does not match the expected data format.\nCropSuite may crash or produce incorrect results!\n')
    elif not silent and not ok:
        input((f'\nWARNING: {filename} does not does not match the expected data format.\nCropSuite may crash or produce incorrect results!\nPress enter to continue anyway...'))