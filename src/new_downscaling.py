import os
import numpy as np
import rasterio
try:
    import data_tools as dt
    import nc_tools as nc
    import temp_interpolation as ti
    import prec_interpolation as pi
except:
    from src import data_tools as dt
    from src import nc_tools as nc
    from src import temp_interpolation as ti
    from src import prec_interpolation as pi
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import shutil
import psutil
import zarr_tools as zt
import zarr
from functools import partial
from datetime import datetime
import sys
import xarray as xr
temp = os.path.join(os.getcwd(), 'temp')
os.makedirs(temp, exist_ok=True)


suffix = '.npz'
mode = 'npz_compressed'
saver = {'zarr': zt.savez, 'npz': np.savez, 'npz_compressed': np.savez_compressed}.get(mode, np.savez)
loader = zt.load if mode == 'zarr' else np.load


def load(filename, *args, **kwargs):
    if filename.endswith('.npz'):
        loader(filename, *args, **kwargs)
    elif filename.endswith('.zarr'):
        loader(filename, *args, **kwargs)


def save(filename, *args, **kwargs):
    if mode == 'zarr':
        filename.replace('.npz', '.zarr')
        saver(filename, *args, **kwargs)
    else:
        filename.replace('.zarr', '.npz')
        saver(filename, *args, **kwargs)


def interpolate_temperature(config_file, domain, area_name):
    """
        domain: [y_max, x_min, y_min, x_max]
    """
    output_dir = os.path.join(config_file['files']['output_dir']+'_downscaled', area_name)
    os.makedirs(output_dir, exist_ok=True)
    print(' -> Downscaling temperature data...')
    temp_files = worldclim_downscaling_temp(domain, config_file, output_dir)

    return temp_files


def interpolate_precipitation(config_file, domain, area_name):
    output_dir = os.path.join(config_file['files']['output_dir']+'_downscaled', area_name)
    os.makedirs(output_dir, exist_ok=True)
    print(' -> Downscaling precipitation data...')
    prec_files = worldclim_downscaling_prec(domain, config_file, output_dir)

    return prec_files

def read_timestep(filename, extent, timestep=-1):
    if timestep == -1:
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            with rasterio.open(filename) as src:
                data = np.asarray(src.read(), dtype=np.float16)
                bounds = src.bounds
                nodata = src.nodata
            data = dt.extract_domain_from_global_raster(data, extent, raster_extent=bounds)
            return data, nodata
        else:
            data, _ = nc.read_area_from_netcdf(filename, extent=[extent[1], extent[0], extent[3], extent[2]])
            data = np.asarray(data).transpose(2, 0, 1)
            nodata = nc.get_nodata_value(filename)
            return data, nodata
    else:
        if filename.endswith('.tif') or filename.endswith('.tiff'):
            data, nodata = dt.load_specified_lines(filename, extent, all_bands=timestep+1) #type:ignore
            return data, nodata
        else:
            varname = nc.get_variable_name_from_nc(filename)
            data, _ = nc.read_area_from_netcdf(filename, extent=extent, day_range=timestep, variable=varname)
            nodata = nc.get_nodata_value(filename)
            return data, nodata


def worldclim_downscaling_temp(extent, config_file, output_dir):
    if os.path.exists(os.path.join(output_dir, 'temperature.zarr')):
        print('    Temperature data already downscaled.')
        return os.path.join(output_dir, 'temperature.zarr')
    
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    world_clim_data_dir = os.path.join(config_file['files']['worldclim_temperature_data_dir'])

    fine_resolution = dt.get_resolution_array(config_file, extent, True)
    
    land_sea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    land_sea_mask = np.asarray(land_sea_mask).astype(np.float16)
    land_sea_mask[land_sea_mask == 0] = np.nan
    if land_sea_mask.shape[0] != fine_resolution[0]:
        land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
    if not os.path.exists(os.path.join(climate_data_dir, 'Temp_avg.tif')):
        temp_file = os.path.join(climate_data_dir, 'Temp_avg.nc')
    else:
        temp_file = os.path.join(climate_data_dir, 'Temp_avg.tif')

    coarse_resolution = read_timestep(temp_file, extent=extent, timestep=0)[0].shape
    ext = nc.check_extent_load_file(os.path.join(world_clim_data_dir, f'factors_month_1.nc'), extent=extent)
    [os.remove(file) for i in range(1, 13) if not ext and (file := os.path.join(world_clim_data_dir, f'factors_month_{i}.nc')) and os.path.exists(file)]
    if not all(os.path.exists(os.path.join(world_clim_data_dir, f'factors_month_{band_index}.nc')) for band_index in range(1, 13)):
        world_clim_data_dir = ti.calculate_temp_factors(fine_resolution, coarse_resolution, world_clim_data_dir, extent)

    temp_data, _ = dt.load_specified_lines(temp_file, extent=extent)
    if np.nanmin(temp_data[0]) > 150:
        temp_data -= 273.15

    chunk_h = 512
    chunk_w = int(fine_resolution[1])

    store = zarr.open(os.path.join(output_dir, 'temperature.zarr'), mode="w", shape=(365, *fine_resolution), chunks=(1, chunk_h, chunk_w), dtype='i2')
    store.attrs["extent"] = extent
    process_day(temp_data, extent, fine_resolution, store, mode='temp', nan_mask=np.isnan(land_sea_mask), worldclim_data_dir=world_clim_data_dir, chunk_size=(chunk_h, chunk_w))
    return os.path.join(output_dir, 'temperature.zarr')


def worldclim_downscaling_prec(extent, config_file, output_dir):
    if os.path.exists(os.path.join(output_dir, 'precipitation.zarr')):
        print('    Precipitation data already downscaled.')
        return os.path.join(output_dir, 'precipitation.zarr')
    
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    world_clim_data_dir = os.path.join(config_file['files']['worldclim_precipitation_data_dir'])

    fine_resolution = dt.get_resolution_array(config_file, extent, True)
    
    land_sea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    land_sea_mask = np.asarray(land_sea_mask).astype(np.float16)
    land_sea_mask[land_sea_mask == 0] = np.nan
    if land_sea_mask.shape[0] != fine_resolution[0]:
        land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
    if not os.path.exists(os.path.join(climate_data_dir, 'Prec_avg.tif')):
        prec_file = os.path.join(climate_data_dir, 'Prec_avg.nc')
    else:
        prec_file = os.path.join(climate_data_dir, 'Prec_avg.tif')

    coarse_resolution = read_timestep(prec_file, extent=extent, timestep=0)[0].shape
    ext = nc.check_extent_load_file(os.path.join(world_clim_data_dir, f'factors_month_1.nc'), extent=extent)
    [os.remove(file) for i in range(1, 13) if not ext and (file := os.path.join(world_clim_data_dir, f'factors_month_{i}.nc')) and os.path.exists(file)]
    if not all(os.path.exists(os.path.join(world_clim_data_dir, f'factors_month_{band_index}.nc')) for band_index in range(1, 13)):
        world_clim_data_dir = pi.calculate_prec_factors(fine_resolution, coarse_resolution, world_clim_data_dir, extent)

    prec_data, _ = dt.load_specified_lines(prec_file, extent=extent)
    if np.nanmax(prec_data[0]) < 5:
        prec_data *= 3600 * 24

    drizzle_threshold = float(config_file.get('options', {}).get('downscaling_precipitation_per_day_threshold', 1.0))
    prec_data[prec_data < drizzle_threshold] = 0

    chunk_h = 512
    chunk_w = int(fine_resolution[1])
    store = zarr.open(os.path.join(output_dir, 'precipitation.zarr'), mode="w", shape=(365, *fine_resolution), chunks=(1, chunk_h, chunk_w), dtype='i2')
    store.attrs["extent"] = extent
    process_day(prec_data, extent, fine_resolution, store, mode='prec', nan_mask=np.isnan(land_sea_mask), worldclim_data_dir=world_clim_data_dir, chunk_size=(chunk_h, chunk_w))

    return os.path.join(output_dir, 'precipitation.zarr')


@njit(parallel=True, cache=True, fastmath=True)
def _bilinear_resize(src, out_h, out_w):
    in_h, in_w = src.shape
    out = np.empty((out_h, out_w), dtype=np.float32)
    ry = (in_h - 1) / (out_h - 1) if out_h > 1 else 0.0
    rx = (in_w - 1) / (out_w - 1) if out_w > 1 else 0.0
    for y in prange(out_h):
        fy = ry * y
        y0 = int(fy)
        y1 = y0 + 1 if y0 + 1 < in_h else y0
        wy = fy - y0
        for x in range(out_w):
            fx = rx * x
            x0 = int(fx)
            x1 = x0 + 1 if x0 + 1 < in_w else x0
            wx = fx - x0
            v00 = src[y0, x0]
            v01 = src[y0, x1]
            v10 = src[y1, x0]
            v11 = src[y1, x1]
            top = v00 * (1.0 - wx) + v01 * wx
            bot = v10 * (1.0 - wx) + v11 * wx
            out[y, x] = top * (1.0 - wy) + bot * wy
    return out

@njit(parallel=True, cache=True, fastmath=True)
def _process_day_core(day2d, factor2d, mode_flag, out_h, out_w, nan_mask):
    up = _bilinear_resize(day2d, out_h, out_w)
    out = np.empty((out_h, out_w), dtype=np.int16)
    nodata = -32767
    if mode_flag == 0:
        for y in prange(out_h):
            for x in range(out_w):
                f = factor2d[y, x]
                u = up[y, x]
                if (f != f) or (u != u) or nan_mask[y, x]:
                    out[y, x] = nodata
                else:
                    out[y, x] = int(np.round((u - f) * 10))
    else:
        for y in prange(out_h):
            for x in range(out_w):
                f = factor2d[y, x]
                u = up[y, x]
                if (f != f) or (u != u) or nan_mask[y, x]:
                    out[y, x] = nodata
                else:
                    out[y, x] = int(np.round((u * f) * 10))
    return out


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    max_len, _ = shutil.get_terminal_size(fallback=(80, 24))
    length = np.min([length, int(max_len) - len(prefix) - 8 - len(suffix)])
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


def process_day(data, extent, fine_resolution, store, mode, nan_mask, worldclim_data_dir, n_threads=None, chunk_size=(1024,1024)):
    out_h, out_w = fine_resolution
    mode_flag = 0 if mode == 'temp' else (1 if mode == 'prec' else 2)
    month_ends = np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365], dtype=np.int16)
    month_map = np.searchsorted(month_ends, np.arange(365, dtype=np.int16))
    if n_threads is None and sys.platform == 'win32':
        avail_gb = psutil.virtual_memory().available / (1024**3)
        gb_per_thread = float(np.prod(fine_resolution)) * np.dtype(np.int16).itemsize * 2 / (1024**3) * 1.5
        cpu_cores = os.cpu_count() or 1
        n_threads = min(cpu_cores, max(1, int(avail_gb // gb_per_thread)))
        limit = "RAM" if n_threads < cpu_cores else "CPU-Cores"
        print(f' -> Multithreading: {n_threads} threads, limited by {limit}.')
        if limit == 'RAM':
            print(f'    Minimum RAM required to use all {cpu_cores} cores: {10 + (gb_per_thread*cpu_cores):.0f} GB.')

    print_progress_bar(0, 12, prefix='    Progress', suffix=f'Month {1}/{12}', decimals=0, length=50, fill='█')

    chunk_size = (chunk_size[0], out_w)

    for month in range(12):
        days_in_month = np.where(month_map == month)[0]
        if len(days_in_month) == 0:
            continue

        fac = nc.read_area_from_netcdf(os.path.join(worldclim_data_dir, f'factors_month_{month+1}.nc'), extent=extent)[0].astype(np.float32, copy=False)
        fac = _bilinear_resize(fac, out_h, out_w)

        def worker(day):
            day2d = np.array(data[day], dtype=np.float32, copy=False)
            if np.isnan(day2d).any():
                mask = np.isnan(day2d)
                day2d[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), day2d[~mask])
            try:
                out2d = _process_day_core(day2d, fac, mode_flag, out_h, out_w, nan_mask)
                for y0 in range(0, out_h, chunk_size[0]):
                    y1 = min(y0 + chunk_size[0], out_h)
                    store[day, y0:y1, :] = out2d[y0:y1, :]
            except Exception as e:
                #print(f'Error during downscaling for day {day}: {e} Trying again')
                worker(day)

        if sys.platform != 'win32':
            for day in days_in_month:
                worker(day)
        else:
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                list(ex.map(worker, days_in_month))

        print_progress_bar(month+1, 12, prefix='    Progress', suffix=f'Month {month+1}/{12}', decimals=0, length=50, fill='█')
    print()


@njit(parallel=True, cache=True, fastmath=True)
def _bilinear_resize_int(src, out_h, out_w):
    in_h, in_w = src.shape
    out = np.empty((out_h, out_w), dtype=np.int8)
    ry = (in_h << 8) / out_h
    rx = (in_w << 8) / out_w
    for y in prange(out_h):
        fy = int(y * ry)
        y0 = fy >> 8
        y1 = min(y0 + 1, in_h - 1)
        wy = fy & 255
        for x in range(out_w):
            fx = int(x * rx)
            x0 = fx >> 8
            x1 = min(x0 + 1, in_w - 1)
            wx = fx & 255
            v00 = int(src[y0, x0])
            v01 = int(src[y0, x1])
            v10 = int(src[y1, x0])
            v11 = int(src[y1, x1])
            top = (v00 * (256 - wx) + v01 * wx) >> 8
            bot = (v10 * (256 - wx) + v11 * wx) >> 8
            val = (top * (256 - wy) + bot * wy) >> 8
            val = max(0, min(100, val))
            out[y, x] = np.int8(val)
    return out


def _make_cubic_lut_int(a=-0.5, steps=256):
    lut = np.zeros((steps, 4), dtype=np.int16)
    for i in range(steps):
        t = i / steps
        weights = np.zeros(4, dtype=np.float64)
        for j in range(-1, 3):
            x = j - t
            ax = abs(x)
            if ax < 1.0:
                w = (a+2)*ax*ax*ax - (a+3)*ax*ax + 1
            elif ax < 2.0:
                w = a*ax*ax*ax - 5*a*ax*ax + 8*a*ax - 4*a
            else:
                w = 0.0
            weights[j+1] = w
        s = np.sum(weights)
        if s != 0:
            weights /= s
        iw = np.round(weights * 256).astype(np.int16)
        diff = 256 - np.sum(iw)
        if diff != 0:
            iw[np.argmax(iw)] += diff
        lut[i] = iw
    return lut


CUBIC_LUT_INT = _make_cubic_lut_int()


@njit(parallel=True, cache=True, fastmath=True)
def _bicubic_resize_int(src, out_h, out_w):
    in_h, in_w = src.shape
    out = np.empty((out_h, out_w), dtype=np.int8)
    scale_y = (in_h << 8) / out_h
    scale_x = (in_w << 8) / out_w
    for y in prange(out_h):
        fy = int(y * scale_y)
        y_int = fy >> 8
        y_frac = fy & 255
        wy = CUBIC_LUT_INT[y_frac]
        for x in range(out_w):
            fx = int(x * scale_x)
            x_int = fx >> 8
            x_frac = fx & 255
            wx = CUBIC_LUT_INT[x_frac]
            val = 0
            for m in range(4):
                yy = y_int + (m - 1)
                yy = min(max(yy, 0), in_h - 1)
                wy_m = wy[m]
                for n in range(4):
                    xx = x_int + (n - 1)
                    xx = min(max(xx, 0), in_w - 1)
                    val += int(src[yy, xx]) * wy_m * wx[n]
            val = val >> 16
            val = max(0, min(100, val))
            out[y, x] = np.int8(val)
    return out


def interpolate_rrpcf(config_file, extent, area_name, crops, rrpcf_name='rrpcf'):
    print(' -> Downscaling recurrence rate of potential crop failures...')
    output_dir = os.path.join(config_file['files']['output_dir']+'_downscaled', area_name)
    climate_data_dir = os.path.join(config_file['files']['climate_data_dir'])
    fine_resolution = dt.get_resolution_array(config_file, extent, True)
    land_sea_mask, _ = dt.load_specified_lines(config_file['files']['land_sea_mask'], extent, False)
    land_sea_mask = np.asarray(land_sea_mask).astype(np.float16)
    land_sea_mask[land_sea_mask == 0] = np.nan
    if land_sea_mask.shape[0] != fine_resolution[0]:
        land_sea_mask = dt.interpolate_nanmask(land_sea_mask, fine_resolution)
    nan_mask = np.isnan(land_sea_mask)
    del land_sea_mask

    method = config_file.get('options', {}).get('rrpcf_interpolation_method', 'bilinear').lower()

    for crop in crops:
        s = datetime.now()
        crop = os.path.splitext(crop)[0]
        print(f'    -> {crop.upper()}')
        if os.path.exists(os.path.join(output_dir, f'rrpcf_{crop}.zarr')):
            print(f'       RRPCF data for {crop} already downscaled. Skipping.')
            continue
        current_rrpcf_file = os.path.join(climate_data_dir, f'{rrpcf_name}_{crop}.nc')

        if not os.path.exists(current_rrpcf_file):
            print(f'       RRPCF data for {crop} not existing. Unable to downscale. Skipping.')

        with xr.open_dataset(current_rrpcf_file) as ds:
            y_max, x_min, y_min, x_max = extent
            lat_slice = slice(y_max, y_min) if ds.lat[0] > ds.lat[-1] else slice(y_min, y_max)
            lon_slice = slice(x_min, x_max) if ds.lon[0] < ds.lon[-1] else slice(x_max, x_min)
            ds_clip = ds.sel(lat=lat_slice, lon=lon_slice)
            t_upper, t_lower = ds_clip["t_upper"].values, ds_clip["t_lower"].values
            p_upper, p_lower = ds_clip["p_upper"].values, ds_clip["p_lower"].values
            coarse_resolution = t_upper.shape
            
        if len(coarse_resolution) == 2:
            rf = np.max([t_upper, t_lower, p_upper, p_lower], axis=0)
            ir = np.max([t_upper, t_lower, p_upper], axis=0)
            del t_upper, t_lower, p_upper, p_lower

            rf = np.nan_to_num(rf, nan=-1).astype(np.int8)
            ir = np.nan_to_num(ir, nan=-1).astype(np.int8)

            if (rf < 0).any():
                mask = rf < 0
                rf[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), rf[~mask])
            if (ir < 0).any():
                mask = ir < 0
                ir[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ir[~mask])

            if method == 'bilinear':
                rf =  _bilinear_resize_int(rf, fine_resolution[0], fine_resolution[1])
                ir =  _bilinear_resize_int(ir, fine_resolution[0], fine_resolution[1])
            elif method == 'bicubic':
                rf = _bicubic_resize_int(rf, fine_resolution[0], fine_resolution[1])
                ir = _bicubic_resize_int(ir, fine_resolution[0], fine_resolution[1])

            rf[nan_mask] = -1
            ir[nan_mask] = -1
            chunk_h = 512
            chunk_w = fine_resolution[1]
            store_path = os.path.join(output_dir, f'rrpcf_{crop}.zarr')
            root = zarr.open(store_path, mode="w")
            root.attrs["extent"] = extent
            arrays = {}
            arrays['rf'] = root.create_array('rf', shape=fine_resolution, chunks=(chunk_h, chunk_w), dtype='i1', overwrite=True) #type:ignore
            arrays['ir'] = root.create_array('ir', shape=fine_resolution, chunks=(chunk_h, chunk_w), dtype='i1', overwrite=True) #type:ignore
            for y0 in range(0, fine_resolution[0], chunk_h):
                y1 = min(y0 + chunk_h, fine_resolution[0])
                arrays['rf'][y0:y1, :] = rf[y0:y1, :]
                arrays['ir'][y0:y1, :] = ir[y0:y1, :]
            del rf, ir

        elif len(coarse_resolution) == 3:
            rf = np.maximum.reduce((t_upper, t_lower, p_upper, p_lower))
            ir = np.maximum.reduce((t_upper, t_lower, p_upper))
            rf = np.nan_to_num(rf, nan=-1).astype(np.int8)
            ir = np.nan_to_num(ir, nan=-1).astype(np.int8)
            del t_upper, t_lower, p_upper, p_lower

            bytes_per_thread = 4 * fine_resolution[0] * fine_resolution[1]
            available_ram = psutil.virtual_memory().available
            cpu_cores = int(os.cpu_count() or 1)
            n_threads = int(max(1, min(cpu_cores, available_ram // (bytes_per_thread * 1.2))))
            limit = "RAM" if n_threads < cpu_cores else "CPU-Cores"
            print(f'       -> Multithreading: {n_threads} processes, limited by {limit}.')
            if limit == 'RAM':
                print(f'          Minimum available RAM required to use all {cpu_cores} cores: {(10+(bytes_per_thread/1e9*cpu_cores)):.0f} GB.')
            chunk_h = 512
            chunk_w = fine_resolution[1]
            store_path = os.path.join(output_dir, f'rrpcf_{crop}.zarr')
            root = zarr.group(store_path, overwrite=True)
            root.attrs["extent"] = extent
            arrays = {}
            for name in ['rf', 'ir']:
                arrays[name] = root.create_array(name, shape=(365, *fine_resolution), chunks=(1, chunk_h, chunk_w), dtype='i1', overwrite=True)

            worker = partial(process_rrpcf_day, rf=rf, ir=ir, array_path=store_path, fine_resolution=fine_resolution, nan_mask=nan_mask, chunk_h=chunk_h, method=method)
            
            ### DEBUG
            # for day in range(365):
            #    worker(day)
            
            
            while True:
                try:
                    with ProcessPoolExecutor(max_workers=n_threads) as executor:
                        list(executor.map(worker, range(365)))
                except:
                    os.remove(store_path)
                    root = zarr.group(store_path, overwrite=True)
                    root.attrs["extent"] = extent
                    arrays = {}
                    for name in ['rf', 'ir']:
                        arrays[name] = root.create_array(name, shape=(365, *fine_resolution), chunks=(1, chunk_h, chunk_w), dtype='i1', overwrite=True)
                break
            del rf, ir
        print(f'       Time required: {datetime.now() - s}\n')


def process_rrpcf_day(day, rf, ir, array_path, fine_resolution, nan_mask, chunk_h, method):
    try:
        if (rf < 0).any():
            mask = rf < 0
            rf[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), rf[~mask])
        if (ir < 0).any():
            mask = ir < 0
            ir[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), ir[~mask])
        if method == 'bilinear':
            rf_res = _bilinear_resize_int(rf[day], *fine_resolution)
            ir_res = _bilinear_resize_int(ir[day], *fine_resolution)
        elif method == 'bicubic':
            rf_res = _bicubic_resize_int(rf[day], *fine_resolution)
            ir_res = _bicubic_resize_int(ir[day], *fine_resolution)
        rf_res = np.where(nan_mask, -1, rf_res)
        ir_res = np.where(nan_mask, -1, ir_res)
        arrays = zarr.open(array_path, mode="a")
        for y0 in range(0, fine_resolution[0], chunk_h):
            y1 = min(y0 + chunk_h, fine_resolution[0])
            arrays['rf'][day, y0:y1, :] = rf_res[y0:y1, :] #type:ignore
            arrays['ir'][day, y0:y1, :] = ir_res[y0:y1, :] #type:ignore        
        del rf_res, ir_res
    except:
        process_rrpcf_day(day, rf, ir, array_path, fine_resolution, nan_mask, chunk_h, method)