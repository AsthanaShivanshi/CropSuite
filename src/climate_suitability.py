import os
import numpy as np
import sys
from numba import njit, prange, types
import re
import rasterio
from rasterio.transform import from_bounds
from datetime import datetime
from netCDF4 import Dataset
import gc
import zarr_tools as zt
import shutil
import zarr

suffix = '.npz'
mode = 'npz'


loader = zt.load if mode == 'zarr' else np.load
saver = {'zarr': zt.savez, 'npz': np.savez, 'npz_compressed': np.savez_compressed}.get(mode, np.savez)


def save(filename, *args, **kwargs):
    if mode == 'zarr':
        filename.replace('.npz', '.zarr')
        saver(filename, *args, **kwargs)
    else:
        filename.replace('.zarr', '.npz')
        saver(filename, *args, **kwargs)

def load(filename, *args, **kwargs):
    if filename.endswith('.npz'):
        loader(filename, *args, **kwargs)
    elif filename.endswith('.zarr'):
        loader(filename, *args, **kwargs)


@njit(inline='always')
def lookup_formula(val, vals, suits):
    if val < vals[0] or val > vals[-1]:
        return 0
    left, right = 0, len(vals) - 2
    while left <= right:
        mid = (left + right) // 2
        if vals[mid] <= val < vals[mid+1]:
            t = (val - vals[mid]) / (vals[mid+1] - vals[mid])
            return int((suits[mid] + t * (suits[mid+1] - suits[mid])) * 100)
        elif val < vals[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return int(suits[-1] * 100) if val == vals[-1] else 0

@njit(inline='always')
def cyclic_sum(arr, start, length, limit=-1.0):
    total = 0.0
    L = arr.shape[0]
    end = start + length
    if end <= L:
        for i in range(start, end):
            total += arr[i]
            if limit > 0 and total >= limit:
                return limit
    else:
        part1_len = L - start
        part2_len = end - L
        for i in range(part1_len):
            total += arr[start + i]
            if limit > 0 and total >= limit:
                return limit
        for i in range(part2_len):
            total += arr[i]
            if limit > 0 and total >= limit:
                return limit
    if limit > 0 and total > limit:
        return limit
    return total

@njit(inline='always')
def has_consecutive_in_window(arr, start, window_len, consec_len, threshold, below=True):
    L = arr.shape[0]
    count = 0
    end = start + window_len
    if end <= L:
        for i in range(start, end):
            val = arr[i]
            if (below and val < threshold) or (not below and val > threshold):
                count += 1
                if count >= consec_len:
                    return True
            else:
                count = 0
    else:
        for i in range(start, L):
            val = arr[i]
            if (below and val < threshold) or (not below and val > threshold):
                count += 1
                if count >= consec_len:
                    return True
            else:
                count = 0
        for i in range(end - L):
            val = arr[i]
            if (below and val < threshold) or (not below and val > threshold):
                count += 1
                if count >= consec_len:
                    return True
            else:
                count = 0
    return False


@njit(parallel=True)
def process_precipitation(prec, growing_cycle, prec_vals, prec_suit, prec_after_sowing, time_after_sowing, lethals, lethal_min_prec,
                          lethal_min_prec_duration, lethal_max_prec, lethal_max_prec_duration):
    L, Y, X = prec.shape
    result_prec = np.empty((L, Y, X), dtype=np.int8)
    if growing_cycle >= 365:
        d = 0
        for y in prange(Y):
            for x in range(X):
                if prec[d, y, x] < -10000:
                    result_prec[0, y, x] = -1
                    continue
                col = prec[:, y, x]
                col_ext = np.empty(L * 2, dtype=col.dtype)
                col_ext[:L] = col
                col_ext[L:] = col
                sum_prec = min(np.sum(col, dtype=np.int32), 60000.0)
                sum_prec_sow = np.sum(col[:time_after_sowing])
                val = lookup_formula(sum_prec, prec_vals, prec_suit)
                if growing_cycle < 365:
                    val = val if sum_prec_sow >= prec_after_sowing else 0
                if lethals == 1:
                    if has_consecutive_in_window(col_ext, d, growing_cycle, lethal_min_prec_duration, lethal_min_prec, below=True) \
                    or has_consecutive_in_window(col_ext, d, growing_cycle, lethal_max_prec_duration, lethal_max_prec, below=False):
                        val = 0
                result_prec[0, y, x] = val
        return result_prec
    else:
        for d in prange(L):
            for y in range(Y):
                for x in range(X):
                    if prec[d, y, x] < -10000:
                        result_prec[d, y, x] = -1
                        continue
                    col = prec[:, y, x]
                    sum_prec = cyclic_sum(col, d, growing_cycle, limit=60000.0)
                    val = lookup_formula(sum_prec, prec_vals, prec_suit)
                    if val > 0:
                        sum_prec_sow = cyclic_sum(col, d, time_after_sowing)
                        if sum_prec_sow <= prec_after_sowing:
                            val = 0.0
                    if lethals == 1 and val > 0:
                        if has_consecutive_in_window(col, d, growing_cycle, lethal_min_prec_duration, lethal_min_prec, below=True) \
                        or has_consecutive_in_window(col, d, growing_cycle, lethal_max_prec_duration, lethal_max_prec, below=False):
                            val = 0
                    result_prec[d, y, x] = val
    return result_prec

@njit(parallel=True)
def process_precipitation_phenology(prec, growing_cycle, prec_vals, prec_suit, prec_after_sowing, time_after_sowing, lethals, lethal_min_prec,
                          lethal_min_prec_duration, lethal_max_prec, lethal_max_prec_duration, phenology_params):
    L, Y, X = prec.shape
    result_prec = np.empty((L, Y, X), dtype=np.int8)
    if growing_cycle == 365:
        d = 0
        for y in prange(Y):
            for x in range(X):
                if prec[d, y, x] < -10000:
                    result_prec[0, y, x] = -1
                    continue
                col = prec[:, y, x]
                col_ext = np.empty(L * 2, dtype=col.dtype)
                col_ext[:L] = col
                col_ext[L:] = col
                sum_prec = min(np.sum(col), 60000.0)
                sum_prec_sow = np.sum(col[:time_after_sowing])
                val = lookup_formula(sum_prec, prec_vals, prec_suit)
                if growing_cycle >= 365:
                    val = val if sum_prec_sow >= prec_after_sowing else 0
                if lethals == 1:
                    if has_consecutive_in_window(col_ext, d, growing_cycle, lethal_min_prec_duration, lethal_min_prec, below=True) \
                    or has_consecutive_in_window(col_ext, d, growing_cycle, lethal_max_prec_duration, lethal_max_prec, below=False):
                        val = 0
                result_prec[0, y, x] = val
        return result_prec
    else:
        for d in prange(L):
            for y in range(Y):
                for x in range(X):
                    if prec[d, y, x] < -10000:
                        result_prec[d, y, x] = -1
                        continue
                    col = prec[:, y, x]
                    if len(phenology_params) > 0 and phenology_params[0][0] != -1:
                        val = 100.0
                        for i in range(len(phenology_params)):
                            if val <= 0.0:
                                break
                            start, end, lower, upper, smooth, xp, fp = phenology_params[i]
                            win_len = end - start
                            if win_len <= 0:
                                continue
                            sp = cyclic_sum(col, d+start, win_len)
                            if xp.shape[0] > 0:
                                x0 = xp[0]
                                x1 = xp[xp.shape[0] - 1]
                                if sp < x0:
                                    spc = x0
                                elif sp > x1:
                                    spc = x1
                                else:
                                    spc = sp
                                v = 0.0
                                if sp > 0.0:
                                    v = lookup_formula(spc, xp, fp)
                            else:
                                v = smooth_curve(sp, lower, upper, smooth)
                            if v < val:
                                val = v
                    else:
                        sum_prec = cyclic_sum(prec[:, y, x], d, growing_cycle, limit=60000.0)
                        val = lookup_formula(sum_prec, prec_vals, prec_suit)
                    sum_prec_sow = cyclic_sum(prec[:, y, x], d, time_after_sowing)
                    if sum_prec_sow <= prec_after_sowing:
                        val = 0.0
                    if lethals == 1 and val > 0:
                        if has_consecutive_in_window(prec[:, y, x], d, growing_cycle, lethal_min_prec_duration, lethal_min_prec, below=True) \
                        or has_consecutive_in_window(prec[:, y, x], d, growing_cycle, lethal_max_prec_duration, lethal_max_prec, below=False):
                            val = 0
                    result_prec[d, y, x] = val
    return result_prec


@njit(fastmath=True)
def smooth_curve(x, lower, upper, smoothness):
    half = smoothness / 2.0
    k = 2.0 * np.log(99.0) / smoothness
    if x <= lower - half:
        rise = 0.0
    elif x >= lower + half:
        rise = 100.0
    else:
        rise = 100.0 / (1.0 + np.exp(-k * (x - lower)))
    if x <= upper - half:
        fall = 100.0
    elif x >= upper + half:
        fall = 0.0
    else:
        fall = 100.0 / (1.0 + np.exp(k * (x - upper)))
    val = rise if rise < fall else fall
    return np.int8(val)


def preprocess_phenology_params(raw_params):
    processed = []
    for param in raw_params:
        start = int(param[1])
        end = int(param[2])
        lower = float(param[3]) * 10
        upper = float(param[4]) * 10
        smoothness = float(param[5]) * 10

        if param[6] != '[]':
            s = " ".join(param[6:])
            xp, fp = zip(*[tuple(map(float, p.split())) 
                           for p in re.findall(r'\(([^)]+)\)', s)])
            xp = np.array(xp) * 10
            fp = np.array(fp)
        else:
            xp, fp = np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
        processed.append((start, end, lower, upper, smoothness, xp, fp/100))
    if len(processed) == 0:
        return []
    return processed


@njit(parallel=True, fastmath=True)
def process_temperature(temp, growing_cycle, temp_vals, temp_suit, lethals, lethal_min_temp, lethal_min_temp_duration, lethal_max_temp, lethal_max_temp_duration):
    L, Y, X = temp.shape
    result_temp = np.empty((L, Y, X), dtype=np.int8)
    if growing_cycle == 365:
        d = 0
        for y in prange(Y):
            for x in range(X):
                if temp[d, y, x] < -10000:
                    result_temp[0, y, x] = -1
                    continue
                mean_temp = cyclic_sum(temp[:, y, x], d, growing_cycle) / growing_cycle
                val = lookup_formula(mean_temp, temp_vals, temp_suit) if mean_temp > 0 else 0
                if lethals == 1:
                    if (has_consecutive_in_window(temp[:, y, x], d, growing_cycle, lethal_min_temp_duration, lethal_min_temp, below=True)
                        or has_consecutive_in_window(temp[:, y, x], d, growing_cycle, lethal_max_temp_duration, lethal_max_temp, below=False)):
                        val = 0
                result_temp[0, y, x] = np.int8(val)
        return result_temp
    for d in prange(L):
        for y in range(Y):
            for x in range(X):
                if temp[d, y, x] < -10000:
                    result_temp[d, y, x] = -1
                    continue
                col = temp[:, y, x]
                mt = cyclic_sum(col, d, growing_cycle) / growing_cycle
                val = 0.0
                if mt > 0.0:
                    val = lookup_formula(mt, temp_vals, temp_suit)
                if lethals == 1 and val > 0.0:
                    if (has_consecutive_in_window(col, d, growing_cycle, lethal_min_temp_duration, lethal_min_temp, below=True)
                        or has_consecutive_in_window(col, d, growing_cycle, lethal_max_temp_duration, lethal_max_temp, below=False)):
                        val = 0.0
                result_temp[d, y, x] = np.int8(val)
    return result_temp

@njit(parallel=True, fastmath=True)
def process_temperature_phenology(temp, growing_cycle, temp_vals, temp_suit, lethals, lethal_min_temp, lethal_min_temp_duration, lethal_max_temp, lethal_max_temp_duration, phenology_params):
    L, Y, X = temp.shape
    result_temp = np.empty((L, Y, X), dtype=np.int8)
    if growing_cycle == 365:
        d = 0
        for y in prange(Y):
            for x in range(X):
                if temp[d, y, x] < -10000:
                    result_temp[0, y, x] = -1
                    continue
                mean_temp = cyclic_sum(temp[:, y, x], d, growing_cycle) / growing_cycle
                val = lookup_formula(mean_temp, temp_vals, temp_suit) if mean_temp > 0 else 0
                if lethals == 1:
                    if (has_consecutive_in_window(temp[:, y, x], d, growing_cycle, lethal_min_temp_duration, lethal_min_temp, below=True)
                        or has_consecutive_in_window(temp[:, y, x], d, growing_cycle, lethal_max_temp_duration, lethal_max_temp, below=False)):
                        val = 0
                result_temp[0, y, x] = np.int8(val)
        return result_temp
    for d in prange(L):
        for y in range(Y):
            for x in range(X):
                if temp[d, y, x] < -10000:
                    result_temp[d, y, x] = -1
                    continue
                col = temp[:, y, x]

                val = 100.0
                for i in range(len(phenology_params)):
                    if val <= 0.0:
                        break
                    start, end, lower, upper, smooth, xp, fp = phenology_params[i]
                    win_len = end - start
                    if win_len <= 0:
                        continue
                    mt = cyclic_sum(col, d+start, win_len) / win_len
                    if xp.shape[0] > 0:
                        x0 = xp[0]
                        x1 = xp[xp.shape[0] - 1]
                        if mt < x0:
                            mtc = x0
                        elif mt > x1:
                            mtc = x1
                        else:
                            mtc = mt
                        v = 0.0
                        if mt > 0.0:
                            v = lookup_formula(mtc, xp, fp)
                    else:
                        v = smooth_curve(mt, lower, upper, smooth)
                    if v < val:
                        val = v
                    else:
                        mt = cyclic_sum(col, d, growing_cycle) / growing_cycle
                        val = 0.0
                        if mt > 0.0:
                            val = lookup_formula(mt, temp_vals, temp_suit)

                if lethals == 1 and val > 0.0:
                    if (has_consecutive_in_window(col, d, growing_cycle, lethal_min_temp_duration, lethal_min_temp, below=True)
                        or has_consecutive_in_window(col, d, growing_cycle, lethal_max_temp_duration, lethal_max_temp, below=False)):
                        val = 0.0
                result_temp[d, y, x] = np.int8(val)
    return result_temp


@njit(parallel=True)
def apply_lookup(data, vals, suits):
    Z, Y, X = data.shape
    result = np.empty((Z, Y, X), dtype=np.int8)
    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                result[z, y, x] = lookup_formula(data[z, y, x], vals, suits)
    return result


@njit(parallel=True)
def calculate_wintercrop_precipitation(prec, date_after, growing_cycle, prec_vals, prec_suit, prec_after_sowing, time_after_sowing, days_to_vern):
    L, Y, X = prec.shape
    result_prec = np.empty((L, Y, X), dtype=np.int8)
    for d in prange(L):
        for y in range(Y):
            for x in range(X):
                if prec[d, y, x] < -10000:
                    result_prec[d, y, x] = -1
                    continue
                col_prec = prec[:, y, x]
                prec_to_vern = cyclic_sum(col_prec, d, days_to_vern, limit=60000.0)
                if prec_to_vern <= 0:
                    result_prec[d, y, x] = 0
                    date_after[d, y, x] = -1
                    continue
                suit_to_vern = lookup_formula(prec_to_vern, prec_vals, prec_suit)
                after_idx = date_after[d, y, x]
                prec_after_vern = cyclic_sum(col_prec, after_idx, growing_cycle - days_to_vern, limit=60000.0)
                suit_after_vern = lookup_formula(prec_after_vern, prec_vals, prec_suit)
                sum_prec_sow = cyclic_sum(col_prec, d, time_after_sowing)
                if sum_prec_sow >= prec_after_sowing:
                    suit = suit_to_vern if suit_to_vern < suit_after_vern else suit_after_vern
                else:
                    suit = 0
                result_prec[d, y, x] = suit
                if suit == 0:
                    date_after[d, y, x] = -1
    return result_prec, date_after


@njit(parallel=True)
def calculate_wintercrop_temperature(temp, growing_cycle, temp_vals, temp_suit, effect_days, vern_tmax, vern_tmin, frost_temp, frost_dur, days_to_vern):
    L, Y, X = temp.shape
    result_temp = np.empty((L, Y, X), dtype=np.int8)
    after_vern = np.empty((L, Y, X), dtype=np.int16)
    for d in prange(L):
        for y in range(Y):
            for x in range(X):
                if temp[d, y, x] < -10000:
                    result_temp[d, y, x] = -1
                    after_vern[d, y, x] = -1
                    continue
                s = 0
                for k in range(days_to_vern):
                    s += temp[(d + k) % L, y, x]
                mean_temp_to_vern = int(s / days_to_vern)
                suit_to_vern = 0
                if mean_temp_to_vern > 0:
                    suit_to_vern = lookup_formula(mean_temp_to_vern, temp_vals, temp_suit)
                if suit_to_vern == 0:
                    result_temp[d, y, x] = 0
                    after_vern[d, y, x] = -1
                    continue
                gc_after_vern = growing_cycle - days_to_vern
                count_vern = 0
                count_frost = 0
                broke_early = False
                for day in range(gc_after_vern):
                    current_temp = temp[(d + days_to_vern + day) % L, y, x]
                    if vern_tmin <= current_temp <= vern_tmax and count_vern < effect_days:
                        count_vern += 1
                        if current_temp < frost_temp:
                            count_frost += 1
                            if count_frost >= frost_dur:
                                result_temp[d, y, x] = 0
                                after_vern[d, y, x] = -1
                                broke_early = True
                                break
                    elif count_vern >= effect_days:
                        break
                if broke_early:
                    continue
                if count_vern < effect_days:
                    result_temp[d, y, x] = 0
                    after_vern[d, y, x] = -1
                    continue
                if count_frost >= frost_dur:
                    result_temp[d, y, x] = 0
                    after_vern[d, y, x] = -1
                    continue
                start_after_vern = (d + days_to_vern + day) % L
                s2 = 0
                for k in range(gc_after_vern):
                    s2 += temp[(start_after_vern + k) % L, y, x]
                mean_temp_after_vern = int(s2 / gc_after_vern)
                suit_after_vern = 0
                if mean_temp_after_vern > 0:
                    suit_after_vern = lookup_formula(mean_temp_after_vern, temp_vals, temp_suit)
                result_temp[d, y, x] = min(suit_to_vern, suit_after_vern)
                after_vern[d, y, x] = start_after_vern
    return result_temp, after_vern


def load_array(file_path, key=None, extent=None, out=None):
    with loader(file_path) as npz:
        if key is None:
            try:
                key = [k for k in npz.files() if k != 'extent'][0]
            except:
                key = [k for k in npz.files if k != 'extent'][0] #type:ignore
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'File does not exist: {file_path}')
        try:
            data = npz[key]
            if extent is not None:
                y_max, x_min, y_min, x_max = extent
                full_ymax, full_xmin, full_ymin, full_xmax = npz['extent']
                n_y, n_x = data.shape[-2], data.shape[-1]
                res_x = (full_xmax - full_xmin) / n_x
                res_y = (full_ymax - full_ymin) / n_y
                x0 = int((x_min - full_xmin) / res_x)
                x1 = int((x_max - full_xmin) / res_x)
                y0 = int((full_ymax - y_max) / res_y)
                y1 = int((full_ymax - y_min) / res_y)
                if data.ndim == 2:
                    data = data[y0:y1, x0:x1]
                elif data.ndim == 3:
                    data = data[:, y0:y1, x0:x1]
            if out is not None:
                np.copyto(out, data)
                return out
            else:
                return data
        except Exception as e:
            print(f'Error loading file {file_path}: {e}')
            return None

def temperature(temp_data, growing_cycle, plant_params, crop, temp_folder, extent):
    temp_vals = np.array(plant_params[crop]['temp_vals'], dtype=np.float64) * 10
    temp_suit = np.array(plant_params[crop]['temp_suit'], dtype=np.float64)
    lethals = int(plant_params[crop].get('lethal_thresholds', [0])[0])
    lethal_min_temp = int(plant_params[crop].get('lethal_min_temp', [0])[0]) * 10
    lethal_min_temp_duration = int(plant_params[crop].get('lethal_min_temp_duration', [3])[0])
    lethal_max_temp = int(plant_params[crop].get('lethal_max_temp', [40])[0]) * 10
    lethal_max_temp_duration = int(plant_params[crop].get('lethal_max_temp_duration', [5])[0])
    additional_conditions_temperature, _ = get_additional_conditions(plant_params[crop])

    wintercrop = 1 if str(plant_params[crop].get('wintercrop', [0])[0]).lower() in ('y', '1') else 0
    if wintercrop:
        effect_days = int(plant_params[crop].get('vernalization_effective_days', [50])[0])
        vern_tmax = int(plant_params[crop].get('vernalization_tmax', [8])[0]) * 10
        vern_tmin = int(plant_params[crop].get('vernalization_tmin', [0])[0]) * 10
        frost_temp = int(plant_params[crop].get('frost_resistance', [-20])[0]) * 10
        frost_dur = int(plant_params[crop].get('frost_resistance_days', [3])[0])
        days_to_vern = int(plant_params[crop].get('days_to_vernalization', [100])[0])

    phenology_params = []
    for key, values in plant_params[crop].items():
        if key.startswith('phen') and key.endswith('_temp'):
            _, range_str, var_type = key.split('_')
            start, end = map(int, range_str.split('-'))
            phenology_params.append([var_type, start, end] + values)   

    temp_file = os.path.join(temp_folder, f'temp_{crop}{suffix}')
    if not os.path.exists(temp_file):
        print(' => Processing temperature suitability')
        print('    → 0 % complete')
        if wintercrop:
            temp, date_after = calculate_wintercrop_temperature(temp_data, growing_cycle, temp_vals, temp_suit, effect_days, vern_tmax, vern_tmin, frost_temp, frost_dur, days_to_vern)
            #np.savez_compressed(os.path.join(temp_folder, f'temp_sowaftvern_{crop}{suffix}'), sowing=date_after, extent=extent)
            save(os.path.join(temp_folder, f'temp_sowaftvern_{crop}{suffix}'), sowing=date_after, extent=extent)
            del date_after
        else:
            phenology_params = preprocess_phenology_params(phenology_params)
            if len(phenology_params) > 0:
                print('      Phenology definitions available - Overriding membership function')
                temp = process_temperature_phenology(temp_data, growing_cycle, temp_vals, temp_suit, lethals, lethal_min_temp, lethal_min_temp_duration, lethal_max_temp, lethal_max_temp_duration, phenology_params)
            else:
                temp = process_temperature(temp_data, growing_cycle, temp_vals, temp_suit, lethals, lethal_min_temp, lethal_min_temp_duration, lethal_max_temp, lethal_max_temp_duration)
        print(f'    → 100 % complete')
        if growing_cycle == 365:
            temp = temp[0]
        else:
            for idx, addcon in enumerate(additional_conditions_temperature):
                print(f'    → Calculation of additional temperature condition #{idx+1}')
                mask = evaluate_additional_condition(temp_data, 0, int(addcon[1]), int(addcon[2]), {'>':0, '>=':1, '<':2, '<=':3}[addcon[3]], int(int(addcon[4])*10))
                temp[mask] = 0
                del mask
        print('    → Writing file...\n')
        #np.savez_compressed(temp_file, temp=temp, extent=extent)
        save(temp_file, temp=temp, extent=extent)
        del temp

    
def precipitation(prec_data, growing_cycle, plant_params, crop, temp_folder, extent):
    prec_vals = np.array(plant_params[crop]['prec_vals'], dtype=np.float64) * 10
    prec_suit = np.array(plant_params[crop]['prec_suit'], dtype=np.float64)

    prec_after_sowing = int(plant_params[crop].get('prec_req_after_sow', [20])[0]) * 10
    time_after_sowing = int(plant_params[crop].get('prec_req_days', [14])[0])
    lethals = int(plant_params[crop].get('lethal_thresholds', [0])[0])
    lethal_min_prec = int(plant_params[crop].get('lethal_min_prec', [1])[0]) * 10
    lethal_min_prec_duration = int(plant_params[crop].get('lethal_min_prec_duration', [21])[0])
    lethal_max_prec = int(plant_params[crop].get('lethal_max_prec', [100])[0]) * 10
    lethal_max_prec_duration = int(plant_params[crop].get('lethal_max_prec_duration', [3])[0])
    _, additional_conditions_precipitation = get_additional_conditions(plant_params[crop])

    phenology_params = []
    for key, values in plant_params[crop].items():
        if key.startswith('phen') and key.endswith('_prec'):
            _, range_str, var_type = key.split('_')
            start, end = map(int, range_str.split('-'))
            phenology_params.append([var_type, start, end] + values)   

    prec_file = os.path.join(temp_folder, f'prec_{crop}{suffix}')
    if not os.path.exists(prec_file):
        print(' => Processing precipitation suitability')
        print('    → 0 % complete')
        wintercrop = 1 if str(plant_params[crop].get('wintercrop', [0])[0]).lower() in ('y', '1') else 0
        if wintercrop:
            days_to_vern = int(plant_params[crop].get('days_to_vernalization', [100])[0])
            date_after = loader(os.path.join(temp_folder, f'temp_sowaftvern_{crop}{suffix}'))['sowing']
            prec, date_after = calculate_wintercrop_precipitation(prec_data, date_after, growing_cycle, prec_vals, prec_suit, prec_after_sowing, time_after_sowing, days_to_vern)
            save(os.path.join(temp_folder, f'temp_sowaftvern_{crop}{suffix}'), sowing=date_after, extent=extent)
            del date_after
        else:
            phenology_params = preprocess_phenology_params(phenology_params)
            if len(phenology_params) > 0:
                print('      Phenology definitions available - Overriding membership function')
                prec = process_precipitation_phenology(prec_data, growing_cycle, prec_vals, prec_suit, prec_after_sowing, time_after_sowing,
                                            lethals, lethal_min_prec, lethal_min_prec_duration, lethal_max_prec, lethal_max_prec_duration, phenology_params)
            else:
                prec = process_precipitation(prec_data, growing_cycle, prec_vals, prec_suit, prec_after_sowing, time_after_sowing,
                                            lethals, lethal_min_prec, lethal_min_prec_duration, lethal_max_prec, lethal_max_prec_duration)
        print(f'    → 100 % complete')
        if growing_cycle == 365:
            prec = prec[0]
        else:
            for idx, addcon in enumerate(additional_conditions_precipitation):
                print(f'    → Calculation of additional precipitation condition #{idx+1}')
                mask = evaluate_additional_condition(prec_data, 1, int(addcon[1]), int(addcon[2]), {'>':0, '>=':1, '<':2, '<=':3}[addcon[3]], int(int(addcon[4])*10))
                prec[mask] = 0
                del mask
        print('    → Writing file...\n')
        save(prec_file, prec=prec, extent=extent)
        del prec


def consider_rrpcf(config_dict, plant_params, crop, temp_folder, extent, irrigation='rf', full_name=''):
    rrpcf_vals = np.array(plant_params[crop]['freqcropfail_vals'], dtype=np.float64)
    rrpcf_suit = np.array(plant_params[crop]['freqcropfail_suit'], dtype=np.float64)

    if irrigation == 'rf':
        rrpcf_file = os.path.join(temp_folder, f'rrpcf_{crop}_rf{suffix}')
    elif irrigation == 'ir':
        rrpcf_file = os.path.join(temp_folder, f'rrpcf_{crop}_ir{suffix}')
    else:
        rrpcf_file = os.path.join(temp_folder, f'rrpcf_{crop}_rf{suffix}')
        rrpcf_file_ir = os.path.join(temp_folder, f'rrpcf_{crop}_ir{suffix}')

    if not os.path.exists(rrpcf_file) or (irrigation == 'rfir' and not os.path.exists(rrpcf_file_ir)):
        print('Loading rrpcf data...')
        source_path = os.path.join(config_dict.get('files', {}).get('output_dir')+'_downscaled', full_name, f'rrpcf_{crop}.zarr')
        if irrigation in ['rf', 'ir']:
            rrpcf = read_climate_data(source_path, extent, parameter=irrigation)
        else:
            rrpcf = read_climate_data(source_path, extent, parameter='rf')
            rrpcf_ir = read_climate_data(source_path, extent, parameter='ir')
            if rrpcf_ir.ndim == 2:
                rrpcf_ir = np.expand_dims(rrpcf_ir, axis=0)
        if rrpcf.ndim == 2:
            rrpcf = np.expand_dims(rrpcf, axis=0)

        print(f'  → rrpcf data loaded with shape {rrpcf.shape}')
        print(f'    rrpcf data loaded as {rrpcf.dtype}')
        print(f'    rrpcf data requires {np.ceil(rrpcf.shape[0] * rrpcf.shape[1] * rrpcf.shape[2] * rrpcf.dtype.itemsize / 1e8) / 10} GB memory\n')
        print('\n => Processing recurrence rate of potential crop failures')

        if irrigation == 'rfir':
            rrpcf = apply_lookup(rrpcf, rrpcf_vals*100, rrpcf_suit)
            rrpcf_ir = apply_lookup(rrpcf_ir, rrpcf_vals*100, rrpcf_suit)
        else:
            rrpcf = apply_lookup(rrpcf, rrpcf_vals*100, rrpcf_suit)
        print('  → Writing file...')
        if irrigation == 'rfir':
            save(rrpcf_file, rrpcf=rrpcf, extent=extent)
            save(rrpcf_file_ir, rrpcf=rrpcf_ir, extent=extent)
        else:
            save(rrpcf_file, rrpcf=rrpcf, extent=extent)
        del rrpcf
        if irrigation == 'rfir':
            del rrpcf_ir


@njit(cache=True)
def day_length_hours(lat_deg, day_of_year):
    lat = np.radians(lat_deg)
    decl = 23.44 * np.pi / 180 * np.sin(2 * np.pi * (day_of_year - 81) / 365.0)
    cos_omega = -np.tan(lat) * np.tan(decl)
    if cos_omega < -1.0:
        cos_omega = -1.0
    elif cos_omega > 1.0:
        cos_omega = 1.0
    omega = np.arccos(cos_omega)
    return 24.0 / np.pi * omega


@njit(parallel=True, cache=True)
def build_daylength_table(lats):
    Y = lats.size
    table = np.empty((365, Y), dtype=np.float32)
    for d in prange(365):
        day_of_year = d + 1
        for y in range(Y):
            table[d, y] = day_length_hours(lats[y], day_of_year)
    return table


@njit(parallel=True, cache=True)
def photoperiod_core(daylength_table, growing_cycle, min_sun, max_sun, D, Y, X):
    result = np.zeros((D, Y, X), dtype=np.int8)
    if growing_cycle == 365:
        avg_sun = np.zeros(Y, dtype=np.float32)
        for y in prange(Y):
            s = 0.0
            for d in range(365):
                s += daylength_table[d, y]
            avg_sun[y] = s / 365.0

        for y in prange(Y):
            cond = (avg_sun[y] >= min_sun) and (avg_sun[y] <= max_sun)
            val = 100 if cond else 0
            for x in range(X):
                result[0, y, x] = val
        return result
    sums = np.empty((D, Y), dtype=np.float32)
    for y in prange(Y):
        vals = daylength_table[:, y]
        extended = np.empty(730, dtype=np.float32)  # 365 * 2
        for i in range(365):
            extended[i] = vals[i]
            extended[i + 365] = vals[i]
        csum = np.zeros(extended.size + 1, dtype=np.float32)
        for i in range(extended.size):
            csum[i+1] = csum[i] + extended[i]
        for d in range(D):
            sums[d, y] = (csum[d + growing_cycle] - csum[d]) / growing_cycle
    for d in prange(D):
        for y in range(Y):
            cond = (sums[d, y] >= min_sun) and (sums[d, y] <= max_sun)
            val = 100 if cond else 0
            for x in range(X):
                result[d, y, x] = val
    return result

def photoperiod(plant_params, crop, growing_cycle, temp_folder, extent, final_shape):
    photoperiod_file = os.path.join(temp_folder, f'photo_{crop}{suffix}')
    if not os.path.exists(photoperiod_file):
        print(' => Processing photoperiodic sensitivity')
        print('    → 0 % complete')
        min_sun = int(plant_params.get(crop, {}).get('minimum_sunlight_hours', [8])[0])
        max_sun = int(plant_params.get(crop, {}).get('maximum_sunlight_hours', [16])[0])
        D, Y, X = final_shape
        top, _, bottom, _ = extent
        lat_res = (top - bottom) / Y
        lats = top - (np.arange(Y) + 0.5) * lat_res
        daylength_table = build_daylength_table(lats)
        result = photoperiod_core(daylength_table, growing_cycle, min_sun, max_sun, D, Y, X)
        print('    → 100 % complete')
        print('  → Writing file...\n')
        save(photoperiod_file, photoperiod=result if growing_cycle < 365 else result[0], extent=extent)


@njit(parallel=True)
def invert_values(arr):
    ny, nx, nz = arr.shape
    result = np.empty_like(arr)
    for i in prange(ny):
        for j in range(nx):
            for k in range(nz):
                result[i, j, k] = 100 - arr[i, j, k]
    return result


def get_additional_conditions(plant_params):
    additional_conditions = [cond for i in range(100) if (cond := plant_params.get(f'AddCon:{i}')) is not None]
    try:
        for adcon in additional_conditions:
            if len(adcon) > 5 and int(adcon[5]) == 0:
                additional_conditions.remove(adcon)
    except:
        additional_conditions = []
    return [cond for cond in additional_conditions if "Temperature" in cond], [cond for cond in additional_conditions if "Precipitation" in cond]


@njit(inline='always')
def compare(val, op_code, threshold):
    # Important: Operators must be reversed!
    if op_code == 0:  # >
        return val <= threshold
    elif op_code == 1:  # >=
        return val < threshold
    elif op_code == 2:  # <
        return val >= threshold
    elif op_code == 3:  # <=
        return val > threshold
    return False


@njit(parallel=True)
def evaluate_additional_condition(data, mode, start, end, op_code, value):
    days, Y, X = data.shape
    window_len = end - start + 1
    result = np.zeros((days, Y, X), dtype=np.bool_)
    for y in prange(Y):
        for x in range(X):
            window_sum = 0
            for i in range(start, end + 1):
                idx = i % days
                window_sum += data[idx, y, x]
            for d in range(days):
                if data[d, y, x] < -10000:
                    result[d, y, x] = False
                else:
                    val = window_sum / window_len if mode == 0 else window_sum
                    result[d, y, x] = compare(val, op_code, value)
                out_idx = (d + start) % days
                in_idx = (d + end + 1) % days
                window_sum = window_sum - data[out_idx, y, x] + data[in_idx, y, x]
    return result


@njit(parallel=True)
def get_emergence(sowing_date, date_after):
    Y, X = sowing_date.shape
    emergences = np.full((Y, X), dtype=np.int16, fill_value=-1)
    for y in prange(Y):
        for x in range(X):
            if sowing_date[y, x] < 0:
                continue
            emergences[y, x] = date_after[y, x]
    return emergences


@njit(parallel=True)
def find_multiple_cropping(data, period_length, threshold=25):
    days, ny, nx = data.shape
    count = np.zeros((ny, nx), dtype=np.int8)
    first_start = np.zeros((ny, nx), dtype=np.int16)
    second_start = np.zeros((ny, nx), dtype=np.int16)
    third_start = np.zeros((ny, nx), dtype=np.int16)
    for y in prange(ny):
        for x in range(nx):
            if np.all(data[:, y, x] == -1):
                count[y, x] = -1
                first_start[y, x] = -1
                second_start[y, x] = -1
                third_start[y, x] = -1
                continue

            max_val = np.max(data[:, y, x])
            starts_found = 0
            start_days = np.zeros(3, dtype=np.int16)

            used_days = np.zeros(days, dtype=np.bool_)

            day = 0
            while day < days and starts_found < 3:
                if not used_days[day] and data[day, y, x] >= threshold:
                    for k in range(period_length):
                        used_days[(day + k) % days] = True
                    start_days[starts_found] = day
                    starts_found += 1
                    day += period_length
                    continue
                day += 1

            sum_first_days = 0
            for i in range(starts_found):
                sum_first_days += data[start_days[i], y, x]

            if starts_found > 0 and sum_first_days > max_val:
                count[y, x] = starts_found
                if starts_found > 0:
                    first_start[y, x] = start_days[0]
                if starts_found > 1:
                    second_start[y, x] = start_days[1]
                if starts_found > 2:
                    third_start[y, x] = start_days[2]
            else:
                if np.any(data[:, y, x] > 0):
                    count[y, x] = 1
                    first_start[y, x] = np.argmax(data[:, y, x])
                else:
                    count[y, x] = 0
    return count, first_start, second_start, third_start


def precompile():
    ### PRE-COMPILING ###
    print('Pre-Compiling required functions...')
    s = datetime.now()
    print(f'    → 0 % complete')
    process_temperature_phenology.compile((types.Array(types.int16, 3, "C"), types.int64, types.Array(types.float64, 1, "C"), types.Array(types.float64, 1, "C"),
                                 types.int64, types.float64, types.int64, types.float64, types.int64,
                                 types.ListType(types.Tuple((types.int64, types.int64, types.float64, types.float64, types.float64,
                                                             types.Array(types.float64, 1, "C"), types.Array(types.float64, 1, "C"))))))
    process_temperature.compile((types.Array(types.int16, 3, "C"), types.int64, types.Array(types.float64, 1, "C"), types.Array(types.float64, 1, "C"),
                                 types.int64, types.float64, types.int64, types.float64, types.int64))
    print(f'    → 20 % complete')
    #process_precipitation_phenology.compile((types.Array(types.int16, 3, "C"), types.int64, types.Array(types.float64, 1, "C"), types.Array(types.float64, 1, "C"),
    #                               types.Array(types.float64, 1, "C"), types.Array(types.float64, 1, "C"), types.int64, types.float64, types.int64, types.float64,
    #                               types.int64, types.ListType(types.Tuple((types.int64, types.int64, types.float64, types.float64, types.float64,
    #                                                                        types.Array(types.float64, 1, "C"), types.Array(types.float64, 1, "C"))))))
    #process_precipitation.compile((types.Array(types.int16, 3, "C"), types.int64, types.Array(types.float64, 1, "C"), types.Array(types.float64, 1, "C"),
    #                               types.Array(types.float64, 1, "C"), types.Array(types.float64, 1, "C"), types.int64, types.float64, types.int64, types.float64,
    #                               types.int64))
    print(f'    → 40 % complete')

    evaluate_additional_condition.compile((types.Array(types.int16, 3, "C"), types.int64, types.int64, types.int64, types.int64, types.int64))
    print(f'    → 60 % complete')

    find_optimal_sowing_date.compile((types.Array(types.int16, 3, "C"),))
    print(f'    → 80 % complete')
    
    invert_values.compile((types.Array(types.int16, 3, "C"),))
    calculate_wintercrop_temperature.compile((types.Array(types.float64, 3, "C"), types.int64, types.Array(types.float64, 1, "C"), types.Array(types.float64, 1, "C"), types.int64, types.float64, types.float64, types.float64, types.int64, types.int64))
    calculate_wintercrop_precipitation.compile((types.Array(types.float64, 3, "C"), types.Array(types.int64, 3, "C"), types.int64, types.Array(types.float64, 1, "C"), types.Array(types.float64, 1, "C"), types.float64, types.int64, types.int64 ))
    apply_lookup.compile((types.Array(types.float64, 3, "C"), types.Array(types.float64, 1, "C"), types.Array(types.int8, 1, "C") ))
    get_limiting_factor.compile((types.Array(types.int8, 3, "C"), types.Array(types.int64, 2, "C") ))

    print(f'    → 100 % complete')
    print(f'... completed within {int((datetime.now() - s).total_seconds())} seconds\n')


def calculate_climate_suitability(temp_files, prec_files, config_dict, plant_params, extent, full_area_name):
    print('\nStarting calculation of climate suitability\n')

    precompile()

    temp_folder = os.path.join(os.getcwd(), 'temp')
    os.makedirs(temp_folder, exist_ok=True)

    consider_variability = int(config_dict['climatevariability'].get('consider_variability', 0))
    if consider_variability:
        print('Module for the consideration of interannual climate variability is activated')
    else:
        print('Module for the consideration of interannual climate variability is deactivated')

    print('\nLoading temperature data...')
    temp_data = read_climate_data(temp_files, extent)
    print('\nLoading precipitation data...')
    prec_data = read_climate_data(prec_files, extent)
    
    print(f'\n  → Climate data loaded with shape {temp_data.shape}')
    print(f'    Temperature data loaded as {temp_data.dtype}')
    print(f'    Temperature data requires {np.ceil(temp_data.shape[0] * temp_data.shape[1] * temp_data.shape[2] * temp_data.dtype.itemsize / 1e8) / 10} GB memory')
    print(f'    Precipitation data loaded as {prec_data.dtype}')
    print(f'    Precipitation data requires {np.ceil(prec_data.shape[0] * prec_data.shape[1] * prec_data.shape[2] * temp_data.dtype.itemsize / 1e8) / 10} GB memory')
    print(f'  → Pixels to process: {temp_data.shape[1] * temp_data.shape[2]}')
    print(f'  → Area to process: {extent[0] - extent[2]}° x {extent[3] - extent[1]}°\n')
    
    for f in os.listdir(temp_folder):
        try:
            os.remove(os.path.join(temp_folder, f))
        except:
            pass

    for crop in plant_params:
        irrigation_suffix = '_ir' if config_dict['options'].get('irrigation', False) else '_rf'
        output = os.path.join(config_dict['files'].get('output_dir'), f'var{irrigation_suffix}' if consider_variability else f'novar{irrigation_suffix}', full_area_name, crop)
        if os.path.exists(os.path.join(output, f'climate_suitability.{"tif" if config_dict.get("options", {}).get("output_format", "geotiff").lower() == "geotiff" else "nc"}')):
            continue

        print(f"\n{'-'*30}\n{crop.replace('_', ' ').title().center(30)}\n{'-'*30}\n")

        growing_cycle = int(plant_params[crop]['growing_cycle'][0])
        
        ### TEMPERATURE ###
        temperature(temp_data, growing_cycle, plant_params, crop, temp_folder, extent)

        ### PRECIPITATION ###
        precipitation(prec_data, growing_cycle, plant_params, crop, temp_folder, extent)

        ### RECURRENCE RATE OF POTENTIAL CROP FAILURES ###
        if consider_variability:
            if int(config_dict['options'].get('irrigation', 0)) == 0:
                consider_rrpcf(config_dict, plant_params, crop, temp_folder, extent, irrigation='rf', full_name=full_area_name)
            if int(config_dict['options'].get('irrigation', 0)) == 1:
                consider_rrpcf(config_dict, plant_params, crop, temp_folder, extent, irrigation='ir', full_name=full_area_name)
            if int(config_dict['options'].get('irrigation', 0)) == 2:
                consider_rrpcf(config_dict, plant_params, crop, temp_folder, extent, irrigation='rfir', full_name=full_area_name)

        if plant_params.get(crop, {}).get('photoperiod', False):
            photoperiod(plant_params, crop, growing_cycle, temp_folder, extent, temp_data.shape)

        print('\n ✖   Module for consideration of extreme events not found.')

    del temp_data
    del prec_data
    gc.collect()

    for crop in plant_params:
        print(f"\n{'-'*30}\n{crop.replace('_', ' ').title().center(30)}\n{'-'*30}\n")

        irr = int(config_dict['options'].get('irrigation', 0))
        rrpcf_map = {0: [(0, f'rrpcf_{crop}_rf{suffix}')], 1: [(1, f'rrpcf_{crop}_ir{suffix}')], 2: [(0, f'rrpcf_{crop}_rf{suffix}'), (1, f'rrpcf_{crop}_ir{suffix}')]}
        rrpcf_list = rrpcf_map.get(irr, rrpcf_map[0])

        temp_file = os.path.join(temp_folder, f'temp_{crop}{suffix}')
        if not os.path.exists(temp_file):
            continue
        prec_file = os.path.join(temp_folder, f'prec_{crop}{suffix}')

        consider_variability = int(config_dict['climatevariability'].get('consider_variability', False))
        variability_range = [consider_variability] if consider_variability < 2 else [0, 1]

        for var in variability_range:
            for irr, rrpcf in rrpcf_list:
                suit_files = [temp_file, prec_file, os.path.join(temp_folder, rrpcf)]
                if plant_params.get(crop, {}).get('photoperiod'):
                    suit_files.append(os.path.join(temp_folder, f'photo_{crop}{suffix}'))
                create_suitability(suit_files, config_dict, plant_params[crop], crop, full_area_name, extent, irrigation=irr, consider_variability=var)

    for f in os.listdir(temp_folder):
        try:
            os.remove(os.path.join(temp_folder, f))
        except FileNotFoundError:
            pass
        except Exception:
            try:
                shutil.rmtree(os.path.join(temp_folder, f))
            except:
                print(f' ⚠ Unable to remove {f}')
        


def create_suitability(suit_files, config_dict, plant_params, crop, full_area_name, extent, irrigation=0, consider_variability=0):
    print(f' → Processing {"irrigated" if irrigation == 1 else "rainfed"} conditions {"with" if consider_variability == 1 else "without"} variability')
    growing_cycle = int(plant_params.get('growing_cycle')[0])
    wintercrop = 1 if str(plant_params.get('wintercrop', [0])[0]).lower() in ('y', '1') else 0
    
    with loader(suit_files[0]) as npz:
        shape = npz['temp'].shape

    arrays = np.full((len(suit_files),) + shape, dtype=np.int8, fill_value=100)
    current_index = 0

    load_array(suit_files[0], 'temp', out=arrays[current_index])
    current_index += 1

    if irrigation:
        arrays[current_index].fill(100)
    else:
        load_array(suit_files[1], 'prec', out=arrays[current_index])
    current_index += 1

    if int(consider_variability) == 1 and os.path.exists(suit_files[2]):
        try:
            if growing_cycle == 365:
                with loader(suit_files[2]) as npz:
                    rrpcf_data = npz['rrpcf'][0]
                    if rrpcf_data.shape != shape:
                        y, x = shape
                        rrpcf_data = rrpcf_data[:y, :x]
                        rrpcf_data = np.pad(rrpcf_data, ((0, y-rrpcf_data.shape[0]), (0, x-rrpcf_data.shape[1])), constant_values=0)
                    np.copyto(arrays[current_index], rrpcf_data)
            else:
                with loader(suit_files[2]) as npz:
                    rrpcf_data = npz['rrpcf']
                    if rrpcf_data.shape != shape:
                        _, y, x = shape
                        rrpcf_data = rrpcf_data[:shape[0], :shape[1], :shape[2]]
                        rrpcf_data = np.pad(rrpcf_data, [(0,s-a) for s,a in zip(shape, rrpcf_data.shape)], constant_values=0)
                    arrays[current_index] = rrpcf_data
                    del rrpcf_data
            gc.collect()
            current_index += 1
        except Exception as e:
            print(f"Error loading variability file {suit_files[2]}: {e}")
            consider_variability = 0

    for f in suit_files[3:]:
        if os.path.exists(f):
            try:
                load_array(f, out=arrays[current_index])
                current_index += 1
            except Exception as e:
                print(f"Error loading {f}: {e}")

    irrigation_suffix = '_ir' if irrigation == 1 else '_rf'
    area_name = f'Area_{int(extent[0])}N{int(extent[1])}E-{int(extent[2])}N{int(extent[3])}E'
    output_directory = os.path.join(config_dict['files'].get('output_dir'), f'var{irrigation_suffix}' if int(consider_variability) == 1 else f'novar{irrigation_suffix}', area_name, crop)
    os.makedirs(output_directory, exist_ok=True)
    mode = 'geotiff' if config_dict.get('options', {}).get('output_format', 'geotiff').lower() == 'geotiff' else 'netcdf'
    
    if growing_cycle >= 365:
        if config_dict.get('outputs', {}).get('climate_suitability', True):
            suitability = np.min(arrays, axis=0)
            write_raster(os.path.join(output_directory, 'climate_suitability.tif'), suitability, extent, mode=mode)
        if config_dict.get('outputs', {}).get('limiting_factor', True):
            limiting_factor = np.argmin(arrays, axis=0).astype(np.int8)
            limiting_factor[suitability < 0] = -1
            write_raster(os.path.join(output_directory, 'limiting_factor.tif'), limiting_factor, extent, mode=mode)
            del limiting_factor
        del suitability

    else:
        suitability = np.min(arrays, axis=0)
        gc.collect()

        sowing_date = None
        if config_dict.get('outputs', {}).get('suitable_sowing_days', True):
            suit_sowing_days = np.sum(suitability > 0, axis=0, dtype=np.int16)
            write_raster(os.path.join(output_directory, 'suitable_sowing_days.tif'), suit_sowing_days, extent, mode=mode)
            del suit_sowing_days
            gc.collect()
        suitability_max = np.max(suitability, axis=0)
        sowing_date = find_optimal_sowing_date(suitability)
        sowing_date[suitability_max <= 0] = -1

        if config_dict.get('outputs', {}).get('climate_suitability', True):
            write_raster(os.path.join(output_directory, 'climate_suitability.tif'), suitability_max, extent, mode=mode)
        del suitability_max

        if config_dict.get('outputs', {}).get('limiting_factor', True):
            limiting_factor = get_limiting_factor(arrays, sowing_date)
            write_raster(os.path.join(output_directory, 'limiting_factor.tif'), limiting_factor, extent, mode=mode)
            del limiting_factor
        del arrays
        gc.collect()

        if config_dict.get('outputs', {}).get('optimal_sowing_date', True):
            write_raster(os.path.join(output_directory, 'optimal_sowing_date.tif'), sowing_date, extent, mode=mode)

        if config_dict.get('options', {}).get('start_growing_cycle_after_vernalization', False) and wintercrop:
            with loader(os.path.join(os.getcwd(), 'temp', f'temp_sowaftvern_{crop}{suffix}')) as npz:
                date_emergence = get_emergence(sowing_date, npz['sowing'])
            write_raster(os.path.join(output_directory, 'start_growing_cycle_after_vernalization.tif'), date_emergence, extent, mode=mode)
            del date_emergence
            gc.collect()
        del sowing_date

        gc.collect()

        if config_dict.get('options', {}).get('consider_crop_rotation', False):
            write_raster(os.path.join(output_directory, 'climatesuitability_temp.tif'), suitability, extent, mode=mode)

        if not wintercrop and config_dict.get('options', {}).get('consider_multiple_cropping', False):
            cultivation_period = int(config_dict.get('options', {}).get('multiple_cropping_turnaround_time', [21]))
            multiple_cropping, mc_first, mc_second, mc_third = find_multiple_cropping(suitability, growing_cycle + cultivation_period, threshold=25)

            if config_dict.get('outputs', {}).get('multiple_cropping', True):
                write_raster(os.path.join(output_directory, 'multiple_cropping.tif'), multiple_cropping, extent, mode=mode)
            del multiple_cropping

            if config_dict.get('outputs', {}).get('optimal_sowing_date_mc_first', True):
                write_raster(os.path.join(output_directory, 'optimal_sowing_date_mc_first.tif'), mc_first, extent, mode=mode)
            del mc_first

            if config_dict.get('outputs', {}).get('optimal_sowing_date_mc_second', True):
                write_raster(os.path.join(output_directory, 'optimal_sowing_date_mc_second.tif'), mc_second, extent, mode=mode)
            del mc_second

            if config_dict.get('outputs', {}).get('optimal_sowing_date_mc_third', True):
                write_raster(os.path.join(output_directory, 'optimal_sowing_date_mc_third.tif'), mc_third, extent, mode=mode)
            del mc_third
            gc.collect()
        try:
            del suitability
        except:
            pass
    gc.collect()

@njit(parallel=True)
def get_limiting_factor(arrays, sowing_date):
    if arrays.ndim == 3:
        L, Y, X = arrays.shape
        result = np.empty((Y, X), dtype=np.int8)
        for y in prange(Y):
            for x in range(X):
                if arrays[0, y, x] < 0:
                    result[y, x] = -1
                    continue
                result[y, x] = np.argmin(arrays[:, y, x])
        return result
    L, T, Y, X = arrays.shape
    result = np.full((Y, X), -1, dtype=np.int8)
    for y in prange(Y):
        for x in range(X):
            date = sowing_date[y, x]
            if date >= 0:
                result[y, x] = np.argmin(arrays[:, date, y, x])
            else:
                if arrays[0, 0, y, x] < 0:
                    result[y, x] = -1
                    continue
                time = np.sum(arrays[:, :, y, x], axis=0)
                max_date = np.argmax(time)
                result[y, x] = np.argmin(arrays[:, max_date, y, x])
    return result


@njit(parallel=True)
def find_optimal_sowing_date(arr3d):
    nt, ny, nx = arr3d.shape
    result = np.zeros((ny, nx), dtype=np.int16)
    for i in prange(ny):
        for j in range(nx):
            max_val = -1e9
            for k in range(nt):
                if arr3d[k, i, j] > max_val:
                    max_val = arr3d[k, i, j]
            if max_val == 0:
                continue
            best_len = 0
            best_start = 0
            k = 0
            while k < nt:
                if arr3d[k, i, j] == max_val:
                    start = k
                    while k < nt and arr3d[k, i, j] == max_val:
                        k += 1
                    length = k - start
                    if length > best_len:
                        best_len = length
                        best_start = start
                else:
                    k += 1
            result[i, j] = best_start + (best_len - 1) // 2
    return result


def write_raster(filename, data, extent, mode='geotiff', noddata=-1):
    if mode == 'geotiff':
        write_geotiff(filename, data, extent, nodata=noddata)
    elif mode == 'netcdf':
        write_netcdf(filename, data, extent, nodata=noddata)
    

def write_geotiff(filename, data, extent, nodata=-1):
    if data.ndim == 2:
        data = data[np.newaxis]
    count, h, w = data.shape
    transform = from_bounds(extent[1], extent[2], extent[3], extent[0], w, h)
    metadata = {
        'AUTHOR': 'Matthias Knüttel & Florian Zabel',
        'CREATION_DATE': datetime.now().strftime('%Y-%m-%d'),
        'VERSION': 'CropSuite v.1.6.0',
        'COPYRIGHT': '© Matthias Knüttel & Florian Zabel 2025'
    }
    with rasterio.open(filename, "w", driver="GTiff", height=h, width=w, count=count,
                       dtype=data.dtype, crs="EPSG:4326", transform=transform,
                       nodata=nodata, compress="LZW", tags=metadata) as dst:
        dst.write(data)


def write_netcdf(filename, data, extent, nodata=-1, crs_epsg="EPSG:4326"):
    if data.ndim == 2:
        data = data[np.newaxis:]
    count, h, w = data.shape
    lat = np.linspace(extent[0], extent[2], h)
    lon = np.linspace(extent[1], extent[3], w)
    with Dataset(filename, "w", format="NETCDF4") as ds:
        ds.createDimension("band", count)
        ds.createDimension("lat", h)
        ds.createDimension("lon", w)
        lat_var = ds.createVariable("lat", "f4", ("lat",))
        lon_var = ds.createVariable("lon", "f4", ("lon",))
        lat_var.standard_name = "latitude"
        lat_var.units = "degrees_north"
        lon_var.standard_name = "longitude"
        lon_var.units = "degrees_east"
        lat_var[:], lon_var[:] = lat, lon
        data_var = ds.createVariable("data", data.dtype.str, ("band", "lat", "lon"), zlib=True, complevel=9, fill_value=nodata)
        data_var.coordinates = "lat lon"
        data_var.grid_mapping = "crs"
        data_var[:] = data
        crs = ds.createVariable("crs", "i4")
        crs.grid_mapping_name = "latitude_longitude"
        crs.epsg_code = crs_epsg
        crs.longitude_of_prime_meridian = 0.0
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        ds.setncatts({
            'AUTHOR': "Matthias Knüttel & Florian Zabel",
            'CREATION_DATE': datetime.now().strftime("%Y-%m-%d"),
            'VERSION': "CropSuite v.1.6.0",
            'COPYRIGHT': "© Matthias Knüttel & Florian Zabel 2025"
        })


def print_progress(i, n, suffix='   ', length=50):
    filled = int(length*i//n)
    sys.stdout.write(f"\r{suffix}[{'█'*filled}{'.'*(length-filled)}] {i}/{n}")
    sys.stdout.flush()
    if i==n: print()



def _find_all_arrays(node, prefix=""):
    arrays = {}
    if isinstance(node, zarr.Array):
        arrays[prefix.rstrip("/")] = node
    elif isinstance(node, zarr.Group):
        for name, arr in node.arrays():
            arrays[prefix + name] = arr
        for name, subgrp in node.groups():
            arrays.update(_find_all_arrays(subgrp, prefix + name + "/"))
    return arrays


def read_climate_data(data_file, extent, parameter=None):
    root = zarr.open(data_file, mode="r")
    arrays = _find_all_arrays(root)
    if parameter is None:
        if len(arrays) == 1:
            arr = next(iter(arrays.values()))
        else:
            raise ValueError(f"Multiple arrays found: {list(arrays.keys())}.")
    else:
        if parameter not in arrays:
            raise KeyError(f"Array '{parameter}' not found. Available: {list(arrays.keys())}")
        arr = arrays[parameter]
    ymax, xmin, ymin, xmax = root.attrs["extent"] #type:ignore
    ny, nx = arr.shape[-2:]
    dy = (ymax - ymin) / ny #type:ignore
    dx = (xmax - xmin) / nx #type:ignore
    y_max_req, x_min_req, y_min_req, x_max_req = extent
    row_start = int((ymax - y_max_req) / dy)
    row_stop  = int((ymax - y_min_req) / dy)
    col_start = int((x_min_req - xmin) / dx)
    col_stop  = int((x_max_req - xmin) / dx)
    row_start = max(0, row_start)
    row_stop  = min(ny, row_stop)
    col_start = max(0, col_start)
    col_stop  = min(nx, col_stop)
    if arr.ndim == 3:
        data = arr[:, row_start:row_stop, col_start:col_stop]
    else:
        data = arr[row_start:row_stop, col_start:col_stop]
    return np.asarray(data)