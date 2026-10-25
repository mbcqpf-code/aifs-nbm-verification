import os
import glob
import time
import argparse
import requests
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from datetime import datetime, timedelta
from herbie import Herbie
from ecmwf.opendata import Client
import warnings
warnings.filterwarnings('ignore')

try:
    import multiurl.http
    multiurl.http.ROBUST_DEFAULTS = {"maximum_tries": 5, "retry_after": 5}
except Exception:
    pass

LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = -125.0, -65.0
grid_lats = np.arange(LAT_MIN, LAT_MAX + 0.25, 0.25)
grid_lons = np.arange(LON_MIN, LON_MAX + 0.25, 0.25)
TARGET_LONS, TARGET_LATS = np.meshgrid(grid_lons, grid_lats)

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def regrid_to_conus(data, lats, lons):
    if lons.max() > 180:
        lons = np.where(lons > 180, lons - 360, lons)
    mask = (lats >= LAT_MIN - 2) & (lats <= LAT_MAX + 2) & (lons >= LON_MIN - 2) & (lons <= LON_MAX + 2)
    return griddata(
        (lons[mask], lats[mask]), 
        data[mask], 
        (TARGET_LONS, TARGET_LATS), 
        method='linear'
    )

# ------------------------------------------------------------------------------
# 1. URMA Ingestion (Hourly Proxies & True Extremes)
# ------------------------------------------------------------------------------
def get_urma_tmp(dt_str):
    H = Herbie(dt_str, model="urma", product="anl")
    ds = H.xarray(":TMP:2 m", remove_grib=True)
    if isinstance(ds, list): ds = ds[0]
    var_key = list(ds.data_vars)[0]
    data_k = ds[var_key].squeeze().values[::2, ::2]
    lats = ds.latitude.values[::2, ::2]
    lons = ds.longitude.values[::2, ::2]
    ds.close()
    return (regrid_to_conus(data_k, lats, lons) - 273.15) * 9/5 + 32

def get_urma_all(target_dt):
    date_str = target_dt.strftime('%Y-%m-%d')
    next_day_str = (target_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # Instantaneous hourly proxies
    t_12z = get_urma_tmp(f"{date_str} 12:00")
    t_18z = get_urma_tmp(f"{date_str} 18:00")
    t_00z = get_urma_tmp(f"{next_day_str} 00:00")
    proxy_min = t_12z
    proxy_max = np.maximum(t_18z, t_00z)

    # True daily extremes
    H_max = Herbie(f"{next_day_str} 08:00", model="urma", product="anl")
    ds_max = H_max.xarray(":TMAX:2 m above ground:", remove_grib=True)
    t_max_data = ds_max[list(ds_max.data_vars)[0]].squeeze().values[::2, ::2]
    true_max = (regrid_to_conus(t_max_data, ds_max.latitude.values[::2, ::2], ds_max.longitude.values[::2, ::2]) - 273.15) * 9/5 + 32

    H_min = Herbie(f"{date_str} 20:00", model="urma", product="anl")
    ds_min = H_min.xarray(":TMIN:2 m above ground:", remove_grib=True)
    t_min_data = ds_min[list(ds_min.data_vars)[0]].squeeze().values[::2, ::2]
    true_min = (regrid_to_conus(t_min_data, ds_min.latitude.values[::2, ::2], ds_min.longitude.values[::2, ::2]) - 273.15) * 9/5 + 32

    return proxy_min, proxy_max, true_min, true_max

# ------------------------------------------------------------------------------
# 2. AIFS Direct Mean Retrieval
# ------------------------------------------------------------------------------
def get_aifs_hourly(init_date_str, lead_day):
    dt_clean = init_date_str.replace("-", "")
    fxx_12z = lead_day * 24 + 12
    fxx_18z = lead_day * 24 + 18
    fxx_00z = lead_day * 24 + 24
    needed_steps = [fxx_12z, fxx_18z, fxx_00z]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
        tmp_target = tmp.name

    success = False
    for source in ["azure", "aws", "ecmwf"]:
        try:
            client = Client(source=source, model="aifs-ens", resol="0p25")
            client.retrieve(
                date=dt_clean,
                time=0,
                step=needed_steps,
                stream="enfo",
                type="em",
                levtype="sfc",
                param="2t",
                target=tmp_target
            )
            success = True
            break
        except Exception:
            continue

    if not success:
        if os.path.exists(tmp_target): os.remove(tmp_target)
        raise RuntimeError(f"Could not retrieve AIFS ENS steps {needed_steps} for {dt_clean}")

    ds = xr.open_dataset(tmp_target, engine='cfgrib', backend_kwargs={'indexpath': ''})
    var_key = 't2m' if 't2m' in ds else list(ds.data_vars)[0]
    da = ds[var_key]

    if da.longitude.max() > 180:
        da = da.assign_coords(longitude=(((da.longitude + 180) % 360) - 180))
    da = da.sortby(['latitude', 'longitude'])
    da_interp = da.interp(latitude=grid_lats, longitude=grid_lons, method='linear')
    vals = da_interp.values
    ds.close()
    for f in glob.glob(f"{tmp_target}*"):
        try: os.remove(f)
        except Exception: pass

    if vals.ndim == 3:
        t_12z = (vals[0] - 273.15) * 9/5 + 32
        t_18z = (vals[1] - 273.15) * 9/5 + 32
        t_00z = (vals[2] - 273.15) * 9/5 + 32
    else:
        v = (vals - 273.15) * 9/5 + 32
        t_12z, t_18z, t_00z = v, v, v

    return t_12z, t_18z, t_00z

# ------------------------------------------------------------------------------
# 3. NBM 06z QMD Retrieval (Hourly 50% TMP & True 50% Max/Min)
# ------------------------------------------------------------------------------
def get_nbm_qmd_grid(init_dt, lead_hrs, var_type):
    date_str = init_dt.strftime('%Y%m%d')
    cycle_str = f"{init_dt.hour:02d}"
    base_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{cycle_str}/qmd"
    grib_name = f"blend.t{cycle_str}z.qmd.f{lead_hrs:03d}.co.grib2"
    
    r_idx = requests.get(f"{base_url}/{grib_name}.idx", timeout=20)
    if r_idx.status_code != 200:
        raise FileNotFoundError(f"Index not found: {grib_name}")
        
    start_byte, end_byte = None, None
    lines = r_idx.text.splitlines()
    for i, line in enumerate(lines):
        line_l = line.lower()
        if "50% level" in line_l and "2 m above ground" in line_l:
            if var_type == 'tmp' and "max" not in line_l and "min" not in line_l:
                start_byte = int(line.split(':')[1])
                if i + 1 < len(lines): end_byte = int(lines[i+1].split(':')[1]) - 1
                break
            elif var_type == 'maxt' and "max" in line_l:
                start_byte = int(line.split(':')[1])
                if i + 1 < len(lines): end_byte = int(lines[i+1].split(':')[1]) - 1
                break
            elif var_type == 'mint' and "min" in line_l:
                start_byte = int(line.split(':')[1])
                if i + 1 < len(lines): end_byte = int(lines[i+1].split(':')[1]) - 1
                break
                
    if start_byte is None:
        raise ValueError(f"Could not locate {var_type} (50%) in {grib_name}.idx")
        
    headers = {"Range": f"bytes={start_byte}-{end_byte}" if end_byte else f"bytes={start_byte}-"}
    r = requests.get(f"{base_url}/{grib_name}", headers=headers, timeout=30)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name
        
    ds = xr.open_dataset(tmp_path, engine='cfgrib', backend_kwargs={'indexpath': ''})
    var_name = list(ds.data_vars)[0]
    data = ds[var_name].values
    lats = ds.latitude.values
    lons = ds.longitude.values
    ds.close()
    for f in glob.glob(f"{tmp_path}*"):
        try: os.remove(f)
        except Exception: pass
    
    data_k = regrid_to_conus(data[::2, ::2], lats[::2, ::2], lons[::2, ::2])
    return (data_k - 273.15) * 9/5 + 32

def get_nbm_all(init_dt, lead_day):
    fxx_12z = lead_day * 24 + 6
    fxx_18z = lead_day * 24 + 12
    fxx_00z = lead_day * 24 + 18

    # Hourly synoptic temperatures
    t_12z = get_nbm_qmd_grid(init_dt, fxx_12z, 'tmp')
    t_18z = get_nbm_qmd_grid(init_dt, fxx_18z, 'tmp')
    t_00z = get_nbm_qmd_grid(init_dt, fxx_00z, 'tmp')

    # True daily extremes (from QMD distribution)
    true_min = get_nbm_qmd_grid(init_dt, fxx_12z, 'mint')
    true_max = get_nbm_qmd_grid(init_dt, fxx_00z, 'maxt')

    return t_12z, t_18z, t_00z, true_min, true_max

# ------------------------------------------------------------------------------
# 4. Pipeline Execution
# ------------------------------------------------------------------------------
def process_date(target_dt, overwrite=False):
    date_str = target_dt.strftime('%Y-%m-%d')
    nc_path = os.path.join(CACHE_DIR, f"diff_{date_str}.nc")
    if os.path.exists(nc_path) and not overwrite:
        print(f"[{date_str}] Cache exists. Skipping.")
        return

    print(f"\n--- Processing Valid Date: {date_str} ---")
    try:
        urma_pmin, urma_pmax, urma_tmin, urma_tmax = get_urma_all(target_dt)
        print("  -> Verifying URMA fields (proxies + true extremes) loaded.")
    except Exception as e:
        print(f"  [!] URMA failed for {date_str}: {e}. Skipping date.")
        return

    daily_diffs = {}
    for lead in range(1, 9):
        init_date = target_dt - pd.Timedelta(days=lead)
        init_date_str = init_date.strftime('%Y-%m-%d')
        init_06z = pd.to_datetime(f"{init_date_str} 06:00")
        
        # 1. Ingest AIFS
        try:
            aifs_12z, aifs_18z, aifs_00z = get_aifs_hourly(init_date_str, lead)
            aifs_pmin = aifs_12z
            aifs_pmax = np.maximum(aifs_18z, aifs_00z)
        except Exception as e:
            print(f"  [!] AIFS Day {lead} failed: {e}")
            aifs_pmin, aifs_pmax = None, None

        time.sleep(1.2)

        # 2. Ingest NBM
        try:
            nbm_12z, nbm_18z, nbm_00z, nbm_tmin, nbm_tmax = get_nbm_all(init_06z, lead)
            nbm_pmin = nbm_12z
            nbm_pmax = np.maximum(nbm_18z, nbm_00z)
        except Exception as e:
            print(f"  [!] NBM Day {lead} failed: {e}")
            nbm_pmin, nbm_pmax, nbm_tmin, nbm_tmax = None, None, None, None

        # 3. Mode 1: Proxy Differences
        if aifs_pmin is not None and urma_pmin is not None:
            daily_diffs[f'aifs_lead_{lead}_maxt_proxy_diff'] = (['lat', 'lon'], (aifs_pmax - urma_pmax).astype(np.float32))
            daily_diffs[f'aifs_lead_{lead}_mint_proxy_diff'] = (['lat', 'lon'], (aifs_pmin - urma_pmin).astype(np.float32))
        else:
            daily_diffs[f'aifs_lead_{lead}_maxt_proxy_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
            daily_diffs[f'aifs_lead_{lead}_mint_proxy_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))

        if nbm_pmin is not None and urma_pmin is not None:
            daily_diffs[f'nbm_lead_{lead}_maxt_proxy_diff'] = (['lat', 'lon'], (nbm_pmax - urma_pmax).astype(np.float32))
            daily_diffs[f'nbm_lead_{lead}_mint_proxy_diff'] = (['lat', 'lon'], (nbm_pmin - urma_pmin).astype(np.float32))
        else:
            daily_diffs[f'nbm_lead_{lead}_maxt_proxy_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
            daily_diffs[f'nbm_lead_{lead}_mint_proxy_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))

        # 4. Mode 2: Diurnal Corrections & True Extreme Differences
        if nbm_pmin is not None and nbm_tmin is not None:
            delta_max = np.clip(nbm_tmax - nbm_pmax, 0, None) # Diurnal boost >= 0
            delta_min = np.clip(nbm_tmin - nbm_pmin, None, 0) # Nighttime cooling <= 0

            daily_diffs[f'lead_{lead}_maxt_delta'] = (['lat', 'lon'], delta_max.astype(np.float32))
            daily_diffs[f'lead_{lead}_mint_delta'] = (['lat', 'lon'], np.abs(delta_min).astype(np.float32))

            # NBM true error
            daily_diffs[f'nbm_lead_{lead}_maxt_corr_diff'] = (['lat', 'lon'], (nbm_tmax - urma_tmax).astype(np.float32))
            daily_diffs[f'nbm_lead_{lead}_mint_corr_diff'] = (['lat', 'lon'], (nbm_tmin - urma_tmin).astype(np.float32))

            # AIFS diurnally corrected error
            if aifs_pmin is not None:
                aifs_corr_max = aifs_pmax + delta_max
                aifs_corr_min = aifs_pmin + delta_min
                daily_diffs[f'aifs_lead_{lead}_maxt_corr_diff'] = (['lat', 'lon'], (aifs_corr_max - urma_tmax).astype(np.float32))
                daily_diffs[f'aifs_lead_{lead}_mint_corr_diff'] = (['lat', 'lon'], (aifs_corr_min - urma_tmin).astype(np.float32))
            else:
                daily_diffs[f'aifs_lead_{lead}_maxt_corr_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
                daily_diffs[f'aifs_lead_{lead}_mint_corr_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
        else:
            daily_diffs[f'lead_{lead}_maxt_delta'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
            daily_diffs[f'lead_{lead}_mint_delta'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
            daily_diffs[f'nbm_lead_{lead}_maxt_corr_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
            daily_diffs[f'nbm_lead_{lead}_mint_corr_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
            daily_diffs[f'aifs_lead_{lead}_maxt_corr_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
            daily_diffs[f'aifs_lead_{lead}_mint_corr_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))

        print(f"  -> Day {lead} successfully computed.")

    ds_out = xr.Dataset(
        data_vars=daily_diffs,
        coords={'lat': grid_lats, 'lon': grid_lons}
    )
    ds_out.to_netcdf(nc_path, encoding={v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars})
    print(f"==> Saved cache: {nc_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", type=int, default=1, help="Number of historical days")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    args = parser.parse_args()

    base_target = pd.Timestamp.utcnow().floor('D') - pd.Timedelta(days=1)
    for d in range(args.backfill):
        process_date(base_target - pd.Timedelta(days=d), overwrite=args.overwrite)

    all_nc = sorted(glob.glob(f"{CACHE_DIR}/diff_*.nc"))
    if len(all_nc) > 25:
        for old_file in all_nc[:-25]:
            os.remove(old_file)
