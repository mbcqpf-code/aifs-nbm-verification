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

# ------------------------------------------------------------------------------
# 1. Neutralize multiurl 120s Sleep
# ------------------------------------------------------------------------------
try:
    import multiurl.http
    multiurl.http.ROBUST_DEFAULTS = {"maximum_tries": 5, "retry_after": 5}
except Exception:
    pass

# ------------------------------------------------------------------------------
# 2. Target 0.25° CONUS Grid
# ------------------------------------------------------------------------------
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
# 3. URMA Hourly 2m Temperatures (12z, 18z, 00z)
# ------------------------------------------------------------------------------
def get_urma_tmp(dt_str):
    """Pulls URMA instantaneous 2m temperature analysis for an exact UTC hour."""
    H = Herbie(dt_str, model="urma", product="anl")
    ds = H.xarray(":TMP:2 m", remove_grib=True)
    if isinstance(ds, list): ds = ds[0]
    var_key = list(ds.data_vars)[0]
    data_k = ds[var_key].squeeze().values[::2, ::2]
    lats = ds.latitude.values[::2, ::2]
    lons = ds.longitude.values[::2, ::2]
    ds.close()
    
    regridded_k = regrid_to_conus(data_k, lats, lons)
    return (regridded_k - 273.15) * 9/5 + 32

def get_urma_proxies(target_dt):
    """Computes URMA 12z Min and max(18z, 00z) Max proxies."""
    date_str = target_dt.strftime('%Y-%m-%d')
    next_day_str = (target_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    t_12z = get_urma_tmp(f"{date_str} 12:00")
    t_18z = get_urma_tmp(f"{date_str} 18:00")
    t_00z = get_urma_tmp(f"{next_day_str} 00:00")

    urma_min = t_12z
    urma_max = np.maximum(t_18z, t_00z)
    return urma_min, urma_max

# ------------------------------------------------------------------------------
# 4. AIFS Direct Mean Batched Retrieval
# ------------------------------------------------------------------------------
def get_aifs_proxies(init_date_str, lead_day):
    """Batched pull of AIFS 12z, 18z, and 00z steps."""
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
        t_min_k = vals[0]
        t_max_k = np.maximum(vals[1], vals[2])
    else:
        t_min_k, t_max_k = vals, vals

    return (t_min_k - 273.15) * 9/5 + 32, (t_max_k - 273.15) * 9/5 + 32

# ------------------------------------------------------------------------------
# 5. NBM Hourly 2m Temperature (Core GRIBs via S3 Byte-Ranges)
# ------------------------------------------------------------------------------
def get_nbm_hourly_tmp(init_dt, lead_hrs):
    """Pulls exact hourly 2m TMP from NBM core."""
    date_str = init_dt.strftime('%Y%m%d')
    cycle_str = f"{init_dt.hour:02d}"
    base_url = f"https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{date_str}/{cycle_str}/core"
    grib_name = f"blend.t{cycle_str}z.core.f{lead_hrs:03d}.co.grib2"
    
    r_idx = requests.get(f"{base_url}/{grib_name}.idx", timeout=20)
    if r_idx.status_code != 200:
        raise FileNotFoundError(f"Index not found: {grib_name}")
        
    start_byte, end_byte = None, None
    lines = r_idx.text.splitlines()
    for i, line in enumerate(lines):
        line_l = line.lower()
        if ":tmp:" in line_l and "2 m" in line_l:
            start_byte = int(line.split(':')[1])
            if i + 1 < len(lines): end_byte = int(lines[i+1].split(':')[1]) - 1
            break
            
    if start_byte is None:
        raise ValueError(f"Could not locate 2m TMP in {grib_name}.idx")
        
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

def get_nbm_proxies(init_dt, lead_day):
    """Pulls matching NBM 12z, 18z, and 00z temperatures."""
    fxx_12z = lead_day * 24 + 6
    fxx_18z = lead_day * 24 + 12
    fxx_00z = lead_day * 24 + 18

    t_12z = get_nbm_hourly_tmp(init_dt, fxx_12z)
    t_18z = get_nbm_hourly_tmp(init_dt, fxx_18z)
    t_00z = get_nbm_hourly_tmp(init_dt, fxx_00z)

    nbm_min = t_12z
    nbm_max = np.maximum(t_18z, t_00z)
    return nbm_min, nbm_max

# ------------------------------------------------------------------------------
# 6. Daily Processing Execution
# ------------------------------------------------------------------------------
def process_date(target_dt):
    date_str = target_dt.strftime('%Y-%m-%d')
    nc_path = os.path.join(CACHE_DIR, f"diff_{date_str}.nc")
    if os.path.exists(nc_path):
        print(f"[{date_str}] Cache exists. Skipping.")
        return

    print(f"\n--- Processing Valid Date: {date_str} ---")
    try:
        urma_min_f, urma_max_f = get_urma_proxies(target_dt)
        print("  -> URMA 12z, 18z, 00z hourly analyses processed.")
    except Exception as e:
        print(f"  [!] URMA unavailable for {date_str}: {e}. Skipping date.")
        return

    daily_diffs = {}
    for lead in range(1, 9):
        init_date = target_dt - pd.Timedelta(days=lead)
        init_date_str = init_date.strftime('%Y-%m-%d')
        init_06z = pd.to_datetime(f"{init_date_str} 06:00")
        
        # AIFS Proxies
        try:
            aifs_min, aifs_max = get_aifs_proxies(init_date_str, lead)
            daily_diffs[f'aifs_lead_{lead}_maxt_diff'] = (['lat', 'lon'], (aifs_max - urma_max_f).astype(np.float32))
            daily_diffs[f'aifs_lead_{lead}_mint_diff'] = (['lat', 'lon'], (aifs_min - urma_min_f).astype(np.float32))
            print(f"  -> Day {lead} AIFS (12z & max(18z,00z)) processed.")
        except Exception as e:
            print(f"  [!] AIFS Day {lead} failed: {e}")
            daily_diffs[f'aifs_lead_{lead}_maxt_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
            daily_diffs[f'aifs_lead_{lead}_mint_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))

        time.sleep(1.2)

        # NBM Proxies
        try:
            nbm_min, nbm_max = get_nbm_proxies(init_06z, lead)
            daily_diffs[f'nbm_lead_{lead}_maxt_diff'] = (['lat', 'lon'], (nbm_max - urma_max_f).astype(np.float32))
            daily_diffs[f'nbm_lead_{lead}_mint_diff'] = (['lat', 'lon'], (nbm_min - urma_min_f).astype(np.float32))
            print(f"  -> Day {lead} NBM (12z & max(18z,00z)) processed.")
        except Exception as e:
            print(f"  [!] NBM Day {lead} failed: {e}")
            daily_diffs[f'nbm_lead_{lead}_maxt_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
            daily_diffs[f'nbm_lead_{lead}_mint_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))

    ds_out = xr.Dataset(
        data_vars=daily_diffs,
        coords={'lat': grid_lats, 'lon': grid_lons}
    )
    ds_out.to_netcdf(nc_path, encoding={v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars})
    print(f"==> Saved cache: {nc_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill", type=int, default=1, help="Number of historical days to process")
    args = parser.parse_args()

    # Yesterday is the most recently completed full diurnal cycle
    base_target = pd.Timestamp.utcnow().floor('D') - pd.Timedelta(days=1)
    for d in range(args.backfill):
        process_date(base_target - pd.Timedelta(days=d))

    all_nc = sorted(glob.glob(f"{CACHE_DIR}/diff_*.nc"))
    if len(all_nc) > 25:
        for old_file in all_nc[:-25]:
            os.remove(old_file)
