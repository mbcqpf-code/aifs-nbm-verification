import os
import glob
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
# 1. Target Grid Setup (0.25° CONUS Mesh)
# ------------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 24.0, 50.0
LON_MIN, LON_MAX = -125.0, -65.0
grid_lats = np.arange(LAT_MIN, LAT_MAX + 0.25, 0.25)
grid_lons = np.arange(LON_MIN, LON_MAX + 0.25, 0.25)
TARGET_LONS, TARGET_LATS = np.meshgrid(grid_lons, grid_lats)

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. Curvilinear Regridder (For NBM and URMA)
# ------------------------------------------------------------------------------
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
# 3. Direct AIFS Ensemble Mean Retrieval (type="em")
# ------------------------------------------------------------------------------
def get_aifs_em_t2m(init_date_str, fxx):
    """
    Retrieves the native AIFS Ensemble Mean (type="em") 2m temperature directly.
    Tries Azure (full archive) first, then AWS, then ECMWF open data.
    """
    # init_date_str is YYYY-MM-DD
    dt_clean = init_date_str.replace("-", "")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
        tmp_target = tmp.name

    success = False
    for source in ["azure", "aws", "ecmwf"]:
        try:
            client = Client(source=source, model="aifs-ens", resol="0p25")
            client.retrieve(
                date=dt_clean,
                time=0,
                step=fxx,
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
        raise RuntimeError(f"Could not retrieve AIFS ENS mean for date={dt_clean} step={fxx}")

    ds = xr.open_dataset(tmp_target, engine='cfgrib')
    var_key = 't2m' if 't2m' in ds else list(ds.data_vars)[0]
    field = ds[var_key]

    if field.longitude.max() > 180:
        field = field.assign_coords(longitude=(((field.longitude + 180) % 360) - 180))
    field = field.sortby(['latitude', 'longitude'])

    field_interp = field.interp(latitude=grid_lats, longitude=grid_lons, method='linear')
    t2m_k = field_interp.squeeze().values
    
    ds.close()
    for f in glob.glob(f"{tmp_target}*"): 
        try: os.remove(f)
        except Exception: pass
        
    return t2m_k

def get_aifs_proxies(init_date_str, lead_day):
    """Min T = 12z proxy; Max T = max(18z, 00z) proxy."""
    fxx_min = lead_day * 24 - 12
    fxx_max1 = lead_day * 24 - 6
    fxx_max2 = lead_day * 24

    t_min_k = get_aifs_em_t2m(init_date_str, fxx_min)
    t_max1_k = get_aifs_em_t2m(init_date_str, fxx_max1)
    t_max2_k = get_aifs_em_t2m(init_date_str, fxx_max2)

    t_max_k = np.maximum(t_max1_k, t_max2_k)
    return (t_min_k - 273.15) * 9/5 + 32, (t_max_k - 273.15) * 9/5 + 32

# ------------------------------------------------------------------------------
# 4. NBM & URMA Helpers
# ------------------------------------------------------------------------------
def get_nbm_qmd(init_dt, lead_hrs, var_type):
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
            if var_type == 'max' and "max" in line_l:
                start_byte = int(line.split(':')[1])
                if i + 1 < len(lines): end_byte = int(lines[i+1].split(':')[1]) - 1
                break
            elif var_type == 'min' and "min" in line_l:
                start_byte = int(line.split(':')[1])
                if i + 1 < len(lines): end_byte = int(lines[i+1].split(':')[1]) - 1
                break
                
    if start_byte is None:
        raise ValueError(f"Could not locate 50% {var_type} in {grib_name}.idx")
        
    headers = {"Range": f"bytes={start_byte}-{end_byte}" if end_byte else f"bytes={start_byte}-"}
    r = requests.get(f"{base_url}/{grib_name}", headers=headers, timeout=30)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
        tmp.write(r.content)
        tmp_path = tmp.name
        
    ds = xr.open_dataset(tmp_path, engine='cfgrib')
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

def get_urma(valid_date_str):
    H_max = Herbie(f"{valid_date_str} 08:00", model="urma", product="anl")
    ds_max = H_max.xarray(":TMAX:2 m above ground:")
    t_max_data = ds_max[list(ds_max.data_vars)[0]].squeeze().values[::2, ::2]
    t_max = regrid_to_conus(t_max_data, ds_max.latitude.values[::2, ::2], ds_max.longitude.values[::2, ::2])

    H_min = Herbie(f"{valid_date_str} 20:00", model="urma", product="anl")
    ds_min = H_min.xarray(":TMIN:2 m above ground:")
    t_min_data = ds_min[list(ds_min.data_vars)[0]].squeeze().values[::2, ::2]
    t_min = regrid_to_conus(t_min_data, ds_min.latitude.values[::2, ::2], ds_min.longitude.values[::2, ::2])

    return (t_min - 273.15) * 9/5 + 32, (t_max - 273.15) * 9/5 + 32

# ------------------------------------------------------------------------------
# 5. Daily Processing Loop
# ------------------------------------------------------------------------------
def process_date(target_dt):
    date_str = target_dt.strftime('%Y-%m-%d')
    nc_path = os.path.join(CACHE_DIR, f"diff_{date_str}.nc")
    if os.path.exists(nc_path):
        print(f"[{date_str}] Already exists. Skipping.")
        return

    print(f"\n--- Processing Valid Date: {date_str} ---")
    try:
        urma_min_f, urma_max_f = get_urma(date_str)
    except Exception as e:
        print(f"  [!] URMA unavailable for {date_str}: {e}. Skipping date.")
        return

    daily_diffs = {}
    for lead in range(1, 9):
        init_date = target_dt - pd.Timedelta(days=lead)
        init_date_str = init_date.strftime('%Y-%m-%d')
        init_06z = pd.to_datetime(f"{init_date_str} 06:00")
        
        # AIFS Direct Mean
        try:
            aifs_min, aifs_max = get_aifs_proxies(init_date_str, lead)
            daily_diffs[f'aifs_lead_{lead}_maxt_diff'] = (['lat', 'lon'], (aifs_max - urma_max_f).astype(np.float32))
            daily_diffs[f'aifs_lead_{lead}_mint_diff'] = (['lat', 'lon'], (aifs_min - urma_min_f).astype(np.float32))
            print(f"  -> Day {lead} AIFS (Direct Mean) processed.")
        except Exception as e:
            print(f"  [!] AIFS Day {lead} failed: {e}")
            daily_diffs[f'aifs_lead_{lead}_maxt_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))
            daily_diffs[f'aifs_lead_{lead}_mint_diff'] = (['lat', 'lon'], np.full(TARGET_LATS.shape, np.nan, dtype=np.float32))

        # NBM QMD Median
        try:
            nbm_max = get_nbm_qmd(init_06z, lead * 24, 'max')
            nbm_min = get_nbm_qmd(init_06z, lead * 24 + 12, 'min')
            daily_diffs[f'nbm_lead_{lead}_maxt_diff'] = (['lat', 'lon'], (nbm_max - urma_max_f).astype(np.float32))
            daily_diffs[f'nbm_lead_{lead}_mint_diff'] = (['lat', 'lon'], (nbm_min - urma_min_f).astype(np.float32))
            print(f"  -> Day {lead} NBM processed.")
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

    base_target = pd.Timestamp.utcnow().floor('D') - pd.Timedelta(days=1)
    for d in range(args.backfill):
        process_date(base_target - pd.Timedelta(days=d))

    all_nc = sorted(glob.glob(f"{CACHE_DIR}/diff_*.nc"))
    if len(all_nc) > 25:
        for old_file in all_nc[:-25]:
            os.remove(old_file)
