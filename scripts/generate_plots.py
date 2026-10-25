import os
import glob
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import box
from shapely.ops import unary_union
import shapely
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

MAP_DIR = "docs/assets/maps"
CHART_DIR = "docs/assets/charts"
os.makedirs(MAP_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

files = sorted(glob.glob("data_cache/diff_*.nc"), reverse=True)
if not files:
    raise FileNotFoundError("No cache files found in data_cache/")

# ------------------------------------------------------------------------------
# 1. High-Resolution CONUS Land Mask Generator
# ------------------------------------------------------------------------------
def get_conus_land_mask(lats, lons):
    """
    Constructs a strict CONUS land mask using Natural Earth state geometries,
    filtering out oceans, Canada, Mexico, and non-CONUS territories.
    """
    mesh_lons, mesh_lats = np.meshgrid(lons, lats)
    try:
        shp_path = shpreader.natural_earth(resolution='50m', category='cultural', name='admin_1_states_provinces_lakes')
        reader = shpreader.Reader(shp_path)
        us_geoms = [
            rec.geometry for rec in reader.records()
            if rec.attributes.get('adm0_a3', rec.attributes.get('ADM0_A3')) == 'USA'
            and rec.attributes.get('name', '') not in ['Alaska', 'Hawaii', 'Puerto Rico']
        ]
        conus_geom = unary_union(us_geoms).intersection(box(-125.0, 24.0, -66.5, 50.0))
    except Exception:
        shp_path = shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries')
        reader = shpreader.Reader(shp_path)
        us_geom = [rec.geometry for rec in reader.records() if rec.attributes.get('ADM0_A3', rec.attributes.get('adm0_a3')) == 'USA'][0]
        conus_geom = us_geom.intersection(box(-125.0, 24.0, -66.5, 50.0))

    try:
        mask = shapely.contains_xy(conus_geom, mesh_lons, mesh_lats)
    except Exception:
        from shapely.prepared import prep
        from shapely.geometry import Point
        prep_geom = prep(conus_geom)
        mask = np.array([prep_geom.contains(Point(x, y)) for x, y in zip(mesh_lons.ravel(), mesh_lats.ravel())]).reshape(mesh_lons.shape)
        
    return mask

# ------------------------------------------------------------------------------
# 2. Spatial Bias & MAE Maps (Masked to CONUS Land)
# ------------------------------------------------------------------------------
def plot_spatial_comparison(diff_stack_aifs, diff_stack_nbm, lats, lons, conus_mask, lead, var_label, mode_label, metric, out_name):
    if metric == 'bias':
        grid_aifs = np.nanmean(diff_stack_aifs, axis=0)
        grid_nbm = np.nanmean(diff_stack_nbm, axis=0)
        cmap = 'bwr'
        vmin, vmax = -10, 10
        metric_label = "Mean Bias (°F)"
    else:
        grid_aifs = np.nanmean(np.abs(diff_stack_aifs), axis=0)
        grid_nbm = np.nanmean(np.abs(diff_stack_nbm), axis=0)
        cmap = 'YlOrRd'
        vmin, vmax = 0, 12
        metric_label = "Mean Absolute Error (°F)"

    # Apply Land Mask (oceans and foreign territories become transparent NaNs)
    grid_aifs = np.where(conus_mask, grid_aifs, np.nan)
    grid_nbm = np.where(conus_mask, grid_nbm, np.nan)

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    
    titles = [
        f"AIFS ({mode_label})\nDay {lead} {var_label} {metric_label}",
        f"NBM ({mode_label})\nDay {lead} {var_label} {metric_label}"
    ]
    
    for ax, data, t in zip(axs, [grid_aifs, grid_nbm], titles):
        ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, edgecolor='gray')
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax.set_title(t, fontsize=11, weight='bold')

    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f"Difference vs URMA (°F) [Window N={len(diff_stack_aifs)} Days]", fontsize=11)
    
    plt.savefig(os.path.join(MAP_DIR, out_name), dpi=130, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 3. Win Rate Map (% Days AIFS Error < NBM Error, Masked to CONUS Land)
# ------------------------------------------------------------------------------
def plot_winrate_map(diff_stack_aifs, diff_stack_nbm, lats, lons, conus_mask, lead, var_label, mode_label, out_name):
    aifs_abs = np.abs(diff_stack_aifs)
    nbm_abs = np.abs(diff_stack_nbm)

    valid_mask = (~np.isnan(aifs_abs)) & (~np.isnan(nbm_abs)) & conus_mask
    aifs_wins = (aifs_abs < nbm_abs) & valid_mask

    n_valid = np.sum(valid_mask, axis=0)
    n_wins = np.sum(aifs_wins, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        win_rate = np.where(conus_mask & (n_valid > 0), (n_wins / n_valid) * 100.0, np.nan)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4, edgecolor='gray')

    im = ax.pcolormesh(
        lons, lats, win_rate, 
        cmap='RdBu', 
        vmin=0, 
        vmax=100, 
        transform=ccrs.PlateCarree()
    )

    ax.set_title(
        f"AIFS vs. NBM Win Rate: % Days AIFS Absolute Error < NBM Absolute Error\n"
        f"Day {lead} {var_label} ({mode_label}) [CONUS Land | Sample N={len(diff_stack_aifs)} Days]",
        fontsize=12, weight='bold', pad=10
    )

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.06, shrink=0.7, ticks=[0, 20, 35, 50, 65, 80, 100])
    cbar.ax.set_xticklabels(['0% (100% NBM)', '20%', '35%', '50% (Tie)', '65%', '80%', '100% (100% AIFS)'])
    cbar.set_label("Blue = AIFS More Accurate | Red = NBM More Accurate", fontsize=10, weight='bold')

    plt.savefig(os.path.join(MAP_DIR, out_name), dpi=130, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 4. Average Diurnal Correction Map (Masked to CONUS Land)
# ------------------------------------------------------------------------------
def plot_delta_map(delta_stack, lats, lons, conus_mask, lead, var, out_name):
    avg_delta = np.nanmean(delta_stack, axis=0)
    avg_delta = np.where(conus_mask, avg_delta, np.nan)
    
    if var == 'maxt':
        title = f"Average Daytime Heating Boost Applied to AIFS (Day {lead} Max T)\nNBM [MaxT - max(18z,00z)] [CONUS Land | N={len(delta_stack)} Days]"
        cbar_lbl = "Average Diurnal Heating Boost Added (°F)"
        cmap = 'YlOrRd'
        vmin, vmax = 0, 8
    else:
        title = f"Average Nighttime Radiational Cooling Applied to AIFS (Day {lead} Min T)\nNBM [12z - MinT] [CONUS Land | N={len(delta_stack)} Days]"
        cbar_lbl = "Average Nocturnal Cooling Subtracted (°F)"
        cmap = 'PuBu'
        vmin, vmax = 0, 8

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.7)
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.4, edgecolor='gray')

    im = ax.pcolormesh(lons, lats, avg_delta, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=12, weight='bold', pad=10)

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.06, shrink=0.7)
    cbar.set_label(cbar_lbl, fontsize=10, weight='bold')

    plt.savefig(os.path.join(MAP_DIR, out_name), dpi=130, bbox_inches='tight')
    plt.close()

# ------------------------------------------------------------------------------
# 5. Regional Summary Line Charts (Strict Land Pixel Averaging)
# ------------------------------------------------------------------------------
def generate_regional_charts(window_files, conus_mask):
    leads = list(range(1, 9))
    datasets = [xr.open_dataset(f) for f in window_files]
    
    try:
        sample_ds = datasets[0]
        mesh_lons, _ = np.meshgrid(sample_ds.lon.values, sample_ds.lat.values)
        
        # Combined Land & Geographic Masks
        east_mask = conus_mask & (mesh_lons >= -105.0)

        domains = [
            ("CONUS Land", "conus", conus_mask),
            ("East of Rockies Land (Lon >= -105°W)", "east_rockies", east_mask)
        ]

        for domain_name, domain_slug, mask in domains:
            for mode, mode_label in [('proxy', 'Hourly Proxies'), ('corr', 'Corrected Extremes')]:
                for var in ['maxt', 'mint']:
                    var_name = "Max Temp" if var == 'maxt' else "Min Temp"
                    plt.figure(figsize=(10, 5))
                    
                    aifs_mae = []
                    nbm_mae = []
                    
                    for l in leads:
                        a_stack = [ds[f'aifs_lead_{l}_{var}_{mode}_diff'].values for ds in datasets if f'aifs_lead_{l}_{var}_{mode}_diff' in ds]
                        n_stack = [ds[f'nbm_lead_{l}_{var}_{mode}_diff'].values for ds in datasets if f'nbm_lead_{l}_{var}_{mode}_diff' in ds]
                        
                        if a_stack and n_stack:
                            a_arr = np.array(a_stack)
                            n_arr = np.array(n_stack)
                            # Slice exclusively over land pixels
                            a_val = np.nanmean(np.abs(a_arr[:, mask]))
                            n_val = np.nanmean(np.abs(n_arr[:, mask]))
                        else:
                            a_val, n_val = np.nan, np.nan
                        
                        aifs_mae.append(a_val)
                        nbm_mae.append(n_val)

                    lbl_aifs = 'AIFS Proxy (Ens Mean)' if mode == 'proxy' else 'AIFS + NBM Diurnal Boost'
                    lbl_nbm = 'NBM Proxy (Core 13z)' if mode == 'proxy' else 'NBM True Extreme (QMD 50%)'

                    plt.plot(leads, aifs_mae, marker='o', linewidth=2, color='#1f77b4', label=lbl_aifs)
                    plt.plot(leads, nbm_mae, marker='s', linewidth=2, color='#ff7f0e', label=lbl_nbm)

                    plt.title(f"{domain_name} MAE vs Lead Time: {var_name} ({mode_label})\nEvaluated Against URMA (Last {len(window_files)} Days)", fontsize=11, weight='bold')
                    plt.xlabel("Forecast Lead Day", fontsize=11)
                    plt.ylabel("Mean Absolute Error (°F)", fontsize=11)
                    plt.grid(True, linestyle='--', alpha=0.5)
                    plt.legend(fontsize=10)
                    plt.xticks(leads)
                    
                    plt.savefig(os.path.join(CHART_DIR, f"regional_mae_{domain_slug}_{var}_{mode}.png"), dpi=130, bbox_inches='tight')
                    plt.close()
    finally:
        for ds in datasets:
            ds.close()

# ------------------------------------------------------------------------------
# 6. Main Execution Loop
# ------------------------------------------------------------------------------
window_files = files[:min(14, len(files))]
with xr.open_dataset(files[0]) as ref:
    lats = ref.lat.values
    lons = ref.lon.values

# Compute CONUS Land Mask once
conus_mask = get_conus_land_mask(lats, lons)

for lead in range(1, 9):
    for var, var_label in [('maxt', 'Max T'), ('mint', 'Min T')]:
        for mode, mode_label in [('proxy', 'Hourly Proxy'), ('corr', 'Corrected Extreme')]:
            a_diffs = []
            n_diffs = []
            for f in window_files:
                try:
                    with xr.open_dataset(f) as ds:
                        k_a = f'aifs_lead_{lead}_{var}_{mode}_diff'
                        k_n = f'nbm_lead_{lead}_{var}_{mode}_diff'
                        if k_a in ds and k_n in ds:
                            a_diffs.append(ds[k_a].values)
                            n_diffs.append(ds[k_n].values)
                except Exception:
                    continue

            if a_diffs and n_diffs:
                for metric in ['bias', 'mae']:
                    out_name = f"day{lead}_{var}_{mode}_{metric}.png"
                    plot_spatial_comparison(a_diffs, n_diffs, lats, lons, conus_mask, lead, var_label, mode_label, metric, out_name)

                winrate_name = f"day{lead}_{var}_{mode}_winrate.png"
                plot_winrate_map(np.array(a_diffs), np.array(n_diffs), lats, lons, conus_mask, lead, var_label, mode_label, winrate_name)

        delta_stack = []
        for f in window_files:
            try:
                with xr.open_dataset(f) as ds:
                    kd = f'lead_{lead}_{var}_delta'
                    if kd in ds:
                        delta_stack.append(ds[kd].values)
            except Exception:
                continue
        if delta_stack:
            plot_delta_map(delta_stack, lats, lons, conus_mask, lead, var, f"day{lead}_{var}_delta.png")

generate_regional_charts(window_files, conus_mask)
print("All masked graphics and land-only statistics successfully generated.")
