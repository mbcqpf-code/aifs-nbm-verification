import os
import glob
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

MAP_DIR = "docs/assets/maps"
CHART_DIR = "docs/assets/charts"
os.makedirs(MAP_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

files = sorted(glob.glob("data_cache/diff_*.nc"), reverse=True)
if not files:
    raise FileNotFoundError("No cache files found in data_cache/")

def plot_spatial_comparison(diff_stack_aifs, diff_stack_nbm, lats, lons, lead, var_label, metric, out_name):
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

    fig, axs = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    for ax, data, model_title in zip(axs, [grid_aifs, grid_nbm], ["AIFS Ensemble Mean (Proxy)", "NBM 06z (13z Operational)"]):
        ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.6)
        ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, edgecolor='gray')
        im = ax.pcolormesh(lons, lats, data, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        ax.set_title(f"{model_title}\nDay {lead} {var_label} {metric_label}", fontsize=11, weight='bold')

    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f"Difference vs URMA (°F) [Window N={len(diff_stack_aifs)}]", fontsize=11)

    plt.savefig(os.path.join(MAP_DIR, out_name), dpi=130, bbox_inches='tight')
    plt.close()

def generate_regional_charts(window_files):
    leads = list(range(1, 9))
    with xr.open_mfdataset(window_files, combine='nested', concat_dim='time') as ds:
        for var in ['maxt', 'mint']:
            var_name = "Max Temp" if var == 'maxt' else "Min Temp"
            plt.figure(figsize=(10, 5))

            aifs_mae = [np.nanmean(np.abs(ds[f'aifs_lead_{l}_{var}_diff'].values)) for l in leads]
            nbm_mae = [np.nanmean(np.abs(ds[f'nbm_lead_{l}_{var}_diff'].values)) for l in leads]

            plt.plot(leads, aifs_mae, marker='o', linewidth=2, color='#1f77b4', label='AIFS Proxy (Ens Mean)')
            plt.plot(leads, nbm_mae, marker='s', linewidth=2, color='#ff7f0e', label='NBM QMD (13z Median)')

            plt.title(f"CONUS Average MAE vs Lead Time: {var_name} (Last {len(window_files)} Days)", fontsize=13, weight='bold')
            plt.xlabel("Forecast Lead Day", fontsize=11)
            plt.ylabel("Mean Absolute Error (°F)", fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(fontsize=11)
            plt.xticks(leads)

            plt.savefig(os.path.join(CHART_DIR, f"regional_mae_conus_{var}.png"), dpi=130, bbox_inches='tight')
            plt.close()

# Use up to 7 recent days for rolling verification
window_files = files[:min(7, len(files))]
with xr.open_dataset(files[0]) as ref:
    lats = ref.lat.values
    lons = ref.lon.values

for lead in range(1, 9):
    for var, var_label in [('maxt', 'Max T'), ('mint', 'Min T')]:
        aifs_diffs = []
        nbm_diffs = []
        for f in window_files:
            with xr.open_dataset(f) as ds:
                aifs_diffs.append(ds[f'aifs_lead_{lead}_{var}_diff'].values)
                nbm_diffs.append(ds[f'nbm_lead_{lead}_{var}_diff'].values)

        for metric in ['bias', 'mae']:
            out_filename = f"day{lead}_{var}_{metric}.png"
            plot_spatial_comparison(
                aifs_diffs, nbm_diffs, lats, lons, 
                lead, var_label, metric, out_filename
            )

generate_regional_charts(window_files)
print("Maps and charts successfully generated.")
