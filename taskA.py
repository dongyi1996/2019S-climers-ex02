import os
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns
import utils


#------------------------------------------------------------------------------
# Task A: for each combination of EOBS and CMORPH/PERSIANN/TRMM-TMPA compute:
#   1) bias and correlation (spatial maps)
#   2) monthly total precipitation for the entire basin area (time series)
#------------------------------------------------------------------------------

def merged_pcp_to_netcdf():
    """
    Read the 4 precipitation data sets, do some cleaning,
    apply the danube river catchment mask and ultimately
    merge them into a single xr.Dataset object which is exported to disk.
    """
    # common time period
    startdate, enddate = '2002-01-01', '2010-12-31'

    # read mask
    ds_mask = xr.open_dataset(os.path.join(data_dir, 'mask.nc'))
    catchment_mask = ds_mask['mask']

    # TRMM
    # -------------------------------------------------------------------------
    ds_trmm = xr.open_dataset(os.path.join(
        os.path.join(data_dir, 'TRMM_TMPA_monthly_0d25.nc'))).rename(
        {'precipitation': 'TRMM'})

    da_trmm = (ds_trmm['TRMM']
               .transpose('time', 'lat', 'lon')  # -> concerns dims
               .sortby(['time', 'lat', 'lon'])
               .sel(time=slice(startdate, enddate))

               )
    # TODO: convert from mm/hr to mm/month by using the appropriate #days/month
    # For now assume a simple average: 24 (hours per day) x 30 (days per months)
    da_trmm = da_trmm * (24. * 30.)
    da_trmm.attrs['units'] = 'mm/month'
    utils.mask_dataarray(da_trmm, catchment_mask)

    # EOBS
    # -------------------------------------------------------------------------
    ds_eobs = xr.open_dataset(
        os.path.join(data_dir, 'EOBS_monthly_0d25.nc')).rename({'rr': 'EOBS'})
    da_eobs = (ds_eobs['EOBS']
               .sel(time=slice(startdate, enddate))
               )
    da_eobs['time'] = da_trmm['time']
    utils.mask_dataarray(da_eobs, catchment_mask)

    # PERSIANN
    # -------------------------------------------------------------------------
    ds_persiann = xr.open_dataset(
        os.path.join(data_dir, 'PERSIANN_monthly_0d25.nc')).rename(
        {'rain': 'PERSIANN'})
    da_persiann = (ds_persiann['PERSIANN']
                   .rename({'longitude': 'lon', 'latitude': 'lat'})
                   .transpose('time', 'lat', 'lon')  # -> concerns dims
                   .sortby(['time', 'lat', 'lon'])
                   .sel(time=slice(startdate, enddate))
                   )
    da_persiann['time'] = da_trmm['time']
    utils.mask_dataarray(da_persiann, catchment_mask)

    # CMORPH
    # -------------------------------------------------------------------------
    ds_cmorph = xr.open_dataset(
        os.path.join(data_dir, 'CMORPH_monthly_0d25.nc')).rename(
        {'cmorph': 'CMORPH'})
    da_cmorph = (ds_cmorph['CMORPH']
                 .drop('lev')
                 .transpose('time', 'lat', 'lon', 'lev')
                 .sortby(['time', 'lat', 'lon'])
                 .squeeze()
                 .sel(time=slice(startdate, enddate))
                 )
    da_cmorph['time'] = da_trmm['time']
    utils.mask_dataarray(da_cmorph, catchment_mask)

    # merge rainfall data into one xr.Dataset object and export
    # -------------------------------------------------------------------------
    ds_rain = xr.merge([da_eobs, da_trmm, da_cmorph, da_persiann])
    ds_rain = ds_rain.transpose('time', 'lat', 'lon')
    ds_rain.to_netcdf(path=os.path.join(data_dir, 'RAINFALL_MERGED_monthly_0d25.nc'))
    return None


def bias_plots():
    """
    Bias is the difference between the mean values. Compute mean over time
    for each pixel and subtract Satellite-based rainfall for each combination
    with EOBS.
    """
    # read and calc temporal mean
    # -------------------------------------------------------------------------
    ds_rain = xr.open_dataset(os.path.join(data_dir,
                                           'RAINFALL_MERGED_monthly_0d25.nc'))
    temporal_mean = ds_rain.mean('time')

    # Spatial BIAS plots
    # -------------------------------------------------------------------------

    # EOBS - CMORPH
    bias_cmorph = temporal_mean.EOBS - temporal_mean.CMORPH
    bias_cmorph.name = 'EOBS - CMORPH [mm]'
    bias_cmorph.plot(cmap='RdBu')
    plt.savefig(os.path.join(out_dir, 'taskA', 'BIAS_eobs_cmorph.png'),
                bbox_inches="tight")
    plt.close()

    # EOBS - PERSIANN
    bias_persiann = temporal_mean.EOBS - temporal_mean.PERSIANN
    bias_persiann.name = 'EOBS - PERSIANN [mm]'
    bias_persiann.plot(cmap='RdBu')
    plt.savefig(os.path.join(out_dir, 'taskA', 'BIAS_eobs_persiann.png'),
                bbox_inches="tight")
    plt.close()

    # EOBS - TRMM
    bias_trmm = temporal_mean.EOBS - temporal_mean.TRMM
    bias_trmm.name = 'EOBS - TRMM [mm]'
    bias_trmm.plot(cmap='RdBu')
    plt.savefig(os.path.join(out_dir, 'taskA', 'BIAS_eobs_trmm.png'),
                bbox_inches="tight")
    plt.close()

    # Compute basin average bias
    # -------------------------------------------------------------------------
    vals_bias_cmorph = bias_cmorph.to_dataframe()['EOBS - CMORPH [mm]'].values
    vals_bias_persiann = bias_persiann.to_dataframe()['EOBS - PERSIANN [mm]'].values
    vals_bias_trmm = bias_trmm.to_dataframe()['EOBS - TRMM [mm]'].values
    print(vals_bias_cmorph.shape, vals_bias_persiann.shape, vals_bias_trmm.shape)

    dict_of_biases = {'CMORPH': vals_bias_cmorph,
                 'TRMM': vals_bias_trmm,
                 'PERSIANN': vals_bias_persiann}

    df_biases = pd.DataFrame.from_dict(dict_of_biases)

    # Boxplot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots()
    sns.boxplot(data=df_biases, ax=ax)
    ax.set_ylabel("Bias [mm]")
    ax.axhline(0, linestyle='--', alpha=0.5, color='grey')

    plt.savefig(os.path.join(out_dir, 'taskA', 'BIAS_boxplot.png'),
                bbox_inches="tight")
    plt.close()

    # Violinplot (Boxplot + KDE)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots()
    sns.violinplot(data=df_biases)
    ax.set_ylabel("Bias [mm]")
    ax.axhline(0, linestyle='--', alpha=0.5, color='grey')

    plt.savefig(os.path.join(out_dir, 'taskA', 'BIAS_violinplot.png'),
                bbox_inches="tight")
    plt.close()


def corr_plots():
    """
    Compute correlation maps and boxplots
    """
    # read rainfall data
    # -------------------------------------------------------------------------
    ds_rain = xr.open_dataset(os.path.join(data_dir,
                                           'RAINFALL_MERGED_monthly_0d25.nc'))

    # ufuncs from http://xarray.pydata.org/en/stable/dask.html example
    # -------------------------------------------------------------------------
    def covariance_gufunc(x, y):
        return ((x - x.mean(axis=-1, keepdims=True))
                * (y - y.mean(axis=-1, keepdims=True))).mean(axis=-1)

    def pearson_correlation_gufunc(x, y):
        return covariance_gufunc(x, y) / (x.std(axis=-1) * y.std(axis=-1))

    # compute correlation maps
    # -------------------------------------------------------------------------
    dim = 'time'

    # EOBS - TRMM
    corr_trmm = xr.apply_ufunc(pearson_correlation_gufunc, # apply function
                          ds_rain.EOBS, ds_rain.TRMM, # arguments for the function
                          input_core_dims=[[dim], [dim]], # list of core dimensions on each input argument that should not be broadcast
                          dask='parallelized', # use dask?
                          output_dtypes=[float] # output dtype
                          )
    corr_trmm.name = 'r(EOBS, TRMM)'
    corr_trmm.plot(vmin=-1, vmax=1, cmap='RdBu')
    plt.savefig(os.path.join(out_dir, 'taskA', 'CORR_eobs_trmm.png'),
                bbox_inches="tight")
    plt.close()

    # EOBS - CMORPH
    corr_cmorph = xr.apply_ufunc(pearson_correlation_gufunc,  # apply function
                               ds_rain.EOBS, ds_rain.CMORPH,
                               # arguments for the function
                               input_core_dims=[[dim], [dim]],
                               # list of core dimensions on each input argument that should not be broadcast
                               dask='parallelized',  # use dask?
                               output_dtypes=[float]  # output dtype
                               )
    corr_cmorph.name = 'r(EOBS, CMORPH)'
    corr_cmorph.plot(vmin=-1, vmax=1, cmap='RdBu')
    plt.savefig(os.path.join(out_dir, 'taskA', 'CORR_eobs_cmorph.png'),
                bbox_inches="tight")
    plt.close()

    # EOBS - PERSIANN
    corr_persiann = xr.apply_ufunc(pearson_correlation_gufunc,  # apply function
                                 ds_rain.EOBS, ds_rain.PERSIANN,
                                 # arguments for the function
                                 input_core_dims=[[dim], [dim]],
                                 # list of core dimensions on each input argument that should not be broadcast
                                 dask='parallelized',  # use dask?
                                 output_dtypes=[float]  # output dtype
                                 )
    corr_persiann.name = 'r(EOBS, PERSIANN)'
    corr_persiann.plot(vmin=-1, vmax=1, cmap='RdBu')
    plt.savefig(os.path.join(out_dir, 'taskA', 'CORR_eobs_persiann.png'),
                bbox_inches="tight")
    plt.close()

    # Compute basin average correlations
    # -------------------------------------------------------------------------
    vals_corr_cmorph = corr_cmorph.to_dataframe()[corr_cmorph.name].values
    vals_corr_trmm = corr_trmm.to_dataframe()[corr_trmm.name].values
    vals_corr_persiann = corr_persiann.to_dataframe()[corr_persiann.name].values

    dict_of_corrs = {'CMORPH': vals_corr_cmorph,
                      'TRMM': vals_corr_trmm,
                      'PERSIANN': vals_corr_persiann}

    df_corrs = pd.DataFrame.from_dict(dict_of_corrs)

    # Boxplot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots()
    sns.boxplot(data=df_corrs, ax=ax)
    ax.set_ylim(0,1)
    ax.set_ylabel("Pearson's r [-]")

    plt.savefig(os.path.join(out_dir, 'taskA', 'CORR_boxplot.png'),
                bbox_inches="tight")
    plt.close()

    # Violinplot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots()
    sns.violinplot(data=df_corrs, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pearson's r [-]")

    plt.savefig(os.path.join(out_dir, 'taskA', 'CORR_violinplot.png'),
                bbox_inches="tight")
    plt.close()


def monthly_basin_avg_plot():
    """
    Plot monthly basin avg rainfall for all data sets in a time series plot.
    -> Spagetthi plot?
    """

    # read rainfall data
    # -------------------------------------------------------------------------
    ds_rain = xr.open_dataset(os.path.join(data_dir,
                                           'RAINFALL_MERGED_monthly_0d25.nc'))

    # compute monthly basin avg and stdv over time for each product and create plot
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15,8))

    alpha_std = .2

    # EOBS
    eobs_spatial_mean = ds_rain.EOBS.groupby('time').mean().to_dataframe()['EOBS']
    eobs_spatial_std = ds_rain.EOBS.groupby('time').std().to_dataframe()['EOBS']

    eobs_spatial_mean.plot(ax=ax)
    ax.fill_between(eobs_spatial_mean.index,
                    eobs_spatial_mean.values - eobs_spatial_std.values,
                    eobs_spatial_mean.values + eobs_spatial_std.values,
                    alpha=alpha_std)

    # CMORPH
    cmorph_spatial_mean = ds_rain.CMORPH.groupby('time').mean().to_dataframe()['CMORPH']
    cmorph_spatial_std = ds_rain.CMORPH.groupby('time').std().to_dataframe()['CMORPH']

    cmorph_spatial_mean.plot(ax=ax)
    ax.fill_between(cmorph_spatial_mean.index,
                    cmorph_spatial_mean.values - cmorph_spatial_std.values,
                    cmorph_spatial_mean.values + cmorph_spatial_std.values,
                    alpha=alpha_std)

    # TRMM
    trmm_spatial_mean = ds_rain.TRMM.groupby('time').mean().to_dataframe()['TRMM']
    trmm_spatial_std = ds_rain.TRMM.groupby('time').std().to_dataframe()['TRMM']

    trmm_spatial_mean.plot(ax=ax)
    ax.fill_between(trmm_spatial_mean.index,
                    trmm_spatial_mean.values - trmm_spatial_std.values,
                    trmm_spatial_mean.values + trmm_spatial_std.values,
                    alpha=alpha_std)


    # PERSIANN
    persiann_spatial_mean = ds_rain.PERSIANN.groupby('time').mean().to_dataframe()['PERSIANN']
    persiann_spatial_std = ds_rain.PERSIANN.groupby('time').std().to_dataframe()['PERSIANN']

    persiann_spatial_mean.plot(ax=ax)
    ax.fill_between(persiann_spatial_mean.index,
                    persiann_spatial_mean.values - persiann_spatial_std.values,
                    persiann_spatial_mean.values + persiann_spatial_std.values,
                    alpha=alpha_std)

    # labeling
    plt.title("Basin-average monthly rainfall")
    plt.ylabel('Rainfall [mm/month]')
    plt.ylim(0, 500)
    plt.legend()

    # save
    plt.savefig(os.path.join(out_dir, 'taskA', 'basin_avg_timeseries.png'),
                bbox_inches="tight")
    plt.close()

    # create scatter matrix plot for basin average
    # -------------------------------------------------------------------------
    # corr and stats for basin avg means
    basin_avg_means_dict = {'EOBS': eobs_spatial_mean,
                            'CMORPH': cmorph_spatial_mean,
                            'TRMM': trmm_spatial_mean,
                            'PERSIANN': persiann_spatial_mean}

    basin_avg_means = pd.DataFrame.from_dict(basin_avg_means_dict)

    # correlation matrix
    corr_basin_avg_means = basin_avg_means.corr(method='pearson')

    # create triangular mask for heatmap
    mask = np.zeros_like(corr_basin_avg_means)
    mask[np.triu_indices_from(mask)] = True

    # plot heatmap of pairwise correlations
    f, ax = plt.subplots(figsize=(8, 8))

    # define correct cbar height and pass to sns.heatmap function
    cbar_kws = {"fraction": 0.046, "pad": 0.04}
    sns.heatmap(corr_basin_avg_means, mask=mask, cmap='coolwarm_r', square=True,
                vmin=-1, vmax=1, annot=True,
                cbar_kws=cbar_kws, ax=ax)
    plt.title("Correlation for monthly basin-average rainfall")
    # save
    plt.savefig(os.path.join(out_dir, 'taskA', 'basin_avg_correlation_matrix.png'),
                bbox_inches="tight")
    plt.close()


if __name__ == '__main__':

    # directory setup
    # -------------------------------------------------------------------------

    # get directory containing this python file.
    project_dir = os.path.dirname(os.path.realpath(__file__))

    # input data directory
    data_dir = os.path.join(project_dir, 'data')

    # output directory
    out_dir = os.path.join(project_dir, 'results')

    # create the out_dir
    try:
        # try to create it
        os.makedirs(out_dir)
    except FileExistsError:
        # if the file already exists, it will throw a "FileExistsError". In that case, do nothing (pass)
        pass

    # merge rainfall data into one xr.Dataset object and export to .nc file
    # -------------------------------------------------------------------------
    merged_pcp_to_netcdf()

    # Bias plots
    # -------------------------------------------------------------------------
    bias_plots()

    # Corr plots
    # -------------------------------------------------------------------------
    corr_plots()

    # Monthly basin average rainfall comparison
    # -------------------------------------------------------------------------
    monthly_basin_avg_plot()