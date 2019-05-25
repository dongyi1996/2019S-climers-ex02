# -*- coding: utf-8 -*-
"""
Created on May 24 11:55 2019

@author: tstachl
"""
import os
import utils
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import xarray as xr


def calc_spi(da):
    """
    calculate SPI for given rainfall DataArray
    """
    # name of variable
    var_name = da.var().name

    # basin mean
    basin_mean = da.groupby('time').mean().to_dataframe()

    # apply gamma fit for each month group
    gammaparams = (basin_mean.groupby(basin_mean.index.month).apply(stats.gamma.fit))

    # dict to store yearly ppf's per month
    ppf_per_month = {}

    # get ppf for each month separately
    for month in gammaparams.index.values:
        values_of_month = basin_mean[var_name][basin_mean.index.month == month]
        a, loc, scale = gammaparams[month]
        cdf = stats.gamma.cdf(values_of_month, a, loc, scale)
        ppf = stats.norm.ppf(cdf)
        ppf_per_month[month] = ppf

    # Reconstruction of time series (kann man sicher sauberer/flotter machen)
    # -------------------------------------------------------------------------

    # create a year index
    idx = np.arange(basin_mean.index.year[0], basin_mean.index.year[-1] + 1)

    # ppf dict to data frame
    df = pd.DataFrame.from_dict(ppf_per_month).transpose()

    # reindex to years
    df.columns = idx
    df = df.transpose()

    # stack and reset the index
    stacked = df.stack().reset_index()

    # create index based on year and month multiindex
    new_idx = pd.to_datetime(stacked.level_0.astype('str') + '-' + stacked.level_1.astype('str') + '-01',
                             format='%Y-%m-%d')
    stacked = stacked.set_index(new_idx)
    stacked = stacked.drop(columns=['level_0', 'level_1'])
    stacked = stacked.rename(columns={0: 'SPI'})

    return stacked


def calc_spei(da_prec, da_pet):
    """
    calculate SPEI, which is difference between precipitation
    and potential evapotranspiration
    """
    # name of variable
    var_name_prec = da_prec.var().name
    var_name_pet = da_pet.var().name

    # only select pet data where also prec data is available
    da_pet = da_pet.sel(time=da_prec.time)

    basin_mean_prec = da_prec.groupby('time').mean().to_dataframe()
    basin_mean_pet = da_pet.groupby('time').mean().to_dataframe()

    merged = pd.concat([basin_mean_prec, basin_mean_pet], axis=1).dropna()
    merged['DIFF'] = merged[var_name_prec] - merged[var_name_pet]

    diff = merged['DIFF']

    gammaparams = (diff.groupby(diff.index.month).apply(stats.gamma.fit))
    ppf_per_month = {}

    # get ppf for each month separately
    for month in gammaparams.index.values:
        values_of_month = diff[diff.index.month == month]
        a, loc, scale = gammaparams[month]
        cdf = stats.gamma.cdf(values_of_month, a, loc, scale)
        ppf = stats.norm.ppf(cdf)
        ppf_per_month[month] = ppf

    # Reconstruction of time series (kann man sicher sauberer/flotter machen)
    # -------------------------------------------------------------------------

    # create a year index
    idx = np.arange(diff.index.year[0], diff.index.year[-1] + 1)

    # ppf dict to data frame
    df = pd.DataFrame.from_dict(ppf_per_month).transpose()

    # reindex to years
    df.columns = idx
    df = df.transpose()

    # stack and reset the index
    stacked = df.stack().reset_index()

    # create index based on year and month multiindex
    new_idx = pd.to_datetime(stacked.level_0.astype('str') + '-' + stacked.level_1.astype('str') + '-01',
                             format='%Y-%m-%d')
    stacked = stacked.set_index(new_idx)
    stacked = stacked.drop(columns=['level_0', 'level_1'])
    stacked = stacked.rename(columns={0: 'SPEI'})

    return stacked


def plot_index(stacked, index_name, var_names, aggr, save, out_dir):

    ts = stacked[index_name]

    out_dir = os.path.join(out_dir, 'taskB', index_name)
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    ax = ts.plot(title='{} time series ({})'.format(index_name, aggr), figsize=(10, 5))
    ax.axhline(0, linestyle='--', color='grey', alpha=0.5)
    min = ts.min() * 1.1
    max = ts.max() * 1.1
    ax.set_ylim(min, max)
    plt.tight_layout()

    if len(var_names) > 1:
        label_var_names = '_'.join(var_names)
    else:
        label_var_names = var_names[0]

    if save:
        ts.to_csv(os.path.join(out_dir, '{}_basin_avg_{}_{}.csv'.format(aggr, index_name, label_var_names)),
                      header=False)
        plt.savefig(
            os.path.join(out_dir, '{}_{}_{}.png'.format(aggr, index_name, label_var_names)),
            bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_ts(da, aggr, out_dir):

    out_dir = os.path.join(out_dir, 'taskB')
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    var_name = da.var().name
    basin_mean = da.groupby('time').mean().to_dataframe()
    basin_mean = basin_mean.resample(aggr).sum()

    ax = basin_mean.plot(title='{} time series ({})'.format(var_name, aggr), figsize=(10, 5))
    ax.axhline(0, linestyle='--', color='grey', alpha=0.5)
    min = basin_mean.min().values * 1.1
    max = basin_mean.max().values * 1.1
    ax.set_ylim(min, max)
    ax.set_ylabel('{} [mm/month]'.format(var_name))
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, '{}_{}_ts.png'.format(aggr, var_name)),
        bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    # get directory containing this python file.
    project_dir = os.path.dirname(os.path.realpath(__file__))

    # input data directory
    data_dir = os.path.join(project_dir, 'data')

    # output directory
    out_dir = os.path.join(project_dir, 'results')

    fname_merged = os.path.join(data_dir, 'RAINFALL_MERGED_monthly_0d25.nc')
    da_trmm = utils.read_nc(fname_merged, 'TRMM')

    fname = os.path.join(data_dir, 'GLEAM_PET_monthly_0d25.nc')
    da_pet = utils.read_nc(fname, 'PET')
    da_pet = da_pet * (30.)
    da_pet.attrs['units'] = 'mm/month'

    fname = os.path.join(data_dir, 'GLEAM_AET_monthly_0d25.nc')
    da_aet = utils.read_nc(fname, 'AET')
    da_aet = da_aet * (30.)
    da_aet.attrs['units'] = 'mm/month'

    aggr = '48M'

    # calc and plot SPI with TRMM
    stacked_spi = calc_spi(da_trmm)
    stacked_spi = stacked_spi.resample(aggr).sum()
    # plot_index(stacked_spi, 'SPI', ['TRMM'], aggr, True, out_dir)

    # calc and plot SPEI with TRMM and GLEAM PET
    stacked_spei = calc_spei(da_trmm, da_pet)
    stacked_spei = stacked_spei.resample(aggr).sum()
    # plot_index(stacked_spei, 'SPEI', ['TRMM', 'GLEAM'], aggr, True, out_dir)

    # plot_ts(da_trmm, aggr, out_dir)

    basin_mean_pet = da_pet.groupby('time').mean().to_dataframe().resample(aggr).sum()
    basin_mean_aet = da_aet.groupby('time').mean().to_dataframe().resample(aggr).sum()
    basin_mean_trmm = da_trmm.groupby('time').mean().to_dataframe().resample(aggr).sum()

    merged = pd.concat([basin_mean_pet, basin_mean_aet, basin_mean_trmm], axis=1)
    merged.plot()
    plt.title('Time series of Precipitation, PET and AET ({} sum)'.format(aggr))
    plt.ylabel('[mm/month]')
    plt.savefig(
        os.path.join(out_dir, 'taskB', '{}_sum_TRMM_PET_AET_ts.png'.format(aggr)),
        bbox_inches="tight")
    plt.show()
    plt.close()

    merged = pd.concat([stacked_spi, stacked_spei], axis=1)
    merged.plot()
    plt.title('Time series of SP(E)I ({} sum)'.format(aggr))
    plt.ylabel('SP(E)I')
    plt.savefig(
        os.path.join(out_dir, 'taskB', '{}_sum_SPI_SPEI_ts.png'.format(aggr)),
        bbox_inches="tight")
    plt.show()
    plt.close()




