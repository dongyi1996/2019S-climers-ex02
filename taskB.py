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
    stacked = stacked.rename(columns={0: 'SPI'})

    return stacked


def plot_spi(stacked, var_name, aggr, save):

    SPI_ts = stacked['SPI']

    ax = SPI_ts.plot(title='SPI time series ({})'.format(aggr), figsize=(10, 5))
    ax.axhline(0, linestyle='--', color='grey', alpha=0.5)
    min = SPI_ts.min() * 1.1
    max = SPI_ts.max() * 1.1
    ax.set_ylim(min, max)
    plt.tight_layout()

    if save:
        SPI_ts.to_csv(os.path.join(out_dir, 'taskB', '{}_basin_avg_SPI_{}.csv'.format(aggr, var_name)), header=False)
        plt.savefig(
            os.path.join(out_dir, 'taskB', '{}_SPI_{}.png'.format(aggr, var_name)),
            bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def calc_spei(da):
    """
    calculate SPEI
    """
    # TODO: implement SPEI (P - ETp) -> should be the same apart from the input
    pass


if __name__ == '__main__':
    # get directory containing this python file.
    project_dir = os.path.dirname(os.path.realpath(__file__))

    # input data directory
    data_dir = os.path.join(project_dir, 'data')

    # output directory
    out_dir = os.path.join(project_dir, 'results')

    fname_merged = os.path.join(data_dir, 'RAINFALL_MERGED_monthly_0d25.nc')
    da_trmm = utils.read_nc(fname_merged, 'TRMM')

    """
    fname = os.path.join(data_dir, 'GLEAM_PET_monthly_0d25.nc')
    da_pet = utils.read_nc(fname, 'PET')
    da_pet = da_pet * (30.)
    da_pet.attrs['units'] = 'mm/month'

    fname = os.path.join(data_dir, 'GLEAM_AET_monthly_0d25.nc')
    da_aet = utils.read_nc(fname, 'AET')
    da_aet = da_aet * (30.)
    da_aet.attrs['units'] = 'mm/month'
    """
    stacked_trmm = calc_spi(da_trmm)

    aggr = '48M'
    stacked_trmm = stacked_trmm.resample(aggr).sum()

    plot_spi(stacked_trmm, 'TRMM', aggr, True)
