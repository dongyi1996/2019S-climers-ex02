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
    # TODO: maybe implement function for spi to work with different rainfall data sets
    pass

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

    # basin mean
    trmm_basin_mean = da_trmm.groupby('time').mean().to_dataframe()

    # apply gamma fit for each month group
    trmm_gammaparams = (trmm_basin_mean
        .groupby(trmm_basin_mean.index.month)
        .apply(stats.gamma.fit))

    # dict to store yearly ppf's per month
    ppf_per_month = {}

    # get ppf for each month separately
    for month in trmm_gammaparams.index.values:
        values_of_month = trmm_basin_mean['TRMM'][trmm_basin_mean.index.month == month]
        cdf = stats.gamma.cdf(values_of_month, trmm_gammaparams[month][0], trmm_gammaparams[month][1], trmm_gammaparams[month][2])
        ppf = stats.norm.ppf(cdf)
        ppf_per_month[month] = ppf

    # Reconstruction of time series (kann man sicher sauberer/flotter machen)
    # -------------------------------------------------------------------------

    # create a year index
    idx = np.arange(trmm_basin_mean.index.year[0], trmm_basin_mean.index.year[-1] + 1)

    # ppf dict to data frame
    df = pd.DataFrame.from_dict(ppf_per_month).transpose()

    # reindex to years
    df.columns = idx
    df = df.transpose()

    # stack and reset the index
    stacked = df.stack().reset_index()

    # create index based on year and month multiindex
    new_idx = pd.to_datetime(stacked.level_0.astype('str') + '-' + stacked.level_1.astype('str') + '-01', format='%Y-%m-%d')
    stacked = stacked.set_index(new_idx)
    stacked = stacked.rename(columns={0: 'SPI'})

    # extract SPI and save as .csv
    SPI_ts = stacked['SPI']
    SPI_ts.to_csv(os.path.join(out_dir, 'taskB', 'monthly_basin_avg_SPI_TRMM.csv'), header=False)

    # plot monthly SPI
    ax = SPI_ts.plot(title='SPI time series (1-month)', figsize=(10,5))
    ax.axhline(0, linestyle='--', color='grey', alpha=0.5)
    ax.set_ylim(-4,4)
    plt.tight_layout()

    # save figure
    plt.savefig(
        os.path.join(out_dir, 'taskB', 'SPI_TRMM_months_1.png'),
        bbox_inches="tight")
    plt.close()

    # Aggregated SPI
    # -------------------------------------------------------------------------
    # TODO: ?