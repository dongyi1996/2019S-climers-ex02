# -*- coding: utf-8 -*-

import os
import utils
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import xarray as xr

def standardized_drought_index(ts, time_scale):
    """
    Calculate SPI/SPEI for a single P / P-ET time series.

    Das glätten muss vor dem schätzen der Verteilung gemacht werden.
    Die CDF Verteilung wird dann mit allen monaten minus jenen die vom
    glätten wegfallen für jedes monat seperat geschätzt und dann
    aber auf die gekürzte Zeitserie mittels PPF angewandt.

    Nicht gut implementiert - braucht eine monatlich gesampelte
    zeitserie und im letzten Jahr zumindest so viele Monate Daten
    wie der parameter time_scale gesetzt ist!
    """

    # store original datetime index
    index_orig = ts.index

    # apply rolling window
    # -------------------------------------------------------------------------
    if time_scale != 1:
        ts_smoothed = (ts
                       .rolling(window=time_scale,
                               min_periods=time_scale,
                               center=False)
                       .mean() # does not matter if mean or sum
                       .dropna())
    else:
        ts_smoothed = ts

    # apply gamma fit for each month group
    # -------------------------------------------------------------------------
    gamma_params = (ts_smoothed
                    .groupby(ts_smoothed.index.month)
                    .apply(stats.gamma.fit))

    # dict to store yearly ppf's per month
    ppf_per_month = {}

    # get ppf for each month separately
    for month in gamma_params.index.values:
        values_of_month = ts_smoothed[ts_smoothed.index.month == month]
        a, loc, scale = gamma_params[month]
        cdf = stats.gamma.cdf(values_of_month, a, loc, scale)
        ppf = stats.norm.ppf(cdf)
        if ppf.ndim != 1:
            ppf_per_month[month] = np.concatenate(ppf).ravel().tolist()
        else:
            ppf_per_month[month] = ppf.tolist()

    # pad values which are missing due to smoothing with np.nans
    # -------------------------------------------------------------------------
    len_dict = {}
    for k, v in ppf_per_month.items():
        len_dict[k] = len(v)

    max_len = np.array(list(len_dict.values())).max()

    for k, v in ppf_per_month.items():
        if len_dict[k] < max_len:
            ppf_per_month[k].insert(0, np.nan)

    # Reconstruction of time series (kann man sicher sauberer/flotter machen)
    # -------------------------------------------------------------------------
    # create a year index
    idx = np.arange(ts_smoothed.index.year[0],
                    ts_smoothed.index.year[-1] + 1)

    # ppf dict to data frame
    df = pd.DataFrame.from_dict(ppf_per_month).transpose()

    # reindex to years
    df.columns = idx
    df = df.transpose()

    # stack and reset the index
    df_stacked = df.stack().reset_index()

    # create index based on year and month multiindex
    new_idx = pd.to_datetime(
        df_stacked.level_0.astype('str') + '-' + df_stacked.level_1.astype(
            'str') + '-01',
        format='%Y-%m-%d')
    df_stacked = df_stacked.set_index(new_idx)
    df_stacked = df_stacked.drop(columns=['level_0', 'level_1'])
    df_stacked = df_stacked.rename(columns={0: 'Agg-{}'.format(time_scale)})

    return df_stacked.reindex(index_orig)


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


def plots_taskb(time_scales=[1, 3, 6, 12, 24, 36, 48]):
    """
    Plots of P, PET, AET and SPI, SPEI for the aggregation times
    of 3, 12 and 48 months
    """

    # read input data
    # -------------------------------------------------------------------------
    da_trmm = utils.read_nc(os.path.join(data_dir,
                                         'TRMM_TMPA_monthly_0d25.nc'),
                            'precipitation')
    da_trmm *= (30. * 24.)
    da_trmm.attrs['units'] = 'mm/month'

    fname = os.path.join(data_dir, 'GLEAM_PET_monthly_0d25.nc')
    da_pet = utils.read_nc(fname, 'PET')
    da_pet *= (30.)
    da_pet.attrs['units'] = 'mm/month'

    fname = os.path.join(data_dir, 'GLEAM_AET_monthly_0d25.nc')
    da_aet = utils.read_nc(fname, 'AET')
    da_aet *= (30.)
    da_aet.attrs['units'] = 'mm/month'

    # compute basin sums
    # -------------------------------------------------------------------------
    trmm_basin = da_trmm.groupby('time').sum().to_dataframe()
    trmm_basin = trmm_basin[:'2017-12-01']  # common period with GLEAM
    trmm_basin = trmm_basin.rename(columns={'precipitation': 'P'})

    pet_basin = da_pet.groupby('time').sum().to_dataframe()
    aet_basin = da_aet.groupby('time').sum().to_dataframe()

    basin_sums = pd.concat([trmm_basin, pet_basin, aet_basin], axis=1)

    # iterate over time scales and create plots
    for time_scale in time_scales:
        # apply moving window to basin sums
        # -------------------------------------------------------------------------
        basin_sums_smoothed = (basin_sums
                       .rolling(window=time_scale,
                                min_periods=time_scale,
                                center=False)
                       .mean()  # does not matter if mean or sum
                       .dropna())

        # calc SPI and SPEI
        # -------------------------------------------------------------------------
        spi = standardized_drought_index(trmm_basin, time_scale=time_scale)

        p_minus_et = basin_sums['P'] - basin_sums['PET']
        spei = standardized_drought_index(p_minus_et, time_scale=time_scale)

        # create plot
        # -------------------------------------------------------------------------
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10,6))

        # AX 0: Basin sums
        basin_sums_smoothed.plot(ax=axes[0])
        axes[0].set_title('Window size for aggregation: {} months'.format(time_scale))
        axes[0].set_ylabel('[mm]')

        # AX 1: SPI
        spi.plot(ax=axes[1], color='black', legend=False, ylim=(-3, 3))
        axes[1].set_title('SPI-{}'.format(time_scale))
        axes[1].set_ylabel('[-]')

        axes[1].axhline(2, linestyle='-.', color='grey', alpha=0.3)
        axes[1].axhline(1, linestyle='--', color='grey', alpha=0.4)
        axes[1].axhline(0, linestyle='-', color='darkgrey', alpha=0.5)
        axes[1].axhline(-1, linestyle='--', color='grey', alpha=0.4)
        axes[1].axhline(-2, linestyle='-.', color='grey', alpha=0.3)

        # AX 2: SPEI
        spei.plot(ax=axes[2], color='black', legend=False, ylim=(-3, 3))
        axes[2].set_title('SPEI-{}'.format(time_scale))
        axes[2].set_ylabel('[-]')

        axes[2].axhline(2, linestyle='-.', color='grey', alpha=0.3)
        axes[2].axhline(1, linestyle='--', color='grey', alpha=0.4)
        axes[2].axhline(0, linestyle='-', color='darkgrey', alpha=0.5)
        axes[2].axhline(-1, linestyle='--', color='grey', alpha=0.4)
        axes[2].axhline(-2, linestyle='-.', color='grey', alpha=0.3)

        # save figure
        plt.savefig(
            os.path.join(out_dir, 'taskB', 'felix',
                         'taskb_time_scale_{}.png'.format(time_scale)),
            bbox_inches="tight")
        plt.close()



if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # get directory containing this python file.
    project_dir = os.path.dirname(os.path.realpath(__file__))

    # input data directory
    data_dir = os.path.join(project_dir, 'data')

    # output directory
    out_dir = os.path.join(project_dir, 'results')

    # read input data
    # -------------------------------------------------------------------------
    """
    da_trmm = utils.read_nc(os.path.join(data_dir,
                                         'TRMM_TMPA_monthly_0d25.nc'),
                            'precipitation')
    da_trmm *= (30. * 24.)
    da_trmm.attrs['units'] = 'mm/month'

    fname = os.path.join(data_dir, 'GLEAM_PET_monthly_0d25.nc')
    da_pet = utils.read_nc(fname, 'PET')
    da_pet *= (30.)
    da_pet.attrs['units'] = 'mm/month'

    fname = os.path.join(data_dir, 'GLEAM_AET_monthly_0d25.nc')
    da_aet = utils.read_nc(fname, 'AET')
    da_aet *= (30.)
    da_aet.attrs['units'] = 'mm/month'
    """

    # task B plots
    # -------------------------------------------------------------------------
    plots_taskb()


    """
    aggr = '48M'

    # calc and plot SPI with TRMM
    spi = calc_spi(da_trmm)
    spi = spi.resample(aggr).mean()
    # plot_index(stacked_spi, 'SPI', ['TRMM'], aggr, True, out_dir)

    # calc and plot SPEI with TRMM and GLEAM PET
    spei = calc_spei(da_trmm, da_pet)
    spei = spei.resample(aggr).mean()
    # plot_index(stacked_spei, 'SPEI', ['TRMM', 'GLEAM'], aggr, True, out_dir)

    # plot_ts(da_trmm, aggr, out_dir)

    basin_mean_pet = da_pet.groupby('time').sum().to_dataframe().resample(aggr).mean()
    basin_mean_aet = da_aet.groupby('time').sum().to_dataframe().resample(aggr).mean()
    basin_mean_trmm = da_trmm.groupby('time').sum().to_dataframe().resample(aggr).mean()

    merged = pd.concat([basin_mean_pet, basin_mean_aet, basin_mean_trmm], axis=1)
    merged.plot()
    plt.title('Time series of Precipitation, PET and AET ({} sum)'.format(aggr))
    plt.ylabel('[mm/month]')
    plt.savefig(
        os.path.join(out_dir, 'taskB', '{}_sum_TRMM_PET_AET_ts.png'.format(aggr)),
        bbox_inches="tight")
    plt.show()
    plt.close()

    merged = pd.concat([spi, spei], axis=1)
    merged.plot()
    plt.title('Time series of SP(E)I ({} sum)'.format(aggr))
    plt.ylabel('SP(E)I')
    plt.savefig(
        os.path.join(out_dir, 'taskB', '{}_sum_SPI_SPEI_ts.png'.format(aggr)),
        bbox_inches="tight")
    plt.show()
    plt.close()
    """