# -*- coding: utf-8 -*-

import os
import utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr


if __name__ == '__main__':
    # set dirs
    # -------------------------------------------------------------------------
    project_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(project_dir, 'data')
    out_dir = os.path.join(project_dir, 'results')

    # read input data
    # -------------------------------------------------------------------------
    fname = os.path.join(data_dir, 'ESA_CCI_SM_monthly_anomalies_0d25.nc')
    da_sm = utils.read_nc(fname, 'SM_anomaly')

    fname = os.path.join(data_dir, 'GRACE_CSR_monthly_0d25.nc')
    da_tws = utils.read_nc(fname, 'lwe_thickness')

    # SPI and SPEI
    # -------------------------------------------------------------------------
    aggs = [3, 6, 12, 24, 36, 48]

    spis = []
    speis = []

    for agg in aggs:
        spi = utils.read_csv(
            infile=os.path.join(out_dir, 'taskB', 'methode_leander',
                                'SPI_time_scale_{}.csv'.format(agg)),
            parse_col='time').rename(
            columns={'Agg-{}'.format(agg): 'SPI-{}'.format(agg)})
        spis.append(spi)

        spei = utils.read_csv(
            infile=os.path.join(out_dir, 'taskB', 'methode_leander',
                                'SPEI_time_scale_{}.csv'.format(agg)),
            parse_col='time').rename(
            columns={'Agg-{}'.format(agg): 'SPEI-{}'.format(agg)})
        speis.append(spei)

    df_spis = pd.concat(spis, axis=1)
    df_speis = pd.concat(speis, axis=1)

    # discharge anomaly
    # -------------------------------------------------------------------------
    df_discharge = utils.read_grdc_discharge(data_dir)
    df_discharge.index.name = 'time'

    ds_discharge = df_discharge.to_xarray()
    climatology = ds_discharge.groupby('time.month').mean('time')

    anomalies = ds_discharge.groupby('time.month') - climatology
    print(anomalies)

    df_discharge_anoms = anomalies.to_dataframe()
    df_discharge_anoms = (df_discharge_anoms
                          .drop('month', axis=1)
                          .rename(columns={'harsova': 'Q_harsova',
                                           'ceatal_izmail': 'Q_ceatal_izmail'}))

    # aggregate spatial data sets
    # -------------------------------------------------------------------------
    sm_basin = da_sm.groupby('time').mean().to_series()
    sm_basin.name = 'SM'
    tws_basin = da_tws.groupby('time').mean().to_series()
    tws_basin.name = 'TWS'

    # merge everything
    # -------------------------------------------------------------------------
    df_merged = pd.concat([sm_basin, tws_basin,
                           df_spis, df_speis,
                           df_discharge_anoms], axis=1)

    # calc correlations and plot
    # -------------------------------------------------------------------------
    corrs = df_merged.corr(method='pearson')
    corrs.to_csv(os.path.join(out_dir, 'taskC', 'correlation_matrix_extended.csv'))

    # create triangular mask for heatmap
    mask = np.zeros_like(corrs)
    mask[np.triu_indices_from(mask)] = True

    # plot heatmap of pairwise correlations
    f, ax = plt.subplots(figsize=(8, 8))

    # define correct cbar height and pass to sns.heatmap function
    cbar_kws = {"fraction": 0.046, "pad": 0.04}
    sns.heatmap(corrs, mask=mask, cmap='coolwarm_r', square=True,
                vmin=-1, vmax=1, annot=True,
                cbar_kws=cbar_kws, ax=ax)
    # save
    plt.savefig(
        os.path.join(out_dir, 'taskC', 'correlation_matrix_extended.png'),
        bbox_inches="tight")
    plt.close()
