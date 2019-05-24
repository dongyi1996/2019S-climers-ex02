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

    trmm_basin_mean = da_trmm.groupby('time').mean().to_dataframe()
    trmm_gammaparams = trmm_basin_mean.groupby(trmm_basin_mean.index.month).apply(
        stats.gamma.fit)

    x = np.linspace(0, 200, 1000)
    x2 = np.linspace(0,1,1000)
    for month in trmm_gammaparams.index.values[:1]:
        cdf = stats.gamma.cdf(x, trmm_gammaparams[month][0], trmm_gammaparams[month][1], trmm_gammaparams[month][2])
        ppf = stats.norm.ppf(cdf)
        plt.plot(ppf, x2, label='{}'.format(month))
    plt.legend()
    plt.title('TRMM - monthly PPFs')
    plt.show()
    #plt.xlabel('precipitation [mm/month]')
    #plt.ylabel('Commulative probability')
    #
    # os.makedirs(os.path.join(out_dir, 'taskB'))
    # plt.savefig(os.path.join(out_dir, 'taskB', 'cdfs_trmm.png'),
    #             bbox_inches="tight")
    # plt.close()



