# -*- coding: utf-8 -*-
"""
Created on May 24 14:34 2019

@author: tstachl
"""

import os
import utils
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

if __name__ == '__main__':
    # get directory containing this python file.
    project_dir = os.path.dirname(os.path.realpath(__file__))

    # input data directory
    data_dir = os.path.join(project_dir, 'data')

    # output directory
    out_dir = os.path.join(project_dir, 'results')

    fname = os.path.join(data_dir, 'ESA_CCI_SM_monthly_anomalies_0d25.nc')
    da_sm = utils.read_nc(fname, 'SM_anomaly')

    fname = os.path.join(data_dir, 'GRACE_CSR_monthly_0d25.nc')
    da_tws = utils.read_nc(fname, 'lwe_thickness')