import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import statsmodels.api as sm


def calc_lagmeans(s, lags):
    """
    Calculates for each observation the mean of the previous x months, for each x in lags
    Assumes that the input series has evenly spaced temporal intervals
    :param s: pandas.Series
        Time series of which to calculate lag avarages of.
    :param lags: list
        list of lags for which to calculate the average for.
    :return: pandas.Dataframe
        data frame with a column for each lag
        The columns are padded at the beginning such that they all have the length of the input series
    """
    csum = s.cumsum().values
    csum = np.insert(csum, 0, 0)
    out_df = pd.DataFrame(index=s.index)
    for lag in lags:
        lagsum = csum[lag:] - csum[:-lag]
        lagsum = np.insert(lagsum, 0, np.nan * np.ones(lag-1))
        out_df['lagmean_' + str(lag)] = lagsum/lag
    return out_df


def read_grdc_discharge(data_dir):
    """
    Read the station discharge measurements
    :param data_dir: string,
        directory containing the data files
    :return: pandas.Dataframe
        data of the two stations
    """
    setup_runoff = {'harsova': '6742800_Q_Month.txt',
                    'ceatal_izmail': '6742900_Q_Month.txt'}
    df_r = pd.DataFrame()
    for var in setup_runoff.keys():
        runoff = pd.read_csv(os.path.join(data_dir, setup_runoff[var]),
                             skiprows=37, delimiter=";", parse_dates=True,
                             usecols=[0, 3], index_col=0)
        runoff.rename(columns={' Calculated': var}, inplace=True)
        runoff = runoff.truncate(before=datetime(2002, 1, 1, 0)).copy()
        runoff[runoff == -999.] = np.nan
        df_r = pd.concat([df_r, runoff], axis=1)
    
    return df_r


def mask_dataarray(da, mask):
    """
    Mask one data array with another one
    :param da: xarray.Dataarray
        data array to be masked
    :param mask: xarray.Dataarray
        data array containing mask (0/1)
    :return:
        "Nothing", it alters the input dataarray inplace
    """
    data = da.values
    data[:, mask.values == 0] = np.nan
    da.values = data


def data_overview(data_dir, out_dir):
    """
    This is an example function whose main purpose is to showcase how to do some common things in python
        that you most likely will need. Scavenge it for useful parts!
    """

    """
    First Part: loading, preparing, masking and subsetting the data
    """
    # Name of file we are going to read. For this example, we are looking at the TRMM TMPA data that you need in all
    #   three tasks.
    prec_data_fname = os.path.join(data_dir, 'TRMM_TMPA_monthly_0d25.nc')
    # open the netcdf files with xarray. xarray is great at handling data with 3 or more dimensions
    #   (here: time, lon, lat), while pandas might be easier to use for 1d or 2d data.
    trmm_tmpa_ds = xr.open_dataset(prec_data_fname)
    # print the dataset to get a summary of it
    print(trmm_tmpa_ds)
    # get the precipitation data and store it as dataarray
    prec_da = trmm_tmpa_ds['precipitation']
    # change the dimension order to be the same as the one of the danube catchment mask loaded later
    prec_da = prec_da.transpose('time', 'lat', 'lon')
    # in some files some dimension values are descending. TRMM_TMPA is already sorted, so here it does nothing
    prec_da = prec_da.sortby(['time', 'lat', 'lon'])
    # open the mask of the danube area
    mask_ds = xr.open_dataset(os.path.join(data_dir, 'mask.nc'))
    # get the mask as dataarray
    mask_da = mask_ds['mask']
    # set the precipitation data outside the mask to nan
    mask_dataarray(prec_da, mask_da)
    # select temporal subset of the precipitation data
    prec_temp_subset = prec_da.loc['2002-01-01':'2010-12-31', :, :]
    # load the river discharge data
    grdc = read_grdc_discharge(data_dir)
    # select temporal subset of the grdc data
    grdc_subset = grdc.loc['2002-01-01':'2010-12-31']

    """
    Part 2: Basic xarray data handling
    """
    # take the mean in the time dimension to get the average rainfall for each locaton
    prec_temporal_mean = prec_temp_subset.mean('time')
    # plot the average rainfall map and save it
    prec_temporal_mean.plot()
    plt.axis('equal')
    plt.title('mean_rainfall_trmm_tmpa')
    plt.savefig(os.path.join(out_dir, 'mean_rainfall_trmm_tmpa.png'), bbox_inches="tight")
    plt.close()

    # lets take the average of each month of the year.
    prec_monthly_sum = prec_temp_subset.groupby('time.month').mean('time')
    # lets plot the mean june precipitation
    prec_june = prec_monthly_sum[5, :, :]
    prec_june.plot()
    plt.axis('equal')
    plt.savefig(os.path.join(out_dir, 'mean_rainfall_june_trmm_tmpa.png'), bbox_inches="tight")
    plt.close()
    # you can also average multiple axes at once. Lets average over lat/lon to
    # create a single time series for the danube catchment
    prec_ts = prec_temp_subset.mean(('lon', 'lat'))
    prec_ts.plot(label='catchment mean')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'mean_rainfall_ts_trmm_tmpa.png'), bbox_inches="tight")
    plt.close()

    """
    Part 3: Calculate the means for different temporal lags
    """
    # to create the SPI/SPEI you are going to need the averages (or sums) for different temporal lags
    # we have prepared a function, calc_lagmeans(),  which does that for you
    # Example use:
    # define the lags for which we want to calculate the lag averages
    lags = [3, 12, 48]
    # xarray dataarray -> pandas.series
    prec_ts_series = prec_ts.to_series()
    # calculate the averages for each temporal lag.
    # Each value represents the average of the 3, 12 or 48 months of precipitation before it
    lagmeans = calc_lagmeans(prec_ts_series, lags)
    # plot it
    lagmeans.plot()
    plt.xlabel('Time')
    plt.ylabel('precipitation [mm/hr]')
    plt.savefig(os.path.join(out_dir, 'trmm_tmpa_precip_lag_averages.png'), bbox_inches="tight")
    plt.close()


def bias(data_dir):
    # Mask
    mask_fname = os.path.join(data_dir, 'mask.nc')
    mask_ds = xr.open_dataset(mask_fname)
    mask_da = mask_ds['mask']

    # E-OBS dataset
    eobs_fname = os.path.join(data_dir, 'EOBS_monthly_0d25.nc')
    eobs_ds = xr.open_dataset(eobs_fname)
    eobs_prec = eobs_ds['rr']
    eobs_prec = eobs_prec.transpose('time', 'lat', 'lon')
    eobs_prec = eobs_prec.sortby(['time', 'lat', 'lon'])

    # CMOPRH dataset
    cmorph_fname = os.path.join(data_dir, 'CMORPH_monthly_0d25.nc')
    cmorph_ds = xr.open_dataset(cmorph_fname)
    cmorph_prec = cmorph_ds['cmorph']

    # "lev" is not needed ?
    cmorph_prec = cmorph_prec.drop('lev')
    cmorph_prec = cmorph_prec.transpose('time', 'lat', 'lon', 'lev')
    cmorph_prec = cmorph_prec.sortby(['time', 'lat', 'lon'])

    # Mask datasets
    mask_dataarray(eobs_prec, mask_da)
    mask_dataarray(cmorph_prec, mask_da)

    # Single time series for each data set.
    cmorph_prec_ts = cmorph_prec.groupby('time').mean()
    eobs_prec_ts = eobs_prec.groupby('time').mean()

    # Trend analysis
    y = eobs_prec_ts.values
    x = list(range(1, len(y) + 1))
    x = np.reshape(x, (-1, 1))
    x = sm.add_constant(x)
    trd = sm.OLS(y, x).fit()
    eobs_trend = xr.DataArray(trd.predict(), coords=eobs_prec_ts.coords)

    y = cmorph_prec_ts.values
    x = list(range(1, len(y) + 1))
    x = np.reshape(x, (-1, 1))
    x = sm.add_constant(x)
    trd = sm.OLS(y, x).fit()
    cmorph_trend = xr.DataArray(trd.predict(), coords=cmorph_prec_ts.coords)

    # Plot data
    eobs_prec_ts.plot(label='E-OBS')
    cmorph_prec_ts.plot(label='CMORPH')
    eobs_trend.plot(label='E-OBS trend')
    cmorph_trend.plot(label='CMORPH trend')
    plt.title("Time series of precipitation data")
    plt.ylabel("Mean precipitation [mm/hr]")
    plt.xlabel("Time")
    plt.grid(color='grey', linestyle='--', linewidth=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def read_nc(fname, data_var, mask=True):
    """
    Reader for the netcdf files of exercise 2
    :param fname: string
        name of file (including path)
    :param data_var: string
        name of the variable you want to read
    :param mask: Boolean, default=True
        Whether you want to mask the data with the area of the danube catchment (True) or not (False)
    :return: xarray.Dataarray
        dataarray with metadata and data of the data_var
    """
    ds = xr.open_dataset(fname)
    try:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    except ValueError:
        pass
    try:
        da = ds[data_var]
    except KeyError:
        print('Invalid "data_var". \n Available variables in this netcdf file: ')
        print(ds.data_vars)
        raise KeyError

    da = da.squeeze(drop=True)
    da = da.transpose('time', 'lat', 'lon')
    da = da.sortby(['time', 'lat', 'lon'])
    da['time'].values = pd.DatetimeIndex(pd.to_datetime(da['time'].values).date).map(
        lambda t: t.replace(day=1)).to_numpy()

    if mask:
        mask_fname = os.path.join(os.path.dirname(fname), 'mask.nc')
        mask_ds = xr.open_dataset(mask_fname)
        mask_da = mask_ds['mask']
        mask_dataarray(da, mask_da)
    return da


def read_csv(infile, parse_col, date_str='%Y-%m-%d'):
    """
    Function to read data.

    Parameters
    ----------
    infile : str
        Path to data.
    parse_col : str
        Column name of time variable.
    date_str : str
        String for datetime conversion.

    returns
    -------
    data : pd.DataFrame
    """
    def dateparse(x): return pd.datetime.strptime(x, date_str)
    data = pd.read_csv(infile, parse_dates=[parse_col], date_parser=dateparse)
    data.index = data[parse_col]
    return data.drop(parse_col, axis=1)


if __name__ == '__main__':
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

    # run the a function showcasing some relevant common python stuff
    data_overview(data_dir, out_dir)
    #bias(data_dir)
