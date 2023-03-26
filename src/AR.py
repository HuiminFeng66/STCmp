import warnings
warnings.filterwarnings('ignore')

# load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA,ARIMA
import scipy.stats as scs

from itertools import product
from tqdm import tqdm_notebook

# importing everything from forecasting quality metrics
from sklearn.metrics import r2_score,mean_absolute_error,median_absolute_error
from sklearn.metrics import median_absolute_error,mean_absolute_error,mean_squared_log_error


# use ACF,PACF to plot picture and to sure the paramters p,q
def mean_absolute_percentage_error(y_true,y_pred):
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

def tsplot(y,lags=None,figsize=(12,7),style='bmh'):
    """
           Plot time series, its ACF and PACF, calculate Dickey??CFuller test

           y - timeseries
           lags - how many lags to include in ACF, PACF calculation
       """ """
        Plot time series, its ACF and PACF, calculate Dickey??CFuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()



