'''
Module to access validation data

1/31/2021 updates:

* Simplified MA-ValidationData-simpleMA.csv with a 
  shorter file that has a date index
* Replaced switch statement for various function
  with simpler named function that returns both 
* Added tests to confirm files exist
* Deleted EMA v1, v2, v3 because there should be
  one source of truth or assert_series_equals
  can be replaced with assert_almost_equals
  and a threshold can be set so users know
  what types of variances to expect.
* Added steps to create validation dataset in function docstring
* Adjusted functions to return a series datatype

'''

import os
import pandas as pd

### Paths
base_path = os.path.dirname(__file__)

sma_validation_data = os.path.join(base_path,
                            '..',
                            'data',
                            'validation_data',
                            'sma_validation_data.csv')

cma_validation_data = os.path.join(base_path, 
                            '..',
                            'data',
                            'validation_data',
                            'cma_validation_data.csv')

wma_validation_data = os.path.join(base_path, 
                            '..',
                            'data',
                            'validation_data',
                            'wma_validation_data.csv')

ema_validation_data = os.path.join(base_path, 
                            '..',
                            'data',
                            'validation_data',
                            'ema_validation_data.csv')


                            
def load_cma_validation_data():
    '''
    To create validation dataset:

    >>> cma_val_df = pd.DataFrame({'data':range(1,501) * np.random.rand(500)})
    >>> cma_val_df.index = pd.date_range(start='12-1-2010', periods=cma_val_df.shape[0])
    >>> cma_val_df.index.name = 'date_idx'
    >>> cma_val_df['target'] = cma_val_df['data'].expanding(20).mean()
    >>> cma_val_df.to_csv('data/validation_data/cma_validation_data.csv')

    '''
    cma_df = pd.read_csv(cma_validation_data, index_col='date_idx')
    return (cma_df['data'],
            cma_df['target'])


def load_sma_validation_data():
    '''
    To create validation dataset:

    >>> sma_val_df = pd.DataFrame({'data':range(1,501) * np.random.rand(500)})
    >>> sma_val_df.index = pd.date_range(start='12-1-2010', periods=sma_val_df.shape[0])
    >>> sma_val_df.index.name = 'date_idx'
    >>> sma_val_df['target'] = sma_val_df['data'].rolling(20).mean()
    >>> sma_val_df.to_csv('data/validation_data/sma_validation_data.csv')

    '''
    sma_df = pd.read_csv(sma_validation_data, index_col='date_idx')
    return (sma_df['data'],
            sma_df['target'])


def load_wma_validation_data():
    '''
    To create validation dataset:

    >>> wma_val_df = pd.DataFrame({'data':range(1,501) * np.random.rand(500)})
    >>> wma_val_df.index = pd.date_range(start='12-1-2010', periods=wma_val_df.shape[0])
    >>> wma_val_df.index.name = 'date_idx'
    >>> wma_val_df['target'] = talib.WMA(wma_val_df['data'],timeperiod=20)
    >>> wma_val_df.to_csv('wma_validation_data.csv')
    >>> wma_val_df.to_csv('data/validation_data/wma_validation_data.csv')

    '''
    wma_df = pd.read_csv(wma_validation_data, index_col='date_idx')
    return (wma_df['data'],
            wma_df['target'])


def load_ema_validation_data():
    '''
    To create validation dataset:

    >>> ema_val_df = pd.DataFrame({'data':range(1,501) * np.random.rand(500)})
    >>> ema_val_df.index = pd.date_range(start='12-1-2010', periods=ema_val_df.shape[0])
    >>> ema_val_df.index.name = 'date_idx'
    >>> ema_val_df['target'] = talib.EMA(ema_val_df['data'],timeperiod=20)
    >>> ema_val_df.to_csv('data/validation_dataewma_validation_data.csv')

    '''
    ema_df = pd.read_csv(ema_validation_data, index_col='date_idx')
    return (ema_df['data'],
            ema_df['target'])


def save_validation_data(function_name: str, data: pd.Series):
    # save validation data
    #_save_validation_data('simpleMA', x)
    #_save_validation_data('CMA_output', z)

    filename = os.path.join(path, '..', 'data', 'validation_data', \
    'ValidationData-{}.csv'.format(function_name))
    #data = fillinValues(data)
    data.to_csv(filename)


if __name__ == '__main__':
    # test directories exist
    assert os.path.exists(sma_validation_data)
    assert os.path.exists(cma_validation_data)
    assert os.path.exists(wma_validation_data)

