'''
Module to access validation data
'''

import os
import pandas as pd

### Paths
base_path = os.path.dirname(__file__)

moving_average_input_csv = os.path.join(base_path,
                            '..',
                            '..',
                            'data',
                            'validation_data',
                            'ma_input_data.csv')

# verify this is the correct file
moving_average_target_csv = os.path.join(base_path,
                            '..',
                            '..',
                            'data',
                            'validation_data',
                            'ma_target_data.csv')

cumulative_average_input_csv = os.path.join(base_path, 
                            '..',
                            '..',
                            'data',
                            'validation_data',
                            'ValidationData-CMA_input.csv')

cumulative_average_target_csv = os.path.join(base_path, 
                            '..',
                            '..',
                            'data',
                            'validation_data',
                            'ValidationData-CMA_output.csv')

exponential_average_input_csv = os.path.join(base_path, 
                            '..',
                            '..',
                            'data',
                            'validation_data',
                            'ValidationData-CMA_input.csv')

exponential_average_target_csv = os.path.join(base_path, 
                            '..',
                            '..',
                            'data',
                            'validation_data',
                            'ValidationData-CMA_output.csv')

                            
def load_cumulative_moving_average():
    '''
    Loads 100k records, only returns 1500
    We should discuss
    '''
    return (pd.read_csv(cumulative_average_input_csv, index_col=0)[:1500]['0'],
            pd.read_csv(cumulative_average_target_csv, index_col=0)[:1500]['0'])


def load_moving_average():
    
    return (pd.read_csv(moving_average_input_csv),
            pd.read_csv(moving_average_target_csv))


def save_validation_data(function_name: str, data: pd.Series):
    # save validation data
    #_save_validation_data('simpleMA', x)
    #_save_validation_data('CMA_output', z)

    filename = os.path.join(path, '..', 'data', 'validation_data', \
    'ValidationData-{}.csv'.format(function_name))
    #data = fillinValues(data)
    data.to_csv(filename)


''' from ema file
def test(function_name: str, i):
    from pandas.testing import assert_series_equal
    from pandas.testing import assert_frame_equal

    path = os.path.dirname(__file__)
    filename = os.path.join(path, '..', 'data', 'validation_data', \
    'EMA-ValidationData-output_{}.csv'.format(function_name))
    validation_data = pd.read_csv(filename, index_col=0)

    # to cross validate the functions against one another
    if function_name == 'v1':
        filename2 = os.path.join(path, '..', 'data', 'validation_data', \
    'EMA-ValidationData-output_v2.csv')
    elif function_name == 'v2':
        filename2 = os.path.join(path, '..', 'data', 'validation_data', \
    'EMA-ValidationData-output_v1.csv')
    else:
        filename2 = os.path.join(path, '..', 'data', 'validation_data', \
    'EMA-ValidationData-output_v2.csv')

    # defines input data and parameters for the EMA functions
    series = pd.read_csv(os.path.join(path, '..', 'data', 'validation_data', \
    'Validation-data_input.csv'))
    mp = 5
    a = 2/(mp + 1)

    # unit test cases:
    x == x, _x == y, y == y, y == x, z == z, z == y

    todo resolve DataFrame shape mismatch
    line 153
    [left]:  (1000000, 2)
    [right]: (1000000, 1)

    if function_name == 'v1':
        x = exponential_moving_average_v1(series, window=mp, alpha=a)
        x = pd.DataFrame(x)
        assert_frame_equal(x, validation_data, check_dtype=False)
        y = pd.read_csv(filename2)
        assert_frame_equal(x, y, check_dtype=False)
    elif function_name == 'v2':
        y = exponential_moving_average_v2(series, window=mp, alpha=a)
        y = pd.DataFrame(y)
        assert_frame_equal(y, validation_data, check_dtype=False)
        x = pd.read_csv(filename2)
        assert_frame_equal(y, x, check_dtype=False)
    else:
        z = exponential_moving_average_v2(series, window=mp)
        z = pd.DataFrame(z)
        assert_frame_equal(z, validation_data, check_dtype=False)
        y = pd.read_csv(filename2)
        assert_frame_equal(z, y, check_dtype=False)
    '''


if __name__ == '__main__':
    # test directories exist
    assert os.path.exists(moving_average_input_csv)
    assert os.path.exists(moving_average_target_csv)
    assert os.path.exists(cumulative_average_input_csv)
    assert os.path.exists(cumulative_average_target_csv)
