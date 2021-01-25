"""
Unittests for Bollinger_2.py

Attempting to add basic testing to charts

"""
import unittest
from unittest.mock import patch
from qttk import Bollinger_2 as b2
import pandas as pd
from unittest.mock import create_autospec
from numpy import nan


class TestBollinger(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
        'date': {0: '2019-12-17', 1: '2019-12-18',  2: '2019-12-19',
                 3: '2019-12-20', 4: '2019-12-23',  5: '2019-12-24',
                 6: '2019-12-26', 7: '2019-12-27',  8: '2019-12-30', 9: '2019-12-31'},
        'open': {0: 336.86, 1: 332.17, 2: 331.29, 3: 331.58, 4: 328.4,
                 5: 331.06, 6: 331.36, 7: 332.47, 8: 323.75, 9: 323.91},
        'close': {0: 334.48, 1: 329.61, 2: 330.76, 3: 325.19, 4: 330.61,
                  5: 330.85, 6: 329.05, 7: 325.38, 8: 324.06, 9: 328.93},
        'low': {0: 332.8, 1: 329.24, 2: 328.85, 3: 323.57, 4: 327.74,
                5: 328.43, 6: 327.47, 7: 325.16, 8: 320.95, 9: 323.7},
        'high': {0: 337.33, 1: 334.58, 2: 333.27, 3: 333.68, 4: 333.24,
                 5: 332.91, 6: 333.05, 7: 332.47, 8: 325.35, 9: 329.09},
        'volume': {0: 279911, 1: 249112, 2: 375898, 3: 983200, 4: 296139,
                   5: 98751, 6: 209398, 7: 225667, 8: 289266, 9: 306657}})

        self.ground_truth_df = pd.DataFrame({
            'open': {'2019-12-17': 336.86, '2019-12-18': 332.17, '2019-12-19': 331.29,
                     '2019-12-20': 331.58, '2019-12-23': 328.4, '2019-12-24': 331.06,
                     '2019-12-26': 331.36, '2019-12-27': 332.47, '2019-12-30': 323.75, '2019-12-31': 323.91},
            'close': {'2019-12-17': 334.48, '2019-12-18': 329.61, '2019-12-19': 330.76,
                      '2019-12-20': 325.19, '2019-12-23': 330.61, '2019-12-24': 330.85,
                      '2019-12-26': 329.05, '2019-12-27': 325.38, '2019-12-30': 324.06, '2019-12-31': 328.93},
            'low': {'2019-12-17': 332.8, '2019-12-18': 329.24, '2019-12-19': 328.85,
                    '2019-12-20': 323.57, '2019-12-23': 327.74,'2019-12-24': 328.43,
                    '2019-12-26': 327.47, '2019-12-27': 325.16, '2019-12-30': 320.95, '2019-12-31': 323.7},
            'high': {'2019-12-17': 337.33, '2019-12-18': 334.58, '2019-12-19': 333.27,
                     '2019-12-20': 333.68, '2019-12-23': 333.24, '2019-12-24': 332.91,
                     '2019-12-26': 333.05, '2019-12-27': 332.47, '2019-12-30': 325.35, '2019-12-31': 329.09},
            'volume': {'2019-12-17': 279911, '2019-12-18': 249112, '2019-12-19': 375898,
                       '2019-12-20': 983200, '2019-12-23': 296139, '2019-12-24': 98751,
                       '2019-12-26': 209398, '2019-12-27': 225667, '2019-12-30': 289266, '2019-12-31': 306657},
            'MA_Close': {'2019-12-17': nan, '2019-12-18': nan, '2019-12-19': nan, '2019-12-20': nan,
                        '2019-12-23': 330.13, '2019-12-24': 329.404, '2019-12-26': 329.29200000000003,
                        '2019-12-27': 328.216,'2019-12-30': 327.99, '2019-12-31': 327.654},
            'std': {'2019-12-17': nan, '2019-12-18': nan, '2019-12-19': nan,
                    '2019-12-20': nan, '2019-12-23': 3.325048871821292, '2019-12-24': 2.4075049324975533,
                    '2019-12-26': 2.4085514318776844, '2019-12-27': 2.764250350456695,'2019-12-30': 3.0993789700518897,
                    '2019-12-31': 2.8230887339932953},
            'BOLU': {'2019-12-17': nan, '2019-12-18': nan,'2019-12-19': nan,'2019-12-20': nan,
                     '2019-12-23': 336.78009774364256, '2019-12-24': 334.2190098649951, '2019-12-26': 334.1091028637554,
                     '2019-12-27': 333.7445007009134, '2019-12-30': 334.1887579401038, '2019-12-31': 333.3001774679866},
            'BOLD': {'2019-12-17': nan, '2019-12-18': nan, '2019-12-19': nan, '2019-12-20': nan,
                     '2019-12-23': 323.47990225635743, '2019-12-24': 324.5889901350049, '2019-12-26': 324.47489713624464,
                     '2019-12-27': 322.6874992990866, '2019-12-30': 321.7912420598962, '2019-12-31': 322.0078225320134},
            'MA_Volume': {'2019-12-17': nan, '2019-12-18': nan, '2019-12-19': nan, '2019-12-20': nan,
                          '2019-12-23': 436852.0, '2019-12-24': 400620.0, '2019-12-26': 392677.2, '2019-12-27': 362631.0,
                          '2019-12-30': 223844.2, '2019-12-31': 225947.8},
            'pct_b': {'2019-12-17': nan, '2019-12-18': nan, '2019-12-19': nan, '2019-12-20': nan,
                   '2019-12-23': -0.5360896951070301, '2019-12-24': -0.6501554556006606, '2019-12-26': -0.47488116749375825,
                   '2019-12-27': -0.24351093059177398, '2019-12-30': -0.18300101101107075, '2019-12-31': -0.6129968024592597},
            'Bandwidth': {'2019-12-17': nan, '2019-12-18': nan, '2019-12-19': nan, '2019-12-20': nan,
                          '2019-12-23': 0.04028775175623277, '2019-12-24': 0.02923467756915575, '2019-12-26': 0.029257333088902215,
                          '2019-12-27': 0.033688185225055525, '2019-12-30': 0.037798456904806876, '2019-12-31': 0.03446426698887609}})
        
        self.ground_truth_df.index.name = 'date'

    def tearDown(self):
        self.df = None
        self.ground_truth_df = None


    def test_bb_graph_formatter_call(self):
        mock_bb_graph_formatter = create_autospec(b2.bb_graph_formatter)
        mock_bb_graph_formatter(self.df)
        mock_bb_graph_formatter.assert_called_once_with(self.df)


    def test_bb_graph_formatter_type_error(self):
        with self.assertRaises(TypeError):        
            b2.bb_graph_formatter(None)


    def test_bollinger_output_columns(self):
        bollinger_df = b2.bollinger(self.df)
        expected_columns = ['MA_Close', 'std', 'BOLU', 'BOLD', 'MA_Volume', 'Bandwidth']
        self.assertTrue(set(expected_columns).issubset(bollinger_df.columns))


    def test_bollinger_ground_truth(self):
        self.df.set_index('date', drop=True, inplace=True)
        bollinger_df = b2.bollinger(self.df,
                                    moving_avg_window=5,
                                    std_window=5,
                                    volume_window=5)

        pd._testing.assert_equal(bollinger_df, self.ground_truth_df)


    def test_bollinger_plt_show(self):
        '''
        Todo: remove integration test components
              depends on files existing, reading csv files
        '''
        with patch("qttk.Bollinger_2.plt.show") as plt_show_path:
            b2.main(save_figure=False)
            assert plt_show_path.called


    def test_bb_graph_formatter_subplots_adjust(self):
        with patch('qttk.Bollinger_2.plt.subplots_adjust') as mock_subplots_adjust:
            b2.bb_graph_formatter(self.ground_truth_df)
            assert mock_subplots_adjust.called


  # def test_bb_graph_formatter_subplots(self):
    #    with patch('b2.plt.subplots_adjust') as mock_subplots_adjust:
    #        with patch('b2.plt.subplots',return) as mock_subplot:
    #           mock_subplot.axs.shape == (4,)


if __name__ == '__main__':
    unittest.main()

