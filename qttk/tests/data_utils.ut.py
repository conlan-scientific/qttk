'''
This file should be in the tests folder
If your dev env is unable to import try this:
   from qttk package folder, `conda develop .`


'''

import unittest
import sys
import os
import pandas as pd
import numpy as np
from qttk.utils.data_utils import check_dataframe_columns


class TestDataUtils(unittest.TestCase):

    def setUp(self):
        test_columns = ['date', 'open', 'close', 'low', 'high', 'volume']
        self.df = pd.DataFrame(np.random.rand(4,6), columns=test_columns)        


    def tearDown(self):
        self.df = None


    def test_exact_match(self):
        test_columns = pd.Series(['date', 'open', 'close',
                                  'low', 'high', 'volume'])
        self.assertIsNone(check_dataframe_columns(self.df, test_columns))


    def test_partial_match_strict(self):
        test_columns = pd.Series(['date', 'open', 'close',
                                  'low', 'high', 'volume', 'donut'])
        self.assertRaises(IndexError, check_dataframe_columns, self.df, test_columns)


    def test_partial_match_forgiving(self):
        test_columns = pd.Series(['date', 'open', 'close', 'low', 'high'])
        self.assertIsNone(check_dataframe_columns(self.df, test_columns, False))

    def test_error_text_required_columns(self):
        test_columns = pd.Series(['date', 'open', 'close',
                                  'low', 'high', 'volume', 'donut'])

        required_err_text = "Required: "
        required_err_text += ', '.join(test_columns)
        
        self.assertRaisesRegex(IndexError,
                               required_err_text,
                               check_dataframe_columns,
                               self.df,
                               test_columns)

    def test_error_text_missing_columns(self):
        test_columns = pd.Series(['date', 'open', 'close',
                                  'low', 'high', 'volume', 'donut'])

        required_err_text = "Missing: "
        required_err_text += ', '.join(['donut'])
        
        self.assertRaisesRegex(IndexError,
                               required_err_text,
                               check_dataframe_columns,
                               self.df,
                               test_columns)



if __name__ == '__main__':
    unittest.main()
