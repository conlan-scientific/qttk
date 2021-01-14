from timeit import default_timer
import pandas as pd
from contextlib import contextmanager
from typing import List, Dict, Any, Callable
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import os
import re
import numpy as np

plt.rcParams['figure.figsize'] = (5, 3.5)
plt.rcParams['figure.dpi'] = 150

mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['legend.title_fontsize'] = 8
mpl.rcParams['legend.handlelength'] = 2

pd.set_option('display.max_rows', None)

__all__ = ["time_this", "report_results", "timed_report", "ExponentialRange"]
# A module-level store of all the evaluation times of things you ran with the 
# @time_this decorator
runtime_table: List[Dict[str, Any]] = list()

utils_dir = os.path.dirname(os.path.abspath(__file__))
listings_dir = os.path.join(utils_dir, '..', '..', 'listings')

if not os.path.exists(listings_dir):
    os.mkdir(listings_dir)

SHOW = False

# BW_MODS will do a number of things to make the book content display better 
# for purposes of black and white printing
BW_MODS = True
if BW_MODS:
    plt.style.use('grayscale')



def clear_runtime_table():
    del runtime_table[:]

def time_this(length_method: Callable[[Any], int]):
    """
    A decorator that stores the evaluation time against a user-defined length 
    and plots it.

    Usage: 
        @time_this(lambda x: len(x))
        def some_function(x, y, z):
            # do something ...
    """
    def _time_this(method):

        def timed_function(*args, **kwargs):
            ts = default_timer()
            result = method(*args, **kwargs)
            te = default_timer()
            print(f'{method.__name__}')

            n = length_method(*args, **kwargs)
            t = (te - ts) * 1000
            n_over_t = round(n / t, 4)
            print(f'    n   = {n} values')
            print(f'    t   = {round(t, 3)} ms')
            print(f'    n/t = {n_over_t} values per ms')
            print()
            runtime_table.append({
                'function': method.__name__,
                'n_values': n,
                't_milliseconds': round(t, 3),
                'values_per_ms': n_over_t,
            })
            return result

        return timed_function

    return _time_this

def title_to_snake(val):
    pre = re.sub(r'[^a-zA-Z0-9]', '_', val)
    return re.sub(r'_+', r'_', pre).lower()

def get_max_rows(df):
    """
    Get last row of each function profile
    """
    max_rows = []
    function_names = df['function'].unique()
    for fname in function_names:
        _df = df[df['function'] == fname]
        _df = _df.sort_values('n_values')
        row = _df.iloc[-1]
        max_rows.append(row)
    table_df = pd.DataFrame(max_rows)
    return table_df    

def build_markdown_table(df, table_path):

    index = df.groupby(['function']).transform(max).index

    table_df = get_max_rows(df)

    table_df['function'] = [
        f'`{v}`' for v in table_df['function']
    ]

    table_df['n_values'] = [
        f'{v:.0E}' for v in table_df['n_values']
    ]

    table_df['t_milliseconds'] = [
        f'{v:.1E}' for v in table_df['t_milliseconds']
    ]

    table_df['values_per_ms'] = [
        f'{v:.1E}' for v in table_df['values_per_ms']
    ]

    display_names_by_col = {
        'function': '$f$',
        'n_values': '$n$',
        't_milliseconds': '$t$ (ms)',
        'values_per_ms': '$n / t$',
    }

    col_names = table_df.columns.values
    dis_names = [display_names_by_col[k] for k in col_names]

    col_lengths_by_col = {
        'function': 64,
        'n_values': 20,
        't_milliseconds': 20,
        'values_per_ms': 20,
    }

    total_length = \
        sum(col_lengths_by_col.values()) + \
        len(col_names) - 1

    def pad(val, _len):
        return str(val) + ' ' * (_len - len(str(val)))

    def right_pad(val, _len):
        return ' ' * (_len - len(str(val))) + str(val)

    table: str = '-'*total_length + '\n'
    for i, col in enumerate(col_names):
        dis = display_names_by_col[col]
        _len = col_lengths_by_col[col]
        if i == 0:
            table += pad(dis, _len)
        else:
            table += ' ' + right_pad(dis, _len)
    table += '\n'

    for col in col_names:
        _len = col_lengths_by_col[col]
        table += '-'*_len + ' '
    table += '\n'

    for _, row in table_df.iterrows():
        for i, col in enumerate(col_names):
            _len = col_lengths_by_col[col]
            if i == 0:
                table += pad(row[col], _len)
            else:
                table += ' ' + right_pad(row[col], _len)
        table += '\n\n'
    table = table[:-1]

    table += '-'*total_length + '\n'

    with open(table_path, 'w') as file_ptr:
        file_ptr.write(table)

    print(table)




def report_results(chapter: int, start_figure: int, title, save=True):
    """
    Plot and print some information about the efficiency of the algorithms you
    just ran
    """

    chapter = int(chapter)
    chapter_dir = os.path.join(listings_dir, f'chapter_{chapter}')
    if not os.path.exists(chapter_dir):
        os.mkdir(chapter_dir)

    start_figure = int(start_figure)

    fig_one_pref = f'figure_{chapter}_{start_figure}_'
    fig_two_pref = f'figure_{chapter}_{start_figure+1}_'
    table_pref = f'table_{chapter}_{start_figure}_'

    title = title_to_snake(title)

    figure_one_path = os.path.join(chapter_dir, f'{fig_one_pref}{title}.png')
    figure_two_path = os.path.join(chapter_dir, f'{fig_two_pref}{title}.png')
    table_path = os.path.join(chapter_dir, f'{table_pref}{title}.Rmd')

    df = pd.DataFrame(runtime_table)
    print(df)
    df_for_table = df.copy()

    if BW_MODS:
        # Intelligently plot a maximum of three lines in 
        # the profiles
        max_df = get_max_rows(df)
        max_df = max_df.sort_values('values_per_ms')

        functions_to_keep = list()
        n_funcs = max_df.shape[0]

        # If only one, keep it, but keep the most efficienct
        if n_funcs >= 1:
            fastest_func = max_df.iloc[-1].function
            functions_to_keep.append(fastest_func)

        # If two, insert the least efficient one
        if n_funcs >= 2:
            slowest_func = max_df.iloc[0].function
            functions_to_keep.insert(0, slowest_func)

        # If three or more, insert the middle efficient one
        if n_funcs >= 3:
            middle_i = n_funcs // 2
            if middle_i != 0:
                middle_func = max_df.iloc[middle_i].function
                functions_to_keep.insert(1, middle_func)

        functions_to_keep.reverse()
        df = df[df['function'].isin(functions_to_keep)]
        print(df)

    pivot_table = df.pivot(
        index='n_values',
        columns='function',
        values='t_milliseconds',
    )
    if BW_MODS:
        # If BW, order fastest to slowest
        pivot_table = pivot_table[functions_to_keep]

    ax = pivot_table.plot(
        logx=True,
        logy=True,
        title='Milliseconds to complete',
    )
    ax.set_ylabel('milliseconds')
    ax.set_xlabel('input length')
    plt.tight_layout()
    plt.grid()

    if save:
        plt.savefig(figure_one_path)

    pivot_table = df.pivot(
        index='n_values',
        columns='function',
        values='values_per_ms',
    )
    if BW_MODS:
        # If BW, order fastest to slowest
        pivot_table = pivot_table[functions_to_keep]

    ax = pivot_table.plot(
        logx=True,
        logy=True,
        title='Values processed per millisecond',
    )
    ax.set_ylabel('values per millisecond')
    ax.set_xlabel('input length')
    plt.tight_layout()
    plt.grid()

    if save:
        plt.savefig(figure_two_path)

    if SHOW:
        plt.show()

    build_markdown_table(df_for_table, table_path)

@contextmanager
def timed_report(chapter, start_figure, title):
    """
    e a s e   o f   u s e
    """
    yield
    report_results(chapter, start_figure, title, save=True)
    clear_runtime_table()


ONE_FOURTH = 1/4

class ExponentialRange(object):
    """
    A range that operates on exponents of 10, inclusive
    """

    def __init__(self, start_exponent: int, end_exponent: int, 
        step_size: float=ONE_FOURTH, int_only: bool=True):

        self.step_size = step_size
        self.start = self.exp_to_int(start_exponent)
        self.end = self.exp_to_int(end_exponent)
        self.int_only = int_only

    def exp_to_int(self, end_exponent: int):
        return math.ceil(end_exponent / self.step_size)

    def get_element(self, i):
        """
        Get the i-th element of the iteration
        """
        val = 10 ** (i * self.step_size)
        if self.int_only:
            return int(val)
        return val

    def iterator(self, alt_end: int=None):
        """
        Yield unique values of get_element for i in start through end
        """
        existing_entries = set()

        start = self.start

        if alt_end:
            end = self.exp_to_int(alt_end)
        else:
            end = self.end

        for i in range(start, end + 1):
            value = self.get_element(i)
            if not value in existing_entries:
                yield value
            existing_entries.add(value)        

    def np_range(self):
        return np.array([*self.iterator()])

    @property
    def max(self):
        return self.get_element(self.end)

