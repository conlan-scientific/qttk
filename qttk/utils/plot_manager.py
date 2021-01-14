import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from contextlib import contextmanager
import os
import re
from typing import Tuple

import matplotlib as mpl

SHOW = True

def title_to_snake(val):
    pre = re.sub(r'[^a-zA-Z0-9]', '_', val)
    return re.sub(r'_+', r'_', pre).lower()


plt.rcParams['figure.figsize'] = (5, 4)
plt.rcParams['figure.dpi'] = 150

mpl.rcParams['grid.color'] = 'k'
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['grid.linewidth'] = 0.5

mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['legend.title_fontsize'] = 6
mpl.rcParams['legend.handlelength'] = 2

utils_dir = os.path.dirname(os.path.abspath(__file__))
listings_dir = os.path.join(utils_dir, '..', '..', 'listings')


def plot_manager_factory(chapter: int):

    chapter = int(chapter)
    chapter_dir = os.path.join(
        listings_dir, 
        f'chapter_{chapter}'
    )
    if not os.path.exists(chapter_dir):
        os.mkdir(chapter_dir)

    @contextmanager
    def plot_manager(figure: int, title: str):
        yield

        plt.grid()

        figure = int(figure)
        figure_prefix = f'figure_{chapter}_{figure}_'
        path_pref = os.path.join(chapter_dir, figure_prefix)
        title = title_to_snake(title)
        figure_path = f'{path_pref}{title}.png'

        plt.savefig(figure_path)

        if SHOW:
            plt.show()

    return plot_manager
