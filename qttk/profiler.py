
from timeit import default_timer
from typing import List, Dict, Any, Callable

__all__ = ["time_this"]

# A module-level store of all the evaluation times of things you ran with the 
# @time_this decorator
runtime_table: List[Dict[str, Any]] = list()

def time_this(f):
    def timed_function(*args, **kwargs):
        ts = default_timer()
        print(f'Running {f.__name__} ...')

        result = f(*args, **kwargs)
        te = default_timer()
        t = 1000 * (te - ts)
        print(f'Completed {f.__name__} in {round(t, 3)} milliseconds')
        print()
        return result

    return timed_function


