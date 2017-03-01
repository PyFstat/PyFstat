"""
Provides helpful functions to facilitate ease-of-use of pyfstat
"""

import os
import sys
import argparse
import logging
import inspect
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np


def set_up_optional_tqdm():
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, *args, **kwargs):
            return x
    return tqdm


def set_up_matplotlib_defaults():
    plt.switch_backend('Agg')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['axes.formatter.useoffset'] = False


def set_up_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--quite", help="Decrease output verbosity",
                        action="store_true")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity",
                        action="store_true")
    parser.add_argument("--no-interactive", help="Don't use interactive",
                        action="store_true")
    parser.add_argument("-c", "--clean", help="Don't use cached data",
                        action="store_true")
    parser.add_argument("-u", "--use-old-data", action="store_true")
    parser.add_argument('-s', "--setup-only", action="store_true")
    parser.add_argument('-n', "--no-template-counting", action="store_true")
    parser.add_argument('unittest_args', nargs='*')
    args, unknown = parser.parse_known_args()
    sys.argv[1:] = args.unittest_args
    if args.quite or args.no_interactive:
        def tqdm(x, *args, **kwargs):
            return x
    else:
        tqdm = set_up_optional_tqdm()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    if args.quite:
        stream_handler.setLevel(logging.WARNING)
    elif args.verbose:
        stream_handler.setLevel(logging.DEBUG)
    else:
        stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)-8s: %(message)s', datefmt='%H:%M'))
    logger.addHandler(stream_handler)
    return args, tqdm


def set_up_ephemeris_configuration():
    config_file = os.path.expanduser('~')+'/.pyfstat.conf'
    if os.path.isfile(config_file):
        d = {}
        with open(config_file, 'r') as f:
            for line in f:
                k, v = line.split('=')
                k = k.replace(' ', '')
                for item in [' ', "'", '"', '\n']:
                    v = v.replace(item, '')
                d[k] = v
        earth_ephem = d['earth_ephem']
        sun_ephem = d['sun_ephem']
    else:
        logging.warning('No ~/.pyfstat.conf file found please provide the '
                        'paths when initialising searches')
        earth_ephem = None
        sun_ephem = None
    return earth_ephem, sun_ephem


def round_to_n(x, n):
    if not x:
        return 0
    power = -int(np.floor(np.log10(abs(x)))) + (n - 1)
    factor = (10 ** power)
    return round(x * factor) / factor


def texify_float(x, d=2):
    if type(x) == str:
        return x
    x = round_to_n(x, d)
    if 0.01 < abs(x) < 100:
        return str(x)
    else:
        power = int(np.floor(np.log10(abs(x))))
        stem = np.round(x / 10**power, d)
        if d == 1:
            stem = int(stem)
        return r'${}{{\times}}10^{{{}}}$'.format(stem, power)


def initializer(func):
    """ Decorator function to automatically assign the parameters to self """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

