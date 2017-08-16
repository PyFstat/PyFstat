""" Additional helper functions dealing with transient-CW F(t0,tau) maps """

import logging

# optional imports
import importlib as imp


def optional_import ( modulename, shorthand=None ):
    '''
    Import a module/submodule only if it's available.

    using importlib instead of __import__
    because the latter doesn't handle sub.modules
    '''
    if shorthand is None:
        shorthand    = modulename
        shorthandbit = ''
    else:
        shorthandbit = ' as '+shorthand
    try:
        globals()[shorthand] = imp.import_module(modulename)
        #logging.debug('Successfully imported module %s%s.' % (modulename, shorthandbit))
        success = True
    except ImportError, e:
        if e.message == 'No module named '+modulename:
            logging.warning('No module {:s} found.'.format(modulename))
            success = False
        else:
            raise
    return success


# dictionary of the actual callable F-stat map functions we support,
# if the corresponding modules are available.
fstatmap_versions = {
                     'lal':    lambda multiFstatAtoms, windowRange:
                               getattr(lalpulsar,'ComputeTransientFstatMap')
                                ( multiFstatAtoms, windowRange, False ),
                     #'pycuda': lambda multiFstatAtoms, windowRange:
                               #pycuda_compute_transient_fstat_map
                                #( multiFstatAtoms, windowRange )
                    }


def init_transient_fstat_map_features ( ):
    '''
    Initialization of available modules (or "features") for F-stat maps.

    Returns a dictionary of method names, to match fstatmap_versions
    each key's value set to True only if
    all required modules are importable on this system.
    '''
    features = {}
    have_lal           = optional_import('lal')
    have_lalpulsar     = optional_import('lalpulsar')
    features['lal']    = have_lal and have_lalpulsar
    features['pycuda'] = False
    logging.debug('Got the following features for transient F-stat maps:')
    logging.debug(features)
    return features


def call_compute_transient_fstat_map ( version, features, multiFstatAtoms=None, windowRange=None ):
    '''Choose which version of the ComputeTransientFstatMap function to call.'''

    if version in fstatmap_versions:
        if features[version]:
            FstatMap = fstatmap_versions[version](multiFstatAtoms, windowRange)
        else:
            raise Exception('Required module(s) for transient F-stat map method "{}" not available!'.format(version))
    else:
        raise Exception('Transient F-stat map method "{}" not implemented!'.format(version))
    return FstatMap
