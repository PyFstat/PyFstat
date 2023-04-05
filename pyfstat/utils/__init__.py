"""
A collection of helpful functions to facilitate ease-of-use of PyFstat.

Most of these are used internally by other parts of the package
and are of interest mostly only for developers,
but others can also be helpful for end users.
"""

from .atoms import copy_FstatAtomVector, extract_singleIFOmultiFatoms_from_multiAtoms
from .cli import match_commandlines, run_commandline
from .converting import (
    convert_aPlus_aCross_to_h0_cosi,
    convert_h0_cosi_to_aPlus_aCross,
    get_dictionary_from_lines,
    gps_to_datestr_utc,
    parse_list_of_numbers,
)
from .ephemeris import get_ephemeris_files
from .formatting import get_doppler_params_output_format, round_to_n, texify_float
from .gsl import convert_array_to_gsl_matrix
from .importing import initializer, safe_X_less_plt
from .io import (
    get_parameters_dict_from_file_header,
    read_par,
    read_parameters_dict_lines_from_file_header,
    read_txt_file_with_header,
)
from .predict import get_predict_fstat_parameters_from_dict, predict_fstat
from .runlalsuite import generate_loudest_file, get_covering_band
from .sft import (
    get_commandline_from_SFTDescriptor,
    get_official_sft_filename,
    get_sft_as_arrays,
)
