import lal


def convert_array_to_gsl_matrix(array):
    """Convert a numpy array to a LAL-wrapped GSL matrix.

    Parameters
    ----------
    array: np.ndarray
        The array to convert.
        `array.shape` must have 2 dimensions.

    Returns
    ----------
    gsl_matrix: lal.gsl_matrix
        The LAL-wrapped GSL matrix object.
    """
    gsl_matrix = lal.gsl_matrix(*array.shape)
    gsl_matrix.data = array
    return gsl_matrix
