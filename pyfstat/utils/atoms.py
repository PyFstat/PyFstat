import logging

import lal
import lalpulsar
import numpy as np

logger = logging.getLogger(__name__)


def write_atoms_to_txt_file(fname, atoms, header=[], comments="%%"):
    """Save F-statistic atoms (time-dependent quantities) to a text file.

    Parameters
    ----------
    fname: str
        Output filename.
    atoms: lalpulsar.MultiFstatAtomVector
        The F-stat atoms data to write out.
    header: list
        A list of header lines to write to the top of the file.
    comments: str
        Comments marker character(s) to be prepended to header lines.
        Note that the column headers line
        (last line of the header before the atoms data)
        is printed by lalpulsar, with `%%` as comments marker,
        so (different from most other PyFstat functions)
        the default here is `%%` too.
    """
    fo = lal.FileOpen(fname, "w")
    for hline in header:
        lal.FilePuts(f"{comments} {hline}\n", fo)
    lalpulsar.write_MultiFstatAtoms_to_fp(fo, atoms)
    del fo  # instead of lal.FileClose() which is not SWIG-exported


def extract_singleIFOmultiFatoms_from_multiAtoms(
    multiAtoms: lalpulsar.MultiFstatAtomVector, X: int
) -> lalpulsar.MultiFstatAtomVector:
    """Extract a length-1 MultiFstatAtomVector from a larger MultiFstatAtomVector.

    The result is needed as input to ``lalpulsar.ComputeTransientFstatMap`` in some places.

    The new object is freshly allocated,
    and we do a deep copy of the actual per-timestamp atoms.

    Parameters
    -------
    multiAtoms:
        Fully allocated multi-detector struct of `length > X`.
    X:
        The detector index for which to extract atoms.
    Returns
    -------
    singleIFOmultiFatoms: lalpulsar.MultiFstatAtomVector
        Length-1 MultiFstatAtomVector with only the data for detector `X`.
    """
    if X >= multiAtoms.length:
        raise ValueError(
            f"Detector index {X} is out of range for multiAtoms of length {multiAtoms.length}."
        )
    singleIFOmultiFatoms = lalpulsar.CreateMultiFstatAtomVector(1)
    singleIFOmultiFatoms.data[0] = lalpulsar.CreateFstatAtomVector(
        multiAtoms.data[X].length
    )
    # we deep-copy the entries of the atoms vector,
    # since just assigning the whole array can cause a segfault
    # from memory cleanup in looping over this function
    copy_FstatAtomVector(singleIFOmultiFatoms.data[0], multiAtoms.data[X])
    return singleIFOmultiFatoms


def copy_FstatAtomVector(
    dest: lalpulsar.FstatAtomVector, src: lalpulsar.FstatAtomVector
):
    """Deep-copy an FstatAtomVector with all its per-SFT FstatAtoms.

    The two vectors must have the same length,
    and the destination vector must already be allocated.

    Parameters
    -------
    dest:
        The destination vector to copy to.
        Must already be allocated.
        Will be modified in-place.
    src:
        The source vector to copy from.
    """
    if dest.length != src.length:
        raise ValueError(
            f"Lengths of destination and source vectors do not match. ({dest.length} != {src.length})"
        )
    dest.TAtom = src.TAtom
    for k in range(dest.length):
        # this is now copying the actual FstatAtom object,
        # with its actual data in memory (no more pointers)
        dest.data[k] = src.data[k]


def reshape_FstatAtomVector_to_dict_of_arrays(atomsVector):
    """Make a dictionary of ndarrays out of an F-stat atoms 'vector' structure.

    Parameters
    ----------
    atomsVector: lalpulsar.FstatAtomVector
        The atoms in a 'vector'-like structure:
        iterating over timestamps as the higher hierarchical level,
        with a set of 'atoms' quantities defined at each timestamp.

    Returns
    -------
    atomsDict: dict
        A dictionary with an entry for each quantity,
        which then is a 1D ndarray over timestamps for that one quantity.
    """

    numAtoms = atomsVector.length
    atomsDict = {}
    tempDict = {}
    atom_fields = [
        ("timestamp", np.uint32),
        ("a2_alpha", np.float32),
        ("b2_alpha", np.float32),
        ("ab_alpha", np.float32),
        ("Fa_alpha", complex),
        ("Fb_alpha", complex),
    ]
    for dtype in atom_fields:
        if dtype[1] == complex:
            for part in "re", "im":
                atomsDict[dtype[0] + "_" + part] = np.ndarray(
                    numAtoms, dtype=np.float32
                )
            tempDict[dtype[0]] = np.ndarray(numAtoms, dtype=complex)
        else:
            atomsDict[dtype[0]] = np.ndarray(numAtoms, dtype=dtype[1])
    for n, atom in enumerate(atomsVector.data):
        for dtype in atom_fields:
            if dtype[1] == complex:
                tempDict[dtype[0]][n] = atom.__getattribute__(dtype[0])
            else:
                atomsDict[dtype[0]][n] = atom.__getattribute__(dtype[0])
    for dtype in atom_fields:
        if dtype[1] == complex:
            for part in "real", "imag":
                atomsDict[dtype[0] + "_" + part[:2]] = np.float32(
                    getattr(tempDict[dtype[0]], part)
                )
    return atomsDict


def reshape_FstatAtomVector_to_array(atomsVector):
    """Reshape a FstatAtomVector into an (Ntimestamps,8) np.ndarray.

    Parameters
    ----------
    atomsVector: lalpulsar.FstatAtomVector
        The atoms in a 'vector'-like structure:
        iterating over timestamps as the higher hierarchical level,
        with a set of 'atoms' quantities defined at each timestamp.

    Returns
    -------
    atoms_for_h5: np.ndarray
        Array of the atoms, with shape (Ntimestamps,8).
    """
    atomsDict = reshape_FstatAtomVector_to_dict_of_arrays(atomsVector)
    atomsArray = np.array(list(atomsDict.values())).transpose()
    return atomsArray
