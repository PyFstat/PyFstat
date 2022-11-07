import logging

import lalpulsar

logger = logging.getLogger(__name__)


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
