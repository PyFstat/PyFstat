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
    singleIFOmultiFatoms:
        Length-1 MultiFstatAtomVector with only the data for detector `X`.
    """
    if X > multiAtoms.length:
        raise ValueError(
            f"Detector index {X} is out of range for multiAtoms of length {multiAtoms.length}."
        )
    singleIFOmultiFatoms = lalpulsar.CreateMultiFstatAtomVector(1)
    singleIFOmultiFatoms.data[0] = lalpulsar.CreateFstatAtomVector(
        multiAtoms.data[X].length
    )
    singleIFOmultiFatoms.data[0].TAtom = multiAtoms.data[X].TAtom
    # we deep-copy the atoms data,
    # since just assigning the whole array can cause a segfault
    # from memory cleanup
    # in looping over this function
    # singleIFOmultiFatoms.data[0].data = (
    #     multiAtoms.data[X].data
    # )
    singleIFOmultiFatoms.data[0] = copy_FstatAtomVector(
        singleIFOmultiFatoms.data[0], multiAtoms.data[X]
    )
    return singleIFOmultiFatoms


def copy_FstatAtomVector(
    dest: lalpulsar.FstatAtomVector, src: lalpulsar.FstatAtomVector
) -> lalpulsar.FstatAtomVector:
    """Deep-copy an FstatAtomVector with all its per-SFT FstatAtoms.

    The two vectors must have the same length,
    and the destination vector must already be allocated.

    Parameters
    -------
    dest:
        The destination vector to copy to.
        Must already be allocated.
    src:
        The source vector to copy from.
    Returns
    -------
    dest:
        The updated destination vector.
    """
    if dest.length != src.length:
        raise ValueError(
            f"Lengths of destination and source vectors do not match. ({dest.length} != {src.length})"
        )
    for k in range(dest.length):
        dest.data[k] = copy_FstatAtom(dest.data[k], src.data[k])
    return dest


def copy_FstatAtom(
    dest: lalpulsar.FstatAtom, src: lalpulsar.FstatAtom
) -> lalpulsar.FstatAtom:
    """Deep-copy an FstatAtom with all its fields.

    Parameters
    -------
    dest:
        The destination atom object to copy to.
    src:
        The source atom object to copy from.
    Returns
    -------
    dest:
        The updated destination atom object.
    """
    for key in [
        "timestamp",
        "a2_alpha",
        "b2_alpha",
        "ab_alpha",
        "Fa_alpha",
        "Fb_alpha",
    ]:
        setattr(
            dest,
            key,
            getattr(src, key),
        )
    return dest
