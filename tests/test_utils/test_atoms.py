import lalpulsar
import pytest

from pyfstat.utils import (
    copy_FstatAtomVector,
    extract_singleIFOmultiFatoms_from_multiAtoms,
)


@pytest.fixture
def arbitrary_singleAtoms():
    single_atoms = lalpulsar.CreateFstatAtomVector(5)

    single_atoms.TAtom = 1800

    for i in range(single_atoms.length):

        for attr in [
            "timestamp",
            "a2_alpha",
            "b2_alpha",
            "ab_alpha",
            "Fa_alpha",
            "Fb_alpha",
        ]:
            setattr(single_atoms.data[i], attr, i)

    return single_atoms


@pytest.fixture
def arbitrary_multiAtoms(arbitrary_singleAtoms):
    ma = lalpulsar.CreateMultiFstatAtomVector(1)
    ma.data[0] = arbitrary_singleAtoms
    return ma


def compare_FstatAtomVector(vectorA, vectorB):

    for attr in ["TAtom", "length"]:
        assert getattr(vectorA, attr) == getattr(vectorB, attr)

    for i in range(vectorA.length):

        for attr in [
            "timestamp",
            "a2_alpha",
            "b2_alpha",
            "ab_alpha",
            "Fa_alpha",
            "Fb_alpha",
        ]:
            assert getattr(vectorA.data[i], attr) == getattr(vectorB.data[i], attr)


def test_extract_singleIFOmultiFatoms_from_multiAtoms(
    arbitrary_singleAtoms, arbitrary_multiAtoms
):

    single_atoms = extract_singleIFOmultiFatoms_from_multiAtoms(arbitrary_multiAtoms, 0)
    compare_FstatAtomVector(single_atoms.data[0], arbitrary_multiAtoms.data[0])

    with pytest.raises(ValueError):
        extract_singleIFOmultiFatoms_from_multiAtoms(arbitrary_multiAtoms, 1)


def test_copy_FstatAtomVector(arbitrary_singleAtoms):

    single_atoms = lalpulsar.CreateFstatAtomVector(arbitrary_singleAtoms.length)
    copy_FstatAtomVector(single_atoms, arbitrary_singleAtoms)
    compare_FstatAtomVector(single_atoms, arbitrary_singleAtoms)

    faulty_atoms = lalpulsar.CreateFstatAtomVector(arbitrary_singleAtoms.length + 1)
    with pytest.raises(ValueError):
        copy_FstatAtomVector(faulty_atoms, arbitrary_singleAtoms)
