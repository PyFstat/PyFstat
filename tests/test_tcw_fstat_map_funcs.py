import logging
import os
import shutil

import lalpulsar
import numpy as np
import pytest

import pyfstat


@pytest.mark.parametrize("snr", [0, 10])
@pytest.mark.parametrize("window", ["rect", "exp"])
@pytest.mark.parametrize("tCWFstatMapVersion", ["lal", "pycuda"])
def test_compute_transient_fstat_map(tCWFstatMapVersion, window, snr):
    logging.info("Initialising transient FstatMap features...")
    features = pyfstat.tcw_fstat_map_funcs._get_transient_fstat_map_features()
    if tCWFstatMapVersion == "pycuda" and not features[tCWFstatMapVersion]:
        pytest.skip(f"Feature {tCWFstatMapVersion} not available.")
    (
        tCWFstatMapFeatures,
        gpu_context,
    ) = pyfstat.tcw_fstat_map_funcs.init_transient_fstat_map_features(
        tCWFstatMapVersion
    )

    outdir = "TestData"
    # ensure a clean working directory
    if os.path.isdir(outdir):
        try:
            shutil.rmtree(outdir)
        except OSError:
            logging.warning(f"{outdir} not removed prior to tests.")
    os.makedirs(outdir, exist_ok=True)

    Tstart = 700000000
    Tsft = 1800
    day = 86400
    duration = day
    windowRange = lalpulsar.transientWindowRange_t()
    windowRange.type = 1
    windowRange.t0 = Tstart
    windowRange.t0Band = duration - 2 * Tsft
    windowRange.dt0 = Tsft
    windowRange.tau = 2 * Tsft
    windowRange.tauBand = duration - 2 * Tsft
    windowRange.dtau = Tsft

    logging.info("Creating synthetic atoms...")
    statsfile = os.path.join(outdir, "synthTS_H1L_stats1.dat")
    atomsfile = os.path.join(outdir, "synthTS_H1L_atoms")
    pyfstat.utils.run_commandline(
        f"lalpulsar_synthesizeTransientStats --fixedSNR={snr} --IFOs=H1 --dataStartGPS {Tstart} --dataDuration {duration} --injectWindow-type={window} --injectWindow-tauDays={0.25} --injectWindow-tauDaysBand=0 --injectWindow-t0Days={0.25} --injectWindow-t0DaysBand=0 --searchWindow-type={window} --searchWindow-tauDays={windowRange.tau/day} --searchWindow-tauDaysBand={windowRange.tauBand/day} --searchWindow-t0Days={0} --searchWindow-t0DaysBand={windowRange.t0Band/day} --computeFtotal=TRUE --numDraws=1 --randSeed=1 --outputStats={statsfile} --outputAtoms={atomsfile}"
    )
    atomsfile += "_0001_of_0001.dat"

    logging.info(f"Loading atoms from {atomsfile} ...")
    atoms_in = pyfstat.utils.read_txt_file_with_header(
        atomsfile,
        comments="%%",
    )
    multiFatoms = lalpulsar.CreateMultiFstatAtomVector(1)
    multiFatoms.data[0] = lalpulsar.CreateFstatAtomVector(len(atoms_in))
    multiFatoms.data[0].TAtom = Tsft
    for ts in range(0, len(atoms_in)):
        multiFatoms.data[0].data[ts].timestamp = int(atoms_in[ts][0])
        multiFatoms.data[0].data[ts].a2_alpha = float(atoms_in[ts][1])
        multiFatoms.data[0].data[ts].b2_alpha = float(atoms_in[ts][2])
        multiFatoms.data[0].data[ts].ab_alpha = float(atoms_in[ts][3])
        multiFatoms.data[0].data[ts].Fa_alpha = float(atoms_in[ts][4]) + 1j * float(
            atoms_in[ts][5]
        )
        multiFatoms.data[0].data[ts].Fb_alpha = float(atoms_in[ts][6]) + 1j * float(
            atoms_in[ts][7]
        )

    logging.info("Computing transient FtatMap...")
    (
        FstatMap,
        timingFstatMap,
    ) = pyfstat.tcw_fstat_map_funcs.call_compute_transient_fstat_map(
        version=tCWFstatMapVersion,
        features=tCWFstatMapFeatures,
        multiFstatAtoms=multiFatoms,
        windowRange=windowRange,
        BtSG=True,
    )
    assert not np.isnan(FstatMap.maxF)
    assert not np.isnan(FstatMap.lnBtSG)

    logging.info(f"Loading {statsfile} for comparison...")
    stats = pyfstat.utils.read_txt_file_with_header(
        statsfile,
        comments="%%",
    )
    # Keitel&Ashton 2018: differences for exp can reach ~10% due to lalpulsar's lookup table
    reltol = 0.15 if window == "exp" else 0.01
    assert pytest.approx(2 * FstatMap.maxF, rel=reltol) == stats["maxTwoF"]
    assert pytest.approx(FstatMap.lnBtSG, rel=reltol) == stats["logBstat"]
    if snr > 0:
        assert (
            pytest.approx(FstatMap.t0_ML, abs=4 * Tsft)
            == stats["t0_MLd"] * day + Tstart
        )
        assert pytest.approx(FstatMap.tau_ML, abs=4 * Tsft) == stats["tau_MLd"] * day
    if window == "rect":
        # first t0 / last tau entry of F_mn should correspond to total F-stat
        # (which for historical reasons is hacked into "fkdot3" column of synth stats file)
        assert pytest.approx(2 * FstatMap.F_mn[0, -1], rel=reltol) == stats["fkdot3"]

    # cleanup
    if gpu_context:
        logging.info("Detaching GPU context...")
        gpu_context.detach()
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
