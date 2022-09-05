import os
import shutil
from glob import glob

from pyfstat import set_up_logger
from pyfstat.utils import run_commandline

exit_on_first_failure = False

basedir = os.path.abspath(os.path.dirname(__file__))

outdir = "PyFstat_example_data"
logger = set_up_logger(outdir=outdir, label="run_all_examples")

# make sure we start from a clean output directory
# and scripts don't just recycle old output
if os.path.isdir(outdir):
    logger.info(f"Removing old output directory {outdir}...")
    shutil.rmtree(outdir)


# In some examples directories, scripts must be executed in a certain order.
# Those need to be manually maintained here.
# For the entries you can lstrip("PyFstat_example_") and rstrip(".py").
ordered_cases = {
    "glitch_examples": [
        "make_data_for_search_on_1_glitch",
        "standard_directed_MCMC_search_on_1_glitch",
        "glitch_robust_directed_grid_search_on_1_glitch",
        "glitch_robust_directed_MCMC_search_on_1_glitch",
    ],
    "transient_examples": [
        "make_data_for_short_transient_search",
        "make_data_for_long_transient_search",
        "short_transient_grid_search",
        "short_transient_MCMC_search",
        "long_transient_MCMC_search",
    ],
}

Nscripts = 0
failures = []
for case in os.listdir(basedir):
    exdir = os.path.join(basedir, case)
    if os.path.isdir(exdir):
        if case in ordered_cases:
            scripts = [
                os.path.join(exdir, f"PyFstat_example_{step}.py")
                for step in ordered_cases[case]
            ]
        else:
            scripts = sorted(glob(os.path.join(exdir, "PyFstat_example_*.py")))
        logger.info(
            f"Executing {len(scripts)} script(s) in example directory {exdir}..."
        )
        for script in scripts:
            Nscripts += 1
            cl = "python " + script
            logger.info(f"Running: {script}")
            try:
                run_commandline(cl, return_output=False)
            except Exception as e:
                logger.info(f"FAILED to run {script}")
                failures.append(script)
                if exit_on_first_failure:
                    logger.info("Exception was:")
                    logger.info(e)
                    raise RuntimeError("Exiting on first failure as requested.")
                else:
                    logger.info("\n")
            else:
                logger.info(f"Successfully ran: {script}\n")
        logger.info("")

if len(failures) > 0:
    raise RuntimeError(
        f"Failed to run {len(failures)}/{Nscripts} example scripts:\n"
        + "\n".join(failures)
    )
else:
    logger.info(f"Successfully ran {Nscripts} example scripts.")
