#!/usr/bin/env python

import pyfstat
import os
import numpy as np
import matplotlib.pyplot as plt

outdir = os.path.join("PyFstat_example_data", "PyFstat_example_simple_mcmc_vs_grid")

F0_inj = 30.0
F1_inj = -1e-10

grid = pyfstat.helper_functions.read_txt_file_with_header(
    os.path.join(outdir, "grid_search_NA_GridSearch.txt")
)
mcmc = pyfstat.helper_functions.read_txt_file_with_header(
    os.path.join(outdir, "mcmc_search_samples.dat")
)

grid_maxidx = np.argmax(grid["twoF"])
mcmc_maxidx = np.argmax(mcmc["twoF"])

zoomx = [mcmc["F0"][mcmc_maxidx] - 6e-6, mcmc["F0"][mcmc_maxidx] + 6e-6]
zoomy = [mcmc["F1"][mcmc_maxidx] - 8e-11, mcmc["F1"][mcmc_maxidx] + 8e-11]

plt.plot(grid["F0"], grid["F1"], ".", label="grid")
plt.plot(mcmc["F0"], mcmc["F1"], ".", label="mcmc")
plt.plot(F0_inj, F1_inj, "*k", label="injection")
plt.plot(grid["F0"][grid_maxidx], grid["F1"][grid_maxidx], "+k", label="max2F(grid)")
plt.plot(mcmc["F0"][mcmc_maxidx], mcmc["F1"][mcmc_maxidx], "xk", label="max2F(mcmc)")
plt.xlabel("F0")
plt.ylabel("F1")
plt.legend()
plt.savefig(os.path.join(outdir, "grid_vs_Mcmc_F01F1.png"))

plt.xlim(zoomx)
plt.ylim(zoomy)
plt.savefig(os.path.join(outdir, "grid_vs_Mcmc_F01F1_zoom.png"))
plt.close()

sc = plt.scatter(grid["F0"], grid["F1"], c=grid["twoF"], s=3)
cb = plt.colorbar(sc)
plt.xlabel("F0")
plt.ylabel("F1")
cb.set_label("2F")
plt.title("grid")
plt.plot(F0_inj, F1_inj, "*k", label="injection")
plt.plot(grid["F0"][grid_maxidx], grid["F1"][grid_maxidx], "+k", label="max2F")
plt.legend()
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.savefig(os.path.join(outdir, "grid_F01F1_2F_zoom.png"))
plt.close()

# can't yet do same plot for MCMC because twoF not stored in samples .dat file!
sc = plt.scatter(mcmc["F0"], mcmc["F1"], c=mcmc["twoF"], s=1)
cb = plt.colorbar(sc)
plt.xlabel("F0")
plt.ylabel("F1")
cb.set_label("2F")
plt.title("MCMC")
plt.plot(F0_inj, F1_inj, "*k", label="injection")
plt.plot(mcmc["F0"][mcmc_maxidx], mcmc["F1"][mcmc_maxidx], "xk", label="max2F")
plt.legend()
plt.xlim(zoomx)
plt.ylim(zoomy)
plt.savefig(os.path.join(outdir, "mcmc_F01F1_2F_zoom.png"))
plt.close()
