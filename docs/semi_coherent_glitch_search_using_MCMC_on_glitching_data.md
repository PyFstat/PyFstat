# Semi-coherent glitch search on data with a single glitch using MCMC

In this example, based on [this
script](../examples/semi_coherent_glitch_search_using_MCMC.py), we show the
basic setup for a single-glitch search. We begin, in the usual way, with
defining some the prior

```python
import pyfstat

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 5e-3
Delta = 6e-2
tref = 362750407.0

tstart = 1000000000
duration = 100*86400
tend = tstart + duration

theta_prior = {'F0': {'type': 'norm', 'loc': F0, 'scale': abs(1e-6*F0)},
               'F1': {'type': 'norm', 'loc': F1, 'scale': abs(1e-6*F1)},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta,
               'delta_F0': {'type': 'halfnorm', 'loc': 0,
                            'scale': 1e-5*F0},
               'delta_F1': 0,
               'tglitch': {'type': 'unif',
                           'lower': tstart+0.1*duration,
                           'upper': tstart+0.9*duration},
               }
```

For simplicity, we have chosen a prior based on the known inputs. The important
steps here are the definition of `delta_F0`, `delta_F1` and `tglitch`, the
prior densities for the glitch-parameters. We then use a parallel-tempered
set-up, in addition to an initialisation step and run the search:
```python
ntemps = 4
log10temperature_min = -1
nwalkers = 100
nsteps = [5000, 1000, 1000]

mcmc = pyfstat.MCMCGlitchSearch(
    'semi_coherent_glitch_search_using_MCMC', 'data',
    sftfilepath='data/*_glitch*sft', theta_prior=theta_prior, tref=tref,
    tstart=tstart, tend=tend, nsteps=nsteps, nwalkers=nwalkers,
    scatter_val=1e-10, nglitch=1, ntemps=ntemps,
    log10temperature_min=log10temperature_min)

mcmc.run()
mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
```

The posterior for this search demonstrates that we recover the input parameters:
![](img/semi_coherent_search_using_MCMC_corner.png)
