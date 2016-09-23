# Fully coherent search on glitching data using MCMC

This example applies the basic [fully coherent
search](fully_coherent_search.md), to the glitching signal data set created in
[make fake data](make_fake_data.md]). The aim here is to illustrate the effect
such a signal can have on a fully-coherent search. The complete script for this
example canbe found
[here](../example/fully_cohrent_search_on_glitching_data.py).


We use the same prior as in the basic fully-coherent search, except that we
will modify the prior on `F0` to a flat uniform prior. The reason for this is
to highlight the multimodal nature of the posterior which results from the
glitch (a normal prior centered on one of the modes will bias one mode over
the other). So our initial set up is

```
from pyfstat import MCMCSearch

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 5e-3
Delta = 6e-2
tref = 362750407.0

tstart = 1000000000
duration = 100*86400
tend = tstart = duration

theta_prior = {'F0': {'type': 'unif', 'lower': F0-5e-5,
                      'upper': F0+5e-5},
               'F1': {'type': 'norm', 'loc': F1, 'scale': abs(1e-6*F1)},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta
               }
```

Next, we will use 10 temperatures and a larger number of walkers - these have
been tuned to illustrate the bimodal nature of the posterior

```
ntemps = 10
betas = np.logspace(0, -30, ntemps)
nwalkers = 500
nsteps = [100, 100, 100]

mcmc = MCMCSearch('fully_coherent_on_glitching_data', 'data',
                  sftlabel='glitch', sftdir='data',
                  theta_prior=theta_prior, tref=tref, tstart=tstart, tend=tend,
                  nsteps=nsteps, nwalkers=nwalkers, ntemps=ntemps, betas=betas,
                  scatter_val=1e-6)

mcmc.run()
mcmc.plot_corner(add_prior=True)
```

Running this takes slightly longer than the basic example (a few minutes) and
produces a multimodal posterior:
![](img/fully_coherent_on_glitching_data_corner.png)

Clearly one central peak pertains to the original frequency `F0=30`. At
`30+0.4e-5`, we find the second largest peak - this is the mode corresponding
to the second half of the data. In reality, we would expect both peaks to be
of equal size since the glitch occurs exactly half way through (see [how the
data was made](make_fake_data.md)). We will confirm this later on by performing
a grid search.

Finally, the maximum twoF value found is

```
>>> mcmc.print_summary()
Max twoF: 411.595
```
That is, compared to the basic search (on a smooth signal) which had a twoF of
`1756.44177246` (in agreement with the predicted twoF), we have lost a large
fraction of the SNR due to the glitch.

