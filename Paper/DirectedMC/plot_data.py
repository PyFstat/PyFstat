import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats

Tspan = 100 * 86400


def Recovery(Tspan, Depth, twoFstar=60):
    rho2 = 4*Tspan/25./Depth**2
    twoF_Hs = scipy.stats.distributions.ncx2(df=4, nc=rho2)
    return 1 - twoF_Hs.cdf(twoFstar)

results_file_name = 'MCResults.txt'

df = pd.read_csv(
    results_file_name, sep=' ', names=['depth', 'h0', 'dF0', 'dF1',
                                       'twoF_predicted', 'twoF'])

twoFstar = 60
depths = np.unique(df.depth.values)
recovery_fraction = []
recovery_fraction_std = []
for d in depths:
    twoFs = df[df.depth == d].twoF.values
    print d, len(twoFs)
    n = float(len(twoFs))
    rf = np.sum(twoFs > twoFstar) / n
    rf_bars = [np.sum(np.concatenate((twoFs[:i-1], twoFs[i:]))>twoFstar)/(n-1)
               for i in range(int(n))]
    var = (n-1)/n * np.sum([(rf_bar - rf)**2 for rf_bar in rf_bars])
    recovery_fraction.append(rf)
    recovery_fraction_std.append(np.sqrt(var))

fig, ax = plt.subplots()
ax.errorbar(depths, recovery_fraction, yerr=recovery_fraction_std)
#ax.plot(depths, recovery_fraction)

depths_smooth = np.linspace(min(depths), max(depths), 100)
recovery_analytic = [Recovery(Tspan, d) for d in depths_smooth]
ax.plot(depths_smooth, recovery_analytic)

ax.set_xlabel(r'Signal depth', size=16)
ax.set_ylabel('Recovered fraction [%]', size=12)
fig.savefig('directed_recovery.png')
