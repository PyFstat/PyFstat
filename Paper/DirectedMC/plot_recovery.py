import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def Recovery(Tspan, Depth, twoFstar=60):
    rho2 = 4*Tspan/25./Depth**2
    twoF_Hs = scipy.stats.distributions.ncx2(df=4, nc=rho2)
    return 1 - twoF_Hs.cdf(twoFstar)

N = 500
Tspan = np.linspace(0.1, 365*86400, N)
Depth = np.linspace(10, 300, N)

X, Y = np.meshgrid(Tspan, Depth)
X = X / 86400

Z = [[Recovery(t, d) for t in Tspan] for d in Depth]

fig, ax = plt.subplots()
pax = ax.pcolormesh(X, Y, Z, cmap=plt.cm.viridis)
CS = ax.contour(X, Y, Z, [0.95])
plt.clabel(CS, inline=1, fontsize=12, fmt='%s', manual=[(200, 180)])
plt.colorbar(pax, label='Recovery fraction')
ax.set_xlabel(r'$T_{\rm span}$ [days]', size=16)
ax.set_ylabel(r'Depth=$\frac{\sqrt{S_{\rm n}}}{h_0}$', size=14)
ax.set_xlim(min(Tspan)/86400., max(Tspan)/86400.)
ax.set_ylim(min(Depth), max(Depth))

fig.savefig('recovery.png')
