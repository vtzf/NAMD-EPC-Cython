import os
import matplotlib
matplotlib.use('AGG')
import numpy as np
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.abspath(__file__))
data = np.genfromtxt(os.path.join(base,'a2f.dat'))
omega = data[:,0]
alpha2f = data[:,1]
dw = np.diff(omega,prepend=0.0)
lambda_omega = np.concatenate([[0.0],np.cumsum(2*alpha2f*dw/omega)])

fig,ax0 = plt.subplots(figsize=(6,4))
ax0.plot(omega,alpha2f,'k-',linewidth=1.2,label=r'$\alpha^2F(\omega)$')
ax0.set_xlabel(r'$\omega$ (meV)')
ax0.set_ylabel(r'$\alpha^2F(\omega)$')
ax0.grid(color='grey',linestyle='--',linewidth=0.5,alpha=0.6)
ax1 = ax0.twinx()
ax1.plot(np.concatenate([[0.0],omega]),lambda_omega,'r--',linewidth=1.2,label=r'$\lambda(\omega)$')
ax1.set_ylabel(r'$\lambda(\omega)$')
fig.tight_layout()
fig.savefig(os.path.join(base,'a2f.png'),dpi=600,bbox_inches='tight')
