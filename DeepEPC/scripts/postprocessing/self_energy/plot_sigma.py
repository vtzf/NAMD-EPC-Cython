import matplotlib
matplotlib.use('AGG')
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Sigma_nk')
fig,ax = plt.subplots(figsize=(5,3.5))
for ib in np.unique(data[:,3]):
    block = data[data[:,3]==ib]
    ax.plot(block[:,4],block[:,5],linewidth=1.2,marker='o',markersize=3,label=f'band {int(ib)}')
ax.set_xlabel(r'$\epsilon_{nk}-\epsilon_F$ (eV)')
ax.set_ylabel(r'$\mathrm{Re}[\Sigma(\epsilon_{nk})-\Sigma(\epsilon_F)]$ (meV)')
ax.grid(color='grey',linestyle='--',linewidth=0.5,alpha=0.6)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig('Sigma_nk.png',dpi=600,bbox_inches='tight')
