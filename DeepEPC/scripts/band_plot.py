import numpy as np
from ase.io import read

Ry2eV = 13.605698065894
Hartree2eV = Ry2eV*2
Ef = -0.19014862296705*Hartree2eV
kpoint = [[0,0,0],[0.0,0.5,0.0],[0.5,0.5,0.0],[0.0,0.0,0.0],[0.0,0.0,0.5],[0.0,0.5,0.5],[0.5,0.5,0.5],[0.0,0.0,0.5]]
klabel = ['$\Gamma$','X','M','$\Gamma$','Z','R','A','Z']
knum = [100,100,100,100,100,100,100]

band = np.load('band_val_k-50.npy')-Ef

pos = read('555/POSCAR_u')
natom = pos.positions.shape[0]
nmodes = natom*3

# get k_list
nk = sum(knum)
kpos0 = np.zeros((len(knum)+1),dtype=int)
kpos0[1:] = np.cumsum(knum)
k_list = np.zeros((nk,3))
for i in range(len(knum)):
    k_list[kpos0[i]:kpos0[i+1]] = \
    np.linspace(kpoint[i],kpoint[i+1],knum[i],endpoint=False)
kpos0[-1] -= 1

# get rabc
abc = pos.cell[:]
tmp = np.cross(abc[1],abc[2])
CellV = np.dot(abc[0],tmp)

rabc = np.zeros((3,3))
tmp = np.cross(abc[1],abc[2])
rabc[0] = 2.0*np.pi*tmp/CellV
tmp = np.cross(abc[2],abc[0])
rabc[1] = 2.0*np.pi*tmp/CellV
tmp = np.cross(abc[0],abc[1])
rabc[2] = 2.0*np.pi*tmp/CellV

kxyz = np.dot(k_list,rabc)
klen = np.linalg.norm(kxyz[1:]-kxyz[0:-1],axis=-1)
kpos = np.zeros((nk))
kpos[1:] = np.cumsum(klen)

label_xcoords = kpos[kpos0]

import matplotlib.pyplot as plt
prop_cycle = plt.rcParams['axes.prop_cycle']
mpl_clrs   = prop_cycle.by_key()['color']

fig = plt.figure(1, figsize=(7, 4),dpi=300)
ax = fig.add_axes([.1, .1, .85, .85])

emin = -4
emax = 4
for i in range(band.shape[1]):
    if i>=19 and i<23:
        ax.plot(kpos,band[:,i],lw=1.0,c='black')
    else:
        ax.plot(kpos,band[:,i],lw=0.6,c='grey')
for i in [20,21]:
    ax.plot(kpos,band[:,i],lw=1.5,c='red')

for x in label_xcoords[1:-1]:
    ax.axvline(x,color='0.5',linestyle='--',lw=0.5)
for x in range(emin+1,emax):
    ax.axhline(x,color='0.5',linestyle='--',lw=0.5)

ax.set_xticks(label_xcoords)
ax.set_xticklabels(klabel)
ax.set_ylabel('Energy (eV)')
ax.axis(xmin=0,xmax=kpos[-1],ymin=emin,ymax=emax)

fig.savefig('band.png')

