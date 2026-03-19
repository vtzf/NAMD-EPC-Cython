import numpy as np
from ase.io import read

Ry2eV = 13.605698065894
Hartree2eV = Ry2eV*2
eV2cm1 = 8.06554815355e3

knum = [100,100,100,100,100,100]
klabel = ['$\Gamma$','X','W','X','U|K','$\Gamma$','L']
kpoint = [[0,0,0],[1/2,0,1/2],[1/2,1/4,3/4],[1/2,0,1/2],[5/8,1/4,5/8],[0,0,0],[1/2,1/2,1/2]]

band = np.load('7_no_so/phonon/val_path.npy')*eV2cm1
pos = read('7_no_so/POSCAR_u')

#band = np.load('7_so/phonon/val_path.npy')*eV2cm1
#pos = read('7_so/POSCAR_u')

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

plt.figure(
    figsize = (6, 4),
    dpi = 300,
    constrained_layout = True
)

for i in range(nmodes):
    plt.plot(kpos,band[:,i],lw=1.5,c=mpl_clrs[0])

plt.axis([0,kpos[-1],-30,350])
ytick_pos = np.arange(0,351,50)
plt.yticks(ytick_pos,fontsize=14)
plt.xticks(label_xcoords,klabel,fontsize=14)
plt.xlabel('Kpath',fontsize=16)
plt.ylabel('Energy (cm$^{-1}$)',fontsize=16)
plt.grid('on', ls='--', lw=1, alpha=0.8, color='grey',which='major', axis='both')

plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)

plt.savefig('phonon.png')

