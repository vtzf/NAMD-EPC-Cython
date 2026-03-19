import numpy as np
from ase.io import read

knum = [200,200,200,200,200]
klabel = [r'$\Gamma$','L','W','X',r'$\Gamma$','K']
kpoint = [[0,0,0],[1/2,1/2,1/2],[1/2,1/4,3/4],[1/2,0,1/2],[0,0,0],[5/8,1/4,5/8]]

Ry2eV = 13.605698065894
Hartree2eV = Ry2eV*2

#Ef = -0.22836870642535*Hartree2eV
#band_up = np.load('7_no_so/epc_band/band_val_k_path_up.npy')
#band_dn = np.load('7_no_so/epc_band/band_val_k_path_dn.npy')
#band = np.hstack((band_up,band_dn))-Ef
#nbands = band.shape[1]
#spin = np.ones_like(band)
#spin[:,nbands//2:nbands] *= -1.0
#pos = read('7_no_so/POSCAR_u')

Ef = -0.22911249882365*Hartree2eV
band = np.load('7_so/epc_band/band_val_k_path.npy')-Ef
spinDM = np.load('7_so/epc_band/spinDM_path.npy')
spin = spinDM[:,:,0,0].real-spinDM[:,:,1,1].real
pos = read('7_so/POSCAR_u')
nbands = band.shape[1]

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
    figsize = (6, 6),
    dpi = 300,
    constrained_layout = True
)

for i in range(nbands):
    plt.plot(kpos,band[:,i],lw=0.2,c='black',alpha=0.6)

cm = 'coolwarm_r'#Reds, bwr, coolwarm, viridis
xx = np.zeros(band.shape)
for i in range(nbands): xx[:,i] = kpos
ss = plt.scatter(xx,band,c=spin,marker='o',cmap=cm,s=5,vmin=-1.0,vmax=1.0)

cbar = plt.colorbar(ss,pad=0.03)
cbar.set_ticks([-1.0,-0.5,0,0.5,1.0])
cbar.set_ticklabels(['-1.0','-0.5','0.0','0.5','1.0'],fontsize=12)
cbar.set_label(r'S$_\mathrm{z}$',fontsize=14)

plt.axis([0,kpos[-1],-0.5,2.5])
ytick_pos = [0.0,0.5,1.0,1.5,2.0]
plt.yticks(ytick_pos,fontsize=14)
plt.xticks(label_xcoords,klabel,fontsize=14)
plt.xlabel('Kpath',fontsize=16)
plt.ylabel('Energy (eV)',fontsize=16)
plt.grid('on', ls='--', lw=1, alpha=0.8, color='grey',which='major', axis='both')

plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['top'].set_linewidth(1.5)
plt.gca().spines['right'].set_linewidth(1.5)

plt.savefig('band.png')

