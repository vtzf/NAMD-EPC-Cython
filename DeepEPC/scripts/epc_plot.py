import numpy as np

TEMP = 300
PHCUT = 0.001
sigma = 0.02
nq = [10,10,10]

Ry2eV = 13.605698065894
Hartree2eV = 2.0*Ry2eV

Efermi = -0.22836870642535*Hartree2eV
phval = np.ascontiguousarray(np.load('7_no_so/phonon/val-10.npy').T)
phval[np.where(phval<0)] = 1e-10
nmodes = phval.shape[0]
bassel_up = np.load('7_no_so/epc_band/bassel-10_up.npy')
Nup = bassel_up.shape[0]
bassel_dn = np.load('7_no_so/epc_band/bassel-10_dn.npy')
Ndn = bassel_dn.shape[0]
nbasis = Nup+Ndn
kidx = np.zeros((nbasis),dtype=np.int32)
kidx[0:Nup] = bassel_up[:,0]
kidx[Nup:nbasis] = bassel_dn[:,0]
band = np.zeros((nbasis),dtype=float)
band[0:Nup] = np.load('7_no_so/epc_band/band_val_k-10_up_p.npy')
band[Nup:nbasis] = np.load('7_no_so/epc_band/band_val_k-10_dn_p.npy')
spin = np.zeros((nbasis),dtype=float)
spin[0:Nup] = 1.0
spin[Nup:nbasis] = -1.0

epc = np.zeros((nbasis,nbasis,nmodes),dtype=complex)
epc[0:Nup,0:Nup] = np.fromfile('7_no_so/epc_all/epc_all_p-10_up.dat',dtype=complex).reshape(Nup,Nup,nmodes)
epc[Nup:nbasis,Nup:nbasis] = np.fromfile('7_no_so/epc_all/epc_all_p-10_dn.dat',dtype=complex).reshape(Ndn,Ndn,nmodes)

#Efermi = -0.22829832315795*Hartree2eV
#phval = np.ascontiguousarray(np.load('7_so/phonon/val-10.npy').T)
#phval[np.where(phval<0)] = 1e-10
#nmodes = phval.shape[0]
#band = np.load('7_so/epc_band/band_val_k-10_p.npy')
#bassel = np.load('7_so/epc_band/bassel-10.npy')
#nbasis = bassel.shape[0]
#kidx = np.zeros((nbasis),dtype=np.int32)
#kidx[:] = bassel[:,0]
#spinDM = np.load('7_so/epc_band/spinDM-10_p.npy')
#spin = spinDM[:,0,0].real-spinDM[:,1,1].real
#epc = np.fromfile('7_so/epc_all/epc_all_p-10.dat',dtype=complex).reshape(nbasis,nbasis,nmodes)

epc2 = np.empty((nmodes,nbasis,nbasis),dtype=float)
for i in range(nmodes):
    epc_t = epc[:,:,i]
    epc_t1 = (epc_t+epc_t.copy().T.conj())/2.0
    epc2[i] = (epc_t1.conj()*epc_t1).real
epc = None

Kb_eV = 8.6173857E-5
KbT = Kb_eV * TEMP
hbar = 0.6582119281559802

# get phval_k
kqidx = np.zeros((nbasis,nbasis),dtype=int)
kidx_xyz = np.zeros((nbasis,3),dtype=int)
kidx_xyz[:,2] = kidx%nq[2]; kidx_xy = kidx//nq[2]
kidx_xyz[:,1] = kidx_xy%nq[1]; kidx_xyz[:,0] = kidx_xy//nq[1]
dkidx_xyz = (kidx_xyz[:,None]-kidx_xyz[:None])%nq
dkidx = (dkidx_xyz[:,:,0]*nq[1]+dkidx_xyz[:,:,1])*nq[2]+dkidx_xyz[:,:,2]
phval_k = np.ascontiguousarray(phval[:,dkidx])
# epc PHCUT
phidx = np.where(phval_k<PHCUT)
# calc coup
dE = np.ascontiguousarray(band[:,None]-band[None])
dEph = np.zeros((nbasis,nbasis))
expdEph = np.zeros((nbasis,nbasis))
epc2[phidx[0],phidx[1],phidx[2]] = 0.0
coup = np.zeros((nbasis,nbasis))

bose = 1.0/(np.exp(phval_k/KbT)-1.0)
BEfactor = bose+0.5

for i in range(nmodes):
    dEph[:] = dE+phval_k[i]+1e-8
    expdEph[:] = np.exp((-0.5/(sigma*sigma))*(dEph*dEph))
    dEph[:] = dE-phval_k[i]-1e-8
    expdEph[:] += np.exp((-0.5/(sigma*sigma))*(dEph*dEph))
    coup += epc2[i]*expdEph*BEfactor[i]

epc2 = None
coup *= np.sqrt(2*np.pi)/sigma/(nq[0]*nq[1]*nq[2])
#coup = np.load('namd/output/epcec-0.npy')

# band energy reorder
sorder = band.argsort()

band_s = band[sorder]
spin_s = spin[sorder]
coup_s = coup.copy()[sorder]
coup_s = coup_s[:,sorder]
coup = None

# spin up,dn reorder
up_order = np.where(spin_s>=0)[0]
dn_order = np.where(spin_s<0)[0]
Nup = up_order.shape[0]
Ndn = nbasis-Nup
print(Nup,Ndn)
#print(up_order,dn_order)

sorder1 = np.zeros((nbasis),dtype=int)
sorder1[0:Nup] = up_order
sorder1[Nup:nbasis] = dn_order

band_s1 = band_s[sorder1]
spin_s1 = spin_s[sorder1]
coup_s1 = coup_s.copy()[sorder1]
coup_s1 = coup_s1[:,sorder1]
coup_s = None

np.fill_diagonal(coup_s1,0)
logcoup = np.log10(coup_s1+1e-10)
print(np.average(coup_s1[0:Nup,0:Nup]),np.average(coup_s1[0:Nup,Nup:nbasis]),np.average(coup_s1[Nup:nbasis,0:Nup]),np.average(coup_s1[Nup:nbasis,Nup:nbasis]))
print(np.max(coup_s1[0:Nup,0:Nup]),np.max(coup_s1[0:Nup,Nup:nbasis]),np.max(coup_s1[Nup:nbasis,0:Nup]),np.max(coup_s1[Nup:nbasis,Nup:nbasis]))
coup_s1 = None

import matplotlib as mpl; mpl.use('agg')
import matplotlib.pyplot as plt

fig = plt.figure()
figsize_x = 4.8
figsize_y = 3.6 # in inches
fig.set_size_inches(figsize_x, figsize_y)

cmap = 'bwr'
Bmin = 0.5; Bmax = nbasis + 0.5
cmin = -4; cmax = 0
norm = mpl.colors.Normalize(cmin,cmax)

plt.imshow(logcoup, cmap=cmap, origin='lower', norm=norm,
    extent=(Bmin,Bmax,Bmin,Bmax), interpolation='none',zorder=1)

plt.axvline(Nup,color='0.0',linestyle='--',lw=0.5,zorder=2)
plt.axhline(Nup,color='0.0',linestyle='--',lw=0.5,zorder=2)

cbar = plt.colorbar()
cbar.set_label('Coupling (eV)')
cbar.set_ticks([-4,-3,-2,-1,0])
cbar.set_ticklabels(['10$^{-4}$','10$^{-3}$','10$^{-2}$','10$^{-1}$','10$^{0}$'])
plt.tight_layout()
plt.savefig('epc.png', dpi=300)
plt.close(fig)
