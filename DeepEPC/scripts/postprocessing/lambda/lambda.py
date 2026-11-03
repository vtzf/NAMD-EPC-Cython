# calculate lambda
import os
import numpy as np
import math
from ase.io import read
import matplotlib.pyplot as plt
import configparser
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

conf = configparser.ConfigParser()
conf.read('postprocessing.ini',encoding='utf-8')

nq_str = conf['main']['nq']
qmesh = np.array([int(i) for i in nq_str[1:-1].split(',')],dtype=np.int32)
kmesh = qmesh

Ry2eV = 13.605698065894
Hartree2eV = Ry2eV*2
Ef = float(conf['main']['Ef'])*Hartree2eV
Nef = float(conf['lambda']['Nef'])
nmodes = int(conf['main']['nmodes'])
nbands = int(conf['main']['nbands'])
sigma = float(conf['lambda']['sigma'])
fthick = float(conf['lambda']['fthick'])
ph_cutoff = float(conf['main']['ph_cutoff'])
QBLOCK = 4096

Ispin = int(conf['main']['Ispin'])
spin = conf['main']['spin'].strip()
inDir = conf['main']['inDir']
bandDir = conf['main']['bandDir']
phononDir = conf['main']['phononDir']
epcDir = conf['main']['epcDir']
valname = conf['main']['valname']
phvalname = conf['main']['phvalname']
basselname = conf['main']['basselname']
epcname = conf['main']['epcname']

def spin_name(name):
    stem,ext = os.path.splitext(name)
    suffix = '_'+spin if Ispin==1 else ''
    return stem+suffix+ext

k_all_list = np.genfromtxt('k_all_list')
q_list = np.genfromtxt('q_list')
omega_q = np.load(os.path.join(inDir,phononDir,phvalname))
band = np.load(os.path.join(inDir,bandDir,spin_name(valname)))
bassel = np.load(os.path.join(inDir,bandDir,spin_name(basselname)))
epc_path = os.path.join(inDir,epcDir,spin_name(epcname))
qnum = len(q_list)
knum = kmesh[0]*kmesh[1]*kmesh[2]
knum_fs = len(bassel)
weight = 2/knum
band_pname = os.path.splitext(spin_name(valname))[0]+'_p.npy'
band_p = np.load(os.path.join(inDir,bandDir,band_pname))

q_indices = np.arange(qnum)
q_split = np.array_split(q_indices,size)
my_q = q_split[rank]
nq_my = len(my_q)
if rank==0:
    lbda_qnu = np.zeros((qnum,nmodes))

def whichk(nk,nq):
    kplusq = np.mod(k_all_list[nk]+q_list[nq][0:3],1.0)
    kx = kplusq[0]*kmesh[0]
    ky = kplusq[1]*kmesh[1]
    kz = kplusq[2]*kmesh[2]
    return round(kz+(kx*kmesh[1]+ky)*kmesh[2])

def guass(E):
    xi = E-Ef
    return np.exp(-xi*xi/(2*sigma*sigma))/(math.sqrt(2*np.pi)*sigma)

def lambda_qblock(qrows):
    lbda_qnu_tmp = np.zeros((len(qrows),nmodes))
    with open(epc_path,'rb') as f:
        for nk in range(knum_fs):
            ik = int(bassel[nk][0])
            iband = int(bassel[nk][1])
            delta_k = guass(band_p[nk])
            count = len(qrows)*nmodes*nbands*nbands
            offset = (ik*qnum+qrows[0])*nmodes*nbands*nbands*16
            f.seek(offset,os.SEEK_SET)
            raw = np.fromfile(f,dtype=np.complex128,count=count)
            if raw.size!=count:
                raise IOError(f'Unexpected EPC size in {epc_path}: k={ik}, q={qrows[0]}')
            g_abs = np.abs(raw).reshape(len(qrows),nmodes,nbands,nbands)
            for j,nq in enumerate(qrows):
                iq = int(q_list[nq][4])
                nkq = whichk(ik,nq)
                for nmode in range(nmodes):
                    if omega_q[iq][nmode]>ph_cutoff:
                        for jb in range(nbands):
                            if abs(band[nkq][jb]-Ef)<fthick:
                                delta_kq = guass(band[nkq][jb])
                                lbda_qnu_tmp[j,nmode] += g_abs[j,nmode,jb,iband]**2*delta_k*delta_kq*weight/(Nef*omega_q[iq][nmode])
    return lbda_qnu_tmp

lbda_qnu_local = np.zeros((nq_my,nmodes))
for start in range(0,nq_my,QBLOCK):
    end = min(start+QBLOCK,nq_my)
    lbda_qnu_local[start:end] = lambda_qblock(my_q[start:end])

lbda_list = comm.gather(lbda_qnu_local,root=0)
if rank==0:
    for proc_id,local_lbda in enumerate(lbda_list):
        qs = q_split[proc_id]
        lbda_qnu[qs,:] = local_lbda
    lbda = 0
    for nq in range(qnum):
        for nmode in range(nmodes):
            lbda += q_list[nq][3]*lbda_qnu[nq][nmode]/knum
    fname = f"lambda_qnu_{kmesh[0]}"
    with open(fname,'w') as f:
        f.write('#      qx        qy         qz    weight nmode    omega(meV)   lambda_qnu\n')
        for nq in range(qnum):
            iq = int(q_list[nq][4])
            for nmode in range(nmodes):
                line = f"{q_list[nq][0]:10.8f} {q_list[nq][1]:10.8f} {q_list[nq][2]:10.8f} {int(q_list[nq][3]):5d} {nmode+1:5d} {omega_q[iq][nmode]*1000:12.8f} {lbda_qnu[nq][nmode]:16.8f}"
                f.write(line+'\n')
        f.write(f'# sigma(eV) : {sigma}\n')
        f.write(f'# phonon cut off(meV) : {ph_cutoff*1000}\n')
        f.write(f'# fthick(eV) : {fthick}\n')
        f.write(f'# total lambda = {lbda}\n')
