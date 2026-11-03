# calculate electron self-energy
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
nk_str = conf['Sigma']['nk']
kmesh = np.array([int(i) for i in nk_str[1:-1].split(',')],dtype=np.int32)

bmin = int(conf['Sigma']['bmin'])
bmax = int(conf['Sigma']['bmax'])
ibnd = bmax-bmin+1
Ry2eV = 13.605698065894
Kelvin2eV = 0.08617/1000
Hartree2eV = Ry2eV*2
Ef = float(conf['main']['Ef'])*Hartree2eV
T = float(conf['Sigma']['T'])*Kelvin2eV
qnum = qmesh[0]*qmesh[1]*qmesh[2]
knum = kmesh[0]*kmesh[1]*kmesh[2]
weight = 1/qnum
nmodes = int(conf['main']['nmodes'])
nbands = int(conf['main']['nbands'])
ph_cutoff = float(conf['main']['ph_cutoff'])
delta = float(conf['Sigma']['delta'])
maxexp = 200
ci = 1j
QBLOCK = 4096

Ispin = int(conf['main']['Ispin'])
spin = conf['main']['spin'].strip()
inDir = conf['main']['inDir']
bandDir = conf['main']['bandDir']
phononDir = conf['main']['phononDir']
epcDir = conf['main']['epcDir']
valname = conf['main']['valname']
phvalname = conf['main']['phvalname']
epcname = conf['main']['epcname']

def spin_name(name):
    stem,ext = os.path.splitext(name)
    suffix = '_'+spin if Ispin==1 else ''
    return stem+suffix+ext

k_all_list = np.genfromtxt('k_all_list')
omega_q = np.load(os.path.join(inDir,phononDir,phvalname))
band = np.load(os.path.join(inDir,bandDir,spin_name(valname)))
kpath = np.genfromtxt('k_list')
q_list = np.genfromtxt('q_list')
epc_path = os.path.join(inDir,epcDir,spin_name(epcname))
numk = len(kpath)

k_indices = np.arange(numk)
k_split = np.array_split(k_indices,size)
my_k = k_split[rank]
nk_my = len(my_k)
if rank==0:
    ReSigma = np.zeros((ibnd,numk),dtype=complex)
    ImSigma = np.zeros((ibnd,numk),dtype=complex)

def whichk(nk,nq):
    kplusq = np.mod(k_all_list[nk]+q_list[nq][0:3],1.0)
    kx = kplusq[0]*kmesh[0]
    ky = kplusq[1]*kmesh[1]
    kz = kplusq[2]*kmesh[2]
    return round(kz+(kx*kmesh[1]+ky)*kmesh[2])

def Bose_dis(wq,T):
    return 1/(np.exp(min(wq/T,maxexp))-1)

def Fermi_dis(E,T):
    return 1/(np.exp(min(abs(E/T),maxexp)*np.sign(E))+1)

def run(nk):
    ik = int(kpath[nk][4])
    ReSigma = np.zeros(ibnd,dtype=complex)
    ImSigma = np.zeros(ibnd,dtype=complex)
    with open(epc_path,'rb') as f:
        for start in range(0,qnum,QBLOCK):
            end = min(start+QBLOCK,qnum)
            iq0 = int(q_list[start][4])
            iq1 = int(q_list[end-1][4])
            count = (iq1-iq0+1)*nmodes*nbands*nbands
            offset = (nk*knum+iq0)*nmodes*nbands*nbands*16
            f.seek(offset,os.SEEK_SET)
            raw = np.fromfile(f,dtype=np.complex128,count=count)
            if raw.size!=count:
                raise IOError(f'Unexpected EPC size in {epc_path}: k={ik}, q={iq0}')
            g_abs = np.abs(raw).reshape(iq1-iq0+1,nmodes,nbands,nbands)
            for nq in range(start,end):
                iq = int(q_list[nq][4])
                nkq = whichk(ik,nq)
                gq = g_abs[iq-iq0]
                for nmode in range(nmodes):
                    if omega_q[nq][nmode]>ph_cutoff:
                        for jb in range(nbands):
                            emkq = band[nkq][jb]
                            boson = np.complex128(Bose_dis(omega_q[nq][nmode],T))
                            fermion = np.complex128(Fermi_dis(emkq-Ef,T))
                            for ib in range(bmin,bmax+1):
                                enk = band[ik][ib]
                                dt1 = enk-emkq+omega_q[nq][nmode]+ci*delta
                                dt2 = dt1-2*omega_q[nq][nmode]
                                g2 = gq[nmode,jb,ib]**2*weight
                                sig_tmp = (boson+fermion)/dt1+(boson-fermion+1)/dt2
                                sig_ef = (boson+fermion)/(dt1-enk+Ef)+(boson-fermion+1)/(dt2-enk+Ef)
                                ImSigma[int(ib-bmin)] += g2*sig_tmp
                                ReSigma[int(ib-bmin)] += g2*(sig_tmp-sig_ef)
    return ReSigma,ImSigma

ReSigma_local = np.zeros((ibnd,nk_my),dtype=complex)
ImSigma_local = np.zeros((ibnd,nk_my),dtype=complex)
for j,k in enumerate(my_k):
    ReSigma_local[:,j],ImSigma_local[:,j] = run(k)

ReSigma_list = comm.gather(ReSigma_local,root=0)
ImSigma_list = comm.gather(ImSigma_local,root=0)
if rank==0:
    for proc_id,local_Re in enumerate(ReSigma_list):
        ks = k_split[proc_id]
        ReSigma[:,ks] = local_Re
    for proc_id,local_Im in enumerate(ImSigma_list):
        ks = k_split[proc_id]
        ImSigma[:,ks] = local_Im
    with open('Sigma_nk','w') as f:
        f.write('#      kx        ky         kz         ib     Xi_nk(eV)     Re(Sigma)(meV)    Im(Sigma)(meV)\n')
        for ib in range(bmin,bmax+1):
            for nk in range(numk):
                ik = int(kpath[nk][4])
                line = f"{kpath[nk][0]:10.8f} {kpath[nk][1]:10.8f} {kpath[nk][2]:10.8f}{ib:8d}{band[ik][ib]-Ef:15.4f}{np.real(ReSigma[int(ib-bmin)][nk])*1000:15.4f}{np.imag(ImSigma[int(ib-bmin)][nk])*1000:15.4f}"
                f.write(line+'\n')
        f.write(f'# phonon cut off(meV) : {ph_cutoff*1000}\n')
        f.write(f'# delta(eV) : {delta}\n')
        f.write(f'# T(K) : {T/Kelvin2eV}\n')
