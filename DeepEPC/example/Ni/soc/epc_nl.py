import readdrSH
import readdrSH_nc
import epcfunc_nl
import epcfunc_nl_nc
import numpy as np
import json
import os, sys
from ase.data import atomic_numbers, atomic_masses
import configparser
import warnings
from mpi4py import MPI

warnings.filterwarnings("ignore", category=DeprecationWarning)
# load config
conf = configparser.ConfigParser()
conf.read('config.ini',encoding='utf-8')

Ispin = int(conf['epc']['Ispin'])
IsH5 = True if conf['epc']['IsH5']=='True' else False
if Ispin == 1 and IsH5 == True:
    print('HDF5 files are now not supported by Ispin=1!')
    sys.exit()

inDir = conf['epc']['inDir']+'/'
infile = conf['epc']['infile_out']
bandDir = conf['epc']['bandDir']+'/'
phononDir = conf['epc']['phononDir']+'/'
epcDir = conf['epc']['epcDir']+'/'
ucellidx_str = conf['epc']['ucellidx']
ucellidx_list = ucellidx_str[1:-1].split(',')
ucellidx = [int(i) for i in ucellidx_list]
EpcType = conf['epc']['EpcType']
if IsH5:
    H5HamName = inDir+conf['epc']['H5HamName']
    nfileham = 1
    H5OlpName = inDir+conf['epc']['H5OlpName']
    nfileolp = int(conf['epc']['H5OlpNum'])
    H5DrName = inDir+conf['epc']['H5DrName']
    nfiledr = int(conf['epc']['H5DrNum'])
else:
    nfileham = 1
    nfileolp = 1
    nfiledr = 1

atom_str = conf['epc']['atom']
atom_list = atom_str[1:-1].split(',')
atom = [int(i) for i in atom_list]
orbital_str = conf['epc']['orbital']
orbital_list = orbital_str[1:-1].split(',')
orbital = [int(i) for i in orbital_list]

IsAllVec = True if conf['epc']['IsAllVec']=='True' else False
IsAllKlist = True if conf['epc']['IsAllKlist']=='True' else False
Mb = int(conf['mpi']['M_BLOCK'])
Nb = int(conf['mpi']['N_BLOCK'])
nm_block = int(conf['mpi']['NMODES_BLOCK'])

hbar = 6.62607015e-34/(2*np.pi)
Hartree2eV = 27.211396641308
Bohr2Ang = 0.529177249
Ang2m = 1e-10
eV2J = 1.602176634e-19
V_c = 299792458#[m/s]
M_C = 1.9927e-26/12#[kg]
m2cm = 100
factor = eV2J/hbar

center = np.array([i//2 for i in ucellidx],dtype='int32')
factor1= np.sqrt(hbar/2/M_C/factor)/Ang2m#[to eV]
ncell = ucellidx[0]*ucellidx[1]*ucellidx[2]
icell = (center[0]*ucellidx[1]+center[1])*ucellidx[2]+center[2]

R_list = np.array(
    [
        [i,j,k] \
        for i in range(ucellidx[0]) \
        for j in range(ucellidx[1]) \
        for k in range(ucellidx[2]) \
    ],
    dtype=np.int32
)-center

atomnum_u = sum(atom)
atom_idx_u = [y for x,y in zip(atom,orbital) for i in range(x)]
atom_idx_sum_u = [sum(atom_idx_u[0:i]) for i in range(atomnum_u+1)]
Np = atom_idx_sum_u[-1]
atom_idx_u = np.array(atom_idx_u,dtype=np.int32)
atom_idx_sum_u = np.array(atom_idx_sum_u,dtype=np.int32)

if Ispin == 0: norb = Np
else: norb = 2*Np
if IsAllVec:
    bmin = 0
    bmax = norb-1
    nbands = norb
else:
    bmin = int(conf['epc']['bmin'])
    bmax = int(conf['epc']['bmax'])
    nbands = bmax-bmin+1

atomnum_s = atomnum_u*ncell
atom_idx_s = [y for j in range(ncell) for x,y in zip(atom,orbital) for i in range(x)]
atom_idx_sum_s = [sum(atom_idx_s[0:i]) for i in range(atomnum_s+1)]
N = atom_idx_sum_s[-1]
atom_idx_s = np.array(atom_idx_s,dtype=np.int32)
atom_idx_sum_s = np.array(atom_idx_sum_s,dtype=np.int32)

nmodes = atomnum_u*3
catom = icell*atomnum_u+np.arange(atomnum_u,dtype=np.int32)

epcname = inDir+epcDir+conf['epc']['epcname'].split('.')[0]
filename = ['','']
if Ispin != 1:
    filename[0] += epcname+'.dat'
else:
    filename[0] += epcname+'_up.dat'
    filename[1] += epcname+'_dn.dat'

def get_mass(Dir):
    idx = [0]*4
    with open(Dir,'r') as f:
        info = f.readlines()
    for i in range(len(info)):
        if info[i] == "<Atoms.SpeciesAndCoordinates\n":
            idx[0] = i
        if info[i] == "Atoms.SpeciesAndCoordinates>\n":
            idx[1] = i
        if info[i] == "<Atoms.UnitVectors\n":
            idx[2] = i
        if info[i] == "Atoms.UnitVectors>\n":
            idx[3] = i

    xyzinfo = np.array(
        [info[i].split() for i in range(idx[0]+1,idx[1])],dtype='U'
    )
    atomlist = xyzinfo[catom,1]

    return np.array([atomic_masses[atomic_numbers[atomlist[i]]] \
           for i in range(atomnum_u)])


def GenKlist(Kpoint_str,nq):
    from math import gcd
    import sys
    k_list = Kpoint_str.replace('[','').replace(']','').split(',')
    k_arr = np.array([eval(i) for i in k_list],dtype=float).reshape(-1,3)
    k_int = np.around(k_arr*nq).astype(int)
    nkpoint = k_arr.shape[0]
    nklist = np.zeros((nkpoint-1),dtype=int)
    for i in range(nkpoint-1):
        dk_int = np.abs(k_int[i+1]-k_int[i])
        if dk_int.max() == 0:
            print('Kpoint error! exit.')
            sys.exit()
        else:
            dk_int_p = dk_int[np.where(dk_int>0)[0]]
            if dk_int_p.shape[0] == 3:
                tmp = gcd(dk_int_p[0],dk_int_p[1])
                tmp1 = gcd(dk_int_p[1],dk_int_p[2])
                nklist[i] = gcd(tmp,tmp1)
            elif dk_int_p.shape[0] == 2:
                nklist[i] = gcd(dk_int_p[0],dk_int_p[1])
            else:
                nklist[i] = dk_int_p[0]

    nkpath = np.sum(nklist)
    nkrange = np.zeros((nkpoint),dtype=int)
    nkrange[1:] = np.cumsum(nklist)
    k_list = np.zeros((nkpath,3),dtype=float)
    for i in range(nkpoint-1):
        k_list[nkrange[i]:nkrange[i+1]] \
        = np.linspace(k_arr[i],k_arr[i+1],nklist[i],endpoint=False)

    return nkpath, k_list


# get comm and shm_comm
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nprocs = comm.Get_size()

shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
shm_id = shm_comm.Get_rank()
shm_nprocs = shm_comm.Get_size()

starttime = MPI.Wtime()
# get sparse matrix info
s_i = 4
s_l = 8
s_d = 8
s_dcplx = 16
if (shm_id==0):
    len_keynum_h = (nfileham+1)*4*s_i
    len_keynum_o = (nfileolp+1)*4*s_i
    len_keynum_dr = (nfiledr+1)*4*s_i
else:
    len_keynum_h = 0
    len_keynum_o = 0
    len_keynum_dr = 0
win00 = MPI.Win.Allocate_shared(len_keynum_h,s_i,comm=shm_comm)
buf00,s_i = win00.Shared_query(0)
key_num_h = np.ndarray(buffer=buf00,dtype=np.int32,shape=(nfileham+1,4))
win01 = MPI.Win.Allocate_shared(len_keynum_o,s_i,comm=shm_comm)
buf01,s_i = win01.Shared_query(0)
key_num_o = np.ndarray(buffer=buf01,dtype=np.int32,shape=(nfileolp+1,4))
win02 = MPI.Win.Allocate_shared(len_keynum_dr,s_i,comm=shm_comm)
buf02,s_i = win02.Shared_query(0)
key_num_dr = np.ndarray(buffer=buf02,dtype=np.int32,shape=(nfiledr+1,4))
shm_comm.Barrier()
if (shm_id==0):
    readdrSH.GetSparseNum(
        (inDir if not IsH5 else 'None').encode('utf-8'),
        (H5HamName if IsH5 else 'None').encode('utf-8'),
        (H5OlpName if IsH5 else 'None').encode('utf-8'),
        (H5DrName if IsH5 else 'None').encode('utf-8'),
        nfileham,nfileolp,nfiledr,atomnum_s,key_num_h,
        key_num_o,key_num_dr,atom_idx_s,IsH5
    )
shm_comm.Barrier()

Nkey_h = key_num_h[nfileham,2]
Nkey_o = key_num_o[nfileolp,2]
Nkey_dr = key_num_dr[nfiledr,2]
Nsparse_h = key_num_h[nfileham,3]
Nsparse_o = key_num_o[nfileolp,3]
Nsparse_dr = key_num_dr[nfiledr,3]
if (shm_id==0):
    len_pubkey_h = Nkey_h*6*s_i
    len_pubkey_o = Nkey_o*6*s_i
    len_pubkey_dr = Nkey_dr*6*s_i
    len_key_h = Nsparse_h*2*s_l
    len_key_o = Nsparse_o*2*s_l
    len_key_dr = Nsparse_dr*2*s_l
    if Ispin != 2:
        len_ham = (Ispin+1)*Nsparse_h*s_d
    else:
        len_ham = 2*4*Nsparse_h*s_d
    len_olp = Nsparse_o*s_d
    len_dr = 3*Nsparse_dr*s_d
else:
    len_pubkey_h = 0
    len_pubkey_o = 0
    len_pubkey_dr = 0
    len_key_h = 0
    len_key_o = 0
    len_key_dr = 0
    len_ham = 0
    len_olp = 0
    len_dr = 0
win03 = MPI.Win.Allocate_shared(len_pubkey_h,s_i,comm=shm_comm)
buf03,s_i = win03.Shared_query(0)
pub_key_h = np.ndarray(buffer=buf03,dtype=np.int32,shape=(Nkey_h,6))
win04 = MPI.Win.Allocate_shared(len_pubkey_o,s_i,comm=shm_comm)
buf04,s_i = win04.Shared_query(0)
pub_key_o = np.ndarray(buffer=buf04,dtype=np.int32,shape=(Nkey_o,6))
win05 = MPI.Win.Allocate_shared(len_pubkey_dr,s_i,comm=shm_comm)
buf05,s_i = win05.Shared_query(0)
pub_key_dr = np.ndarray(buffer=buf05,dtype=np.int32,shape=(Nkey_dr,6))
win06 = MPI.Win.Allocate_shared(len_key_h,s_l,comm=shm_comm)
buf06,s_l = win06.Shared_query(0)
keyinfo_h = np.ndarray(buffer=buf06,dtype=np.int64,shape=(Nsparse_h,2))
win07 = MPI.Win.Allocate_shared(len_key_o,s_l,comm=shm_comm)
buf07,s_l = win07.Shared_query(0)
keyinfo_o = np.ndarray(buffer=buf07,dtype=np.int64,shape=(Nsparse_o,2))
win08 = MPI.Win.Allocate_shared(len_key_dr,s_l,comm=shm_comm)
buf08,s_l = win08.Shared_query(0)
keyinfo_dr = np.ndarray(buffer=buf08,dtype=np.int64,shape=(Nsparse_dr,2))

win = MPI.Win.Allocate_shared(len_olp,s_d,comm=shm_comm)
buf,s_d = win.Shared_query(0)
olp = np.ndarray(buffer=buf,dtype=float,shape=(Nsparse_o))
if Ispin != 2:
    win2 = MPI.Win.Allocate_shared(len_ham,s_d,comm=shm_comm)
    buf2,s_d = win2.Shared_query(0)
    ham = np.ndarray(buffer=buf2,dtype=float,shape=(Ispin+1,Nsparse_h))
else:
    win2 = MPI.Win.Allocate_shared(len_ham,s_d,comm=shm_comm)
    buf2,s_d = win2.Shared_query(0)
    ham = np.ndarray(buffer=buf2,dtype=float,shape=(2,4,Nsparse_h))
win4 = MPI.Win.Allocate_shared(len_dr,s_d,comm=shm_comm)
buf4,s_d = win4.Shared_query(0)
dr = np.ndarray(buffer=buf4,dtype=float,shape=(3,Nsparse_dr))
shm_comm.Barrier()
if (shm_id==0):
    readdrSH.GetSparseIdx(
        (inDir if not IsH5 else 'None').encode('utf-8'),
        (H5HamName if IsH5 else 'None').encode('utf-8'),
        (H5OlpName if IsH5 else 'None').encode('utf-8'),
        (H5DrName if IsH5 else 'None').encode('utf-8'),
        nfileham,nfileolp,nfiledr,atomnum_s,N,key_num_h,
        key_num_o,key_num_dr,pub_key_h,pub_key_o,pub_key_dr,
        keyinfo_h,keyinfo_o,keyinfo_dr,atom_idx_s,atom_idx_sum_s,IsH5
    )
shm_comm.Barrier()
endtime = MPI.Wtime()
if myid == 0:
    print("Reading sparse info time: %.5fs."%(endtime-starttime),flush=True)
starttime = MPI.Wtime()
if Ispin != 2:
    readdrSH.GetSparseData(
        shm_comm,(inDir if not IsH5 else 'None').encode('utf-8'),
        (H5HamName if IsH5 else 'None').encode('utf-8'),
        (H5OlpName if IsH5 else 'None').encode('utf-8'),
        (H5DrName if IsH5 else 'None').encode('utf-8'),
        nfileham,nfileolp,nfiledr,atomnum_s,max(orbital),
        key_num_h,key_num_o,key_num_dr,pub_key_h,pub_key_o,pub_key_dr,
        keyinfo_h,keyinfo_o,keyinfo_dr,ham,olp,dr,Ispin,IsH5
    )
else:
    readdrSH_nc.GetSparseData(
        shm_comm,(inDir if not IsH5 else 'None').encode('utf-8'),
        (H5HamName if IsH5 else 'None').encode('utf-8'),
        (H5OlpName if IsH5 else 'None').encode('utf-8'),
        (H5DrName if IsH5 else 'None').encode('utf-8'),
        nfileham,nfileolp,nfiledr,atomnum_s,max(orbital),
        key_num_h,key_num_o,key_num_dr,pub_key_h,pub_key_o,pub_key_dr,
        keyinfo_h,keyinfo_o,keyinfo_dr,ham,olp,dr,IsH5
    )
# free sparse key_num, pub_key
MPI.Win.Free(win00)
MPI.Win.Free(win01)
MPI.Win.Free(win02)
MPI.Win.Free(win03)
MPI.Win.Free(win04)
MPI.Win.Free(win05)

if shm_id == 0:
    len_olp_keyinfo = Nsparse_o*2*s_i
    len_ham_keyinfo = Nsparse_h*2*s_i
    len_dr_csr_ridx = N*2*s_i
    len_dr_csr_cidx = Nsparse_dr*s_i
    len_ham_csr_ridx = N*2*s_i
    len_ham_csr_cidx = Nsparse_h*s_i
else:
    len_olp_keyinfo = 0
    len_ham_keyinfo = 0
    len_dr_csr_ridx = 0
    len_dr_csr_cidx = 0
    len_ham_csr_ridx = 0
    len_ham_csr_cidx = 0
win1 = MPI.Win.Allocate_shared(len_olp_keyinfo,s_i,comm=shm_comm)
buf1,s_i = win1.Shared_query(0)
olp_keyinfo = np.ndarray(buffer=buf1,dtype=np.int32,shape=(2,Nsparse_o))
win3 = MPI.Win.Allocate_shared(len_ham_keyinfo,s_i,comm=shm_comm)
buf3,s_i = win3.Shared_query(0)
ham_keyinfo = np.ndarray(buffer=buf3,dtype=np.int32,shape=(2,Nsparse_h))
win5 = MPI.Win.Allocate_shared(len_dr_csr_ridx,s_i,comm=shm_comm)
buf5,s_i = win5.Shared_query(0)
dr_csr_ridx = np.ndarray(buffer=buf5,dtype=np.int32,shape=(2,N))
win6 = MPI.Win.Allocate_shared(len_dr_csr_cidx,s_i,comm=shm_comm)
buf6,s_i = win6.Shared_query(0)
dr_csr_cidx = np.ndarray(buffer=buf6,dtype=np.int32,shape=(Nsparse_dr))
win8 = MPI.Win.Allocate_shared(len_ham_csr_ridx,s_i,comm=shm_comm)
buf8,s_i = win8.Shared_query(0)
ham_csr_ridx = np.ndarray(buffer=buf8,dtype=np.int32,shape=(2,N))
win9 = MPI.Win.Allocate_shared(len_ham_csr_cidx,s_i,comm=shm_comm)
buf9,s_i = win9.Shared_query(0)
ham_csr_cidx = np.ndarray(buffer=buf9,dtype=np.int32,shape=(Nsparse_h))

shm_comm.Barrier()
if shm_id == 0:
    ham_keyinfo[0] = (keyinfo_h[:,0]//N).astype(np.int32)
    ham_keyinfo[1] = (keyinfo_h[:,0]%N).astype(np.int32)
    olp_keyinfo[0] = (keyinfo_o[:,0]//N).astype(np.int32)
    olp_keyinfo[1] = (keyinfo_o[:,0]%N).astype(np.int32)
    readdrSH.coo2csridx(N,Nsparse_dr,keyinfo_dr,dr_csr_ridx,dr_csr_cidx)
    readdrSH.coo2csridx(N,Nsparse_h,keyinfo_h,ham_csr_ridx,ham_csr_cidx)
shm_comm.Barrier()
# free sparse key_info
MPI.Win.Free(win06)
MPI.Win.Free(win07)
MPI.Win.Free(win08)
# create drp buffer <- dr[:,icell*Np:(icell+1)*Np]
rowmin = icell*Np
rowmax = (icell+1)*Np
drp_min = dr_csr_ridx[0,rowmin]
drp_max = dr_csr_ridx[0,rowmax]
Nsparse_drp = drp_max-drp_min
if shm_id == 0:
    len_drp_csr_ridx = Np*2*s_i
else:
    len_drp_csr_ridx = 0
win7 = MPI.Win.Allocate_shared(len_drp_csr_ridx,s_i,comm=shm_comm)
buf7,s_i = win7.Shared_query(0)
drp_csr_ridx = np.ndarray(buffer=buf7,dtype=np.int32,shape=(2,Np))
if (shm_id==0):
    drp_csr_ridx[:] = dr_csr_ridx[:,rowmin:rowmax]-drp_min
shm_comm.Barrier()
MPI.Win.Free(win5)
endtime = MPI.Wtime()
if myid == 0:
    print("Reading sparse data time: %.5fs."%(endtime-starttime),flush=True)

Npproc_num = np.zeros((nprocs),dtype=np.int32)
Npproc = np.zeros((nprocs+1),dtype=np.int32)
for i in range(nprocs):
    Np_min = (Np*i)//nprocs
    Np_max = (Np*(i+1))//nprocs
    Npproc_num[i] = Np_max-Np_min
    Npproc[i+1] = Np_max
Np_num = Npproc_num[myid]
# calculate drSH[Np_num,Np]
if Ispin != 2:
    drSH = np.empty((Ispin+1,3,ncell,Np_num*Np),dtype=float)
    for i in range(Ispin+1):
        readdrSH.olp_inv(
            comm,nprocs,myid,olp,olp_keyinfo,Nsparse_o,
            ham[i],ham_keyinfo,Nsparse_h,dr,drp_csr_ridx,dr_csr_cidx,
            Nsparse_drp,drp_min,N,ncell,Mb,Nb,drSH[i]
        )
else:
    drSH = np.empty((4,3,ncell,Np_num*Np),dtype=complex)
    readdrSH_nc.olp_inv(
        comm,nprocs,myid,olp,olp_keyinfo,Nsparse_o,
        ham,ham_csr_ridx,ham_csr_cidx,Nsparse_h,
        dr,drp_csr_ridx,dr_csr_cidx,Nsparse_drp,drp_min,
        N,ncell,Mb,Nb,drSH
    )

# free sparse data
MPI.Win.Free(win)
MPI.Win.Free(win1)
MPI.Win.Free(win2)
MPI.Win.Free(win3)
MPI.Win.Free(win4)
MPI.Win.Free(win6)
MPI.Win.Free(win7)
MPI.Win.Free(win8)
MPI.Win.Free(win9)

# split nmodes
natom_block = 1 if nm_block<3 else nm_block//3
natom_loop = atomnum_u//natom_block
natom_buffer = natom_block
if atomnum_u%natom_block != 0:
    natom_loop += 1
    natom_buffer += 1
natom_split = np.zeros((natom_loop+1),dtype=np.int32)
for i in range(natom_loop):
    natom_min = (atomnum_u*i)//natom_loop
    natom_max = (atomnum_u*(i+1))//natom_loop
    natom_split[i+1] = natom_max


# start epc
starttime = MPI.Wtime()
mass = get_mass(inDir+infile)
# create epcDir
if myid == 0:
    if not os.path.exists(inDir+epcDir):
        os.mkdir(inDir+epcDir)

if IsAllKlist:
    # read kpath, split knum
    kproc_num = np.zeros((nprocs),dtype=np.int32)
    kproc = np.zeros((nprocs+1),dtype=np.int32)
    if EpcType == 'A':
        nq_str = conf['epc']['nq']
        nq_list = nq_str[1:-1].split(',')
        nq = np.array([int(i) for i in nq_list],dtype=np.int32)
        knum = int(nq[0]*nq[1]*nq[2])
        nkpath = knum; kpath = np.empty((0,0))
    else:
        nq_str = conf['epc']['nq']
        nq_list = nq_str[1:-1].split(',')
        nq = np.array([int(i) for i in nq_list],dtype=np.int32)
        knum = int(nq[0]*nq[1]*nq[2])
        Kpoint_str = conf['epc']['Kpoint']
        nkpath, kpath = GenKlist(Kpoint_str,nq)
        if myid == 0:
            if EpcType == 'K':
                np.save(inDir+epcDir+'kpath.npy',kpath)
            else:
                np.save(inDir+epcDir+'qpath.npy',kpath)
    kfactor = nkpath

    for i in range(nprocs):
        knum_min = (knum*i)//nprocs
        knum_max = (knum*(i+1))//nprocs
        kproc_num[i] = knum_max - knum_min
        kproc[i+1] = knum_max

    # create shm buffer
    s_dcplx = 16
    if (shm_id==0):
        len_bandvec = int(knum)*norb*nbands*s_dcplx
    else:
        len_bandvec = 0
    win2 = MPI.Win.Allocate_shared(len_bandvec,s_dcplx,comm=shm_comm)
    buf2,s_dcplx = win2.Shared_query(0)
    if Ispin != 2:
        bandveck = np.ndarray(
            buffer=buf2,dtype=complex,shape=(Ispin+1,knum,Np,nbands)
        )
    else:
        bandveck = np.ndarray(
            buffer=buf2,dtype=complex,shape=(knum,norb,nbands)
        )
    nmnb2 = nmodes*nbands*nbands
    phvecval_p = np.empty((kproc_num[myid],nmodes*nmodes),dtype=complex)
    epc_t = np.empty((kfactor*kproc_num[myid],nmnb2),dtype=complex)

    # read bandvec & phvecval
    phvalname = conf['epc']['phvalname']
    phval = np.load(inDir+phononDir+phvalname)
    phvecname = conf['epc']['phvecname']
    if (shm_id==0):
        vecname = ['','']
        if Ispin != 1:
            vecname[0] += conf['epc']['vecname']
        else:
            vecname_t = conf['epc']['vecname']
            vecname[0] += vecname_t.split('.')[0]+'_up.npy'
            vecname[1] += vecname_t.split('.')[0]+'_dn.npy'
        if Ispin != 2:
            for i in range(Ispin+1):
                if IsAllVec:
                    bandveck[i] = np.load(inDir+bandDir+vecname[i])[:,:,bmin:bmax+1]
                else:
                    bandveck[i] = np.load(inDir+bandDir+vecname[i])
        else:
            if IsAllVec:
                bandveck[:] = np.load(inDir+bandDir+vecname[0])[:,:,bmin:bmax+1]
            else:
                bandveck[:] = np.load(inDir+bandDir+vecname[0])
    phval = np.where(phval>0,phval,1e-10)
    phval = 1.0/np.sqrt(phval)
    epcfunc_nl.phvec_read(
        comm,myid,nmodes,kproc,kproc_num,
        factor1,mass,phval,phvecval_p,
        (inDir+phononDir+phvecname).encode('utf-8')
    )
    if Ispin != 2:
        for i in range(Ispin+1):
            # read file output from epcsparse.py
#            epc_t[:] = 0.0
            epcfunc_nl.FileRead(
                filename[i].encode('utf-8'),EpcType.encode('utf-8'),comm,
                myid,nprocs,4,kfactor,knum,nmnb2,kproc,kproc_num,epc_t
            )
            comm.Barrier()
            # calculate epc left & right
            if EpcType != 'Q':
                epcfunc_nl.MPIepcNL_L(
                    comm,nmodes,natom_loop,natom_buffer,Np,nbands,
                    ncell,knum,natom_split,nq,R_list,nkpath,kpath,
                    drSH[i],bandveck[i],phvecval_p,kproc,kproc_num,
                    Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
                )
                epcfunc_nl.MPIepcNL_R(
                    comm,nmodes,natom_loop,natom_buffer,Np,nbands,
                    ncell,knum,natom_split,nq,R_list,nkpath,kpath,
                    drSH[i],bandveck[i],phvecval_p,kproc,kproc_num,
                    Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
                )
                epcfunc_nl.MPIepc_write_p(
                    comm,kfactor,nmnb2,kproc,kproc_num,
                    epc_t,filename[i].encode('utf-8')
                )
            else:
                epcfunc_nl.MPIepcNL_L_q(
                    comm,nmodes,natom_loop,natom_buffer,Np,nbands,
                    ncell,knum,natom_split,nq,R_list,nkpath,kpath,
                    drSH[i],bandveck[i],phvecval_p,kproc,kproc_num,
                    Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
                )
                epcfunc_nl.MPIepcNL_R_q(
                    comm,nmodes,natom_loop,natom_buffer,Np,nbands,
                    ncell,knum,natom_split,nq,R_list,nkpath,kpath,
                    drSH[i],bandveck[i],phvecval_p,kproc,kproc_num,
                    Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
                )
                epcfunc_nl.MPIepc_write(
                    comm,nkpath*nmnb2,kproc,kproc_num,
                    epc_t,filename[i].encode('utf-8')
                )
    else:
        # read file output from epcsparse.py
#        epc_t[:] = 0.0
        epcfunc_nl.FileRead(
            filename[0].encode('utf-8'),EpcType.encode('utf-8'),comm,
            myid,nprocs,4,kfactor,knum,nmnb2,kproc,kproc_num,epc_t
        )
        comm.Barrier()
        # calculate epc left & right
        if EpcType != 'Q':
            epcfunc_nl_nc.MPIepcNL_L(
                comm,nmodes,natom_loop,natom_buffer,Np,nbands,
                ncell,knum,2,natom_split,nq,R_list,nkpath,kpath,
                drSH,bandveck,phvecval_p,kproc,kproc_num,
                Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
            )
            epcfunc_nl_nc.MPIepcNL_R(
                comm,nmodes,natom_loop,natom_buffer,Np,nbands,
                ncell,knum,2,natom_split,nq,R_list,nkpath,kpath,
                drSH,bandveck,phvecval_p,kproc,kproc_num,
                Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
            )
            epcfunc_nl.MPIepc_write_p(
                comm,kfactor,nmnb2,kproc,kproc_num,
                epc_t,filename[0].encode('utf-8')
            )
        else:
            epcfunc_nl_nc.MPIepcNL_L_q(
                comm,nmodes,natom_loop,natom_buffer,Np,nbands,
                ncell,knum,2,natom_split,nq,R_list,nkpath,kpath,
                drSH,bandveck,phvecval_p,kproc,kproc_num,
                Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
            )
            epcfunc_nl_nc.MPIepcNL_R_q(
                comm,nmodes,natom_loop,natom_buffer,Np,nbands,
                ncell,knum,2,natom_split,nq,R_list,nkpath,kpath,
                drSH,bandveck,phvecval_p,kproc,kproc_num,
                Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
            )
            epcfunc_nl.MPIepc_write(
                comm,nkpath*nmnb2,kproc,kproc_num,
                epc_t,filename[0].encode('utf-8')
            )
else:
    EpcType = 'A'
    # split knum
    qproc_num = np.zeros((nprocs),dtype=np.int32)
    qproc = np.zeros((nprocs+1),dtype=np.int32)
    kproc_num = np.zeros((nprocs),dtype=np.int32)
    kproc = np.zeros((nprocs+1),dtype=np.int32)
    nq_str = conf['epc']['nq']
    nq_list = nq_str[1:-1].split(',')
    nq = np.array([int(i) for i in nq_list],dtype=np.int32)
    knum = nq[0]*nq[1]*nq[2]

    if Ispin == 1:
        l_spin = 2
        name_ex = ['_up','_dn']
    else:
        l_spin = 1
        name_ex = ['']
    basselname = inDir+bandDir+conf['epc']['basselname'].split('.')[0]
    bassel = []
    knum_p = np.zeros((l_spin),dtype=np.int32)
    kproc_num = np.zeros((l_spin,nprocs),dtype=np.int32)
    kproc = np.zeros((l_spin,nprocs+1),dtype=np.int32)
    for spin in range(l_spin):
        bassel_t = np.load(basselname+'%s.npy'%name_ex[spin])
        bassel.append(bassel_t)
        knum_p[spin] = bassel_t.shape[0]
        for i in range(nprocs):
            knum_min = (knum_p[spin]*i)//nprocs
            knum_max = (knum_p[spin]*(i+1))//nprocs
            kproc_num[spin,i] = knum_max - knum_min
            kproc[spin,i+1] = knum_max
    knum_buf = int(max(knum_p))
    kproc_num_buf = np.max(kproc_num,axis=0)

    for i in range(nprocs):
        qnum_min = (knum*i)//nprocs
        qnum_max = (knum*(i+1))//nprocs
        qproc_num[i] = qnum_max - qnum_min
        qproc[i+1] = qnum_max

    # create shm buffer
    s_dcplx = 16
    if (shm_id==0):
        if Ispin != 2:
            len_bandvec = (Ispin+1)*knum_buf*norb*s_dcplx
        else:
            len_bandvec = knum_buf*norb*s_dcplx
    else:
        len_bandvec = 0
    win2 = MPI.Win.Allocate_shared(len_bandvec,s_dcplx,comm=shm_comm)
    buf2,s_dcplx = win2.Shared_query(0)
    if Ispin != 2:
        bandveck = np.ndarray(
            buffer=buf2,dtype=complex,shape=(Ispin+1,knum_buf,Np)
        )
    else:
        bandveck = np.ndarray(
            buffer=buf2,dtype=complex,shape=(knum_buf,norb)
        )
    phvecval_p = np.empty((qproc_num[myid],nmodes*nmodes),dtype=complex)
    epc_t = np.empty((knum_buf*kproc_num_buf[myid],nmodes),dtype=complex)

    # read bandvec & phvecval
    phvalname = conf['epc']['phvalname']
    phval = np.load(inDir+phononDir+phvalname)
    phvecname = conf['epc']['phvecname']
    if (shm_id==0):
        vecname = conf['epc']['vecname'].split('.')[0]
        vecpname = ['','']
        if Ispin != 1:
            vecpname[0] += vecname+'_p.npy'
        else:
            vecpname[0] += vecname+'_up_p.npy'
            vecpname[1] += vecname+'_dn_p.npy'
        if Ispin != 2:
            for i in range(Ispin+1):
                bandveck[i,0:knum_p[i]] = np.load(inDir+bandDir+vecpname[i])
        else:
            bandveck[:] = np.load(inDir+bandDir+vecpname[0])
    phval = np.where(phval>0,phval,1e-10)
    phval = 1.0/np.sqrt(phval)
    epcfunc_nl.phvec_read(
        comm,myid,nmodes,qproc,qproc_num,
        factor1,mass,phval,phvecval_p,
        (inDir+phononDir+phvecname).encode('utf-8')
    )
    if Ispin != 2:
        for i in range(Ispin+1):
            # read file output from epcsparse.py
#            epc_t[:] = 0.0
            epcfunc_nl.FileRead(
                filename[i].encode('utf-8'),EpcType.encode('utf-8'),
                comm,myid,nprocs,4,knum_p[i],knum_p[i],
                nmodes,kproc[i],kproc_num[i],epc_t
            )
            comm.Barrier()
            # calculate epc left & right
            epcfunc_nl.MPIepcNL_p_L(
                comm,nmodes,natom_loop,natom_buffer,Np,ncell,
                knum_p[i],knum,natom_split,nq,R_list,drSH[i],bandveck[i],
                phvecval_p,kproc[i],kproc_num[i],qproc,qproc_num,bassel[i],
                Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
            )
            epcfunc_nl.MPIepcNL_p_R(
                comm,nmodes,natom_loop,natom_buffer,Np,ncell,
                knum_p[i],knum,natom_split,nq,R_list,drSH[i],bandveck[i],
                phvecval_p,kproc[i],kproc_num[i],qproc,qproc_num,bassel[i],
                Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
            )
            epcfunc_nl.MPIepc_write_p(
                comm,knum_p[i],nmodes,kproc[i],kproc_num[i],
                epc_t,filename[i].encode('utf-8')
            )
    else:
        # read file output from epcsparse.py
#        epc_t[:] = 0.0
        epcfunc_nl.FileRead(
            filename[0].encode('utf-8'),EpcType.encode('utf-8'),
            comm,myid,nprocs,4,knum_p[0],knum_p[0],
            nmodes,kproc[0],kproc_num[0],epc_t
        )
        comm.Barrier()
        # calculate epc left & right
        epcfunc_nl_nc.MPIepcNL_p_L(
            comm,nmodes,natom_loop,natom_buffer,Np,ncell,
            knum_p[0],knum,2,natom_split,nq,R_list,drSH,bandveck,
            phvecval_p,kproc[0],kproc_num[0],qproc,qproc_num,bassel[0],
            Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
        )
        epcfunc_nl_nc.MPIepcNL_p_R(
            comm,nmodes,natom_loop,natom_buffer,Np,ncell,
            knum_p[0],knum,2,natom_split,nq,R_list,drSH,bandveck,
            phvecval_p,kproc[0],kproc_num[0],qproc,qproc_num,bassel[0],
            Npproc,Npproc_num,atom_idx_sum_u,atom_idx_u,epc_t
        )
        epcfunc_nl.MPIepc_write_p(
            comm,knum_p[0],nmodes,kproc[0],kproc_num[0],
            epc_t,filename[0].encode('utf-8')
        )

MPI.Win.Free(win2)
endtime = MPI.Wtime()
if (myid==0):
    print("epc time:%.6fs"%(endtime-starttime),flush=True)
