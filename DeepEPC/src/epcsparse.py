import numpy as np
import json
import os
import sys
from ase.data import atomic_numbers, atomic_masses
import configparser
import readhamilsparse
import readhamilsparse_nc
import epcfuncsparse
import epcfuncsparse_nc
import time
import warnings
from mpi4py import MPI

warnings.filterwarnings("ignore", category=DeprecationWarning)
# load config
conf = configparser.ConfigParser()
conf.read(sys.argv[1] if len(sys.argv)>1 else 'config.ini',encoding='utf-8')

Ispin = int(conf['epc']['Ispin'])
IsH5 = True if conf['epc']['IsH5']=='True' else False
if Ispin == 1 and IsH5 == True:
    print('HDF5 files are now not supported by Ispin=1!')
    sys.exit()
if IsH5: H5HamName = conf['epc']['H5HamName']

dhamil_method = conf['epc']['dhamil_method']
inDir = conf['epc']['inDir']+'/'
bandDir = conf['epc']['bandDir']+'/'
phononDir = conf['epc']['phononDir']+'/'
dhamilDir = conf['epc']['dhamilDir']+'/'
epcDir = conf['epc']['epcDir']+'/'
ucellidx_str = conf['epc']['ucellidx']
ucellidx_list = ucellidx_str[1:-1].split(',')
ucellidx = [int(i) for i in ucellidx_list]
infile = conf['epc']['infile_out']
EpcType = conf['epc']['EpcType']

atom_str = conf['epc']['atom']
atom_list = atom_str[1:-1].split(',')
atom = [int(i) for i in atom_list]
orbital_str = conf['epc']['orbital']
orbital_list = orbital_str[1:-1].split(',')
orbital = [int(i) for i in orbital_list]

dQ = float(conf['epc']['dQ'])
IsAllVec = True if conf['epc']['IsAllVec']=='True' else False
IsAllKlist = True if conf['epc']['IsAllKlist']=='True' else False

dH_block = int(conf['mpi']['DHAMIL_BLOCK'])
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

atomnum = sum(atom)
nmodes = atomnum*3

atom_type = [x*y for x,y in zip(atom,orbital)]
atom_idx0 = [y for x,y in zip(atom,orbital) for i in range(x)]
atom_idx = [sum(atom_idx0[0:i]) for i in range(len(atom_idx0)+1)]
norbital = atom_idx[-1]
atom_idx = np.array(atom_idx,dtype=np.int32)
atom_idx0 = np.array(atom_idx0,dtype=np.int32)

if Ispin == 0: norb = norbital
else: norb = 2*norbital
if IsAllVec:
    bmin = 0
    bmax = norb-1
    nbands = norb
else:
    bmin = int(conf['epc']['bmin'])
    bmax = int(conf['epc']['bmax'])
    nbands = bmax-bmin+1

ncell = ucellidx[0]*ucellidx[1]*ucellidx[2]
icell = (center[0]*ucellidx[1]+center[1])*ucellidx[2]+center[2]
catom = icell*atomnum+np.arange(atomnum,dtype=np.int32)

ucellnum = np.array([ucellidx]*3).T
atom_idx_all0 = [y for j in range(ncell) for x,y in zip(atom,orbital) for i in range(x)]
atom_idx_all = [sum(atom_idx_all0[0:i]) for i in range(len(atom_idx_all0)+1)]
atom_idx_all = np.array(atom_idx_all,dtype=np.int32)
atom_idx_all0 = np.array(atom_idx_all0,dtype=np.int32)

R_list = np.array(
    [
        [i,j,k] \
        for i in range(ucellidx[0]) \
        for j in range(ucellidx[1]) \
        for k in range(ucellidx[2]) \
    ],
    dtype=np.int32
)-center
R_num = ucellidx[0]*ucellidx[1]*ucellidx[2]

epcname = inDir+epcDir+conf['epc']['epcname'].split('.')[0]
filename = ['','']
if Ispin != 1:
    spin_loop = 1
    filename[0] += epcname+'.dat'
else:
    spin_loop = 2
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
           for i in range(atomnum)])


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
nprocs_shm = shm_comm.Get_size()
shm_id = shm_comm.Get_rank()
nnodes = nprocs//nprocs_shm

s_int = 4
s_d = 8
s_dcplx = 16
# split nmodes
nm_loop = nmodes//nm_block
nm_buffer = nm_block
if nmodes%nm_block != 0:
    nm_loop += 1
    nm_buffer += 1
nmodes_split = np.zeros((nm_loop+1),dtype=np.int32)
for i in range(nm_loop):
    nmodes_min = (nmodes*i)//nm_loop
    nmodes_max = (nmodes*(i+1))//nm_loop
    nmodes_split[i+1] = nmodes_max

# get sparse matrix info
ncell2 = ncell*ncell
if (shm_id==0):
    len_keynum = (ncell2+1)*4*s_int
else:
    len_keynum = 0
win00 = MPI.Win.Allocate_shared(len_keynum,s_int,comm=shm_comm)
buf00,s_int = win00.Shared_query(0)
key_num = np.ndarray(
    buffer=buf00,dtype=np.int32,shape=(ncell2+1,4)
)
shm_comm.Barrier()
if (shm_id==0):
    readhamilsparse.GetSparseNum(
        inDir.encode('utf-8'),
        (H5HamName if IsH5 else 'None').encode('utf-8'),
        key_num,atom_idx_all0,atom_idx_all,atomnum*ncell,
        norbital,ncell,ncell2,IsH5
    )
shm_comm.Barrier()
if (shm_id==0):
    len_pubkey = key_num[ncell2,2]*6*s_int
    len_key = key_num[ncell2,3]*2*s_int
    len_key1 = key_num[ncell2,3]*2*s_int
else:
    len_pubkey = 0
    len_key = 0
    len_key1 = 0
win01 = MPI.Win.Allocate_shared(len_pubkey,s_int,comm=shm_comm)
buf01,s_int = win01.Shared_query(0)
pub_key = np.ndarray(
    buffer=buf01,dtype=np.int32,shape=(key_num[ncell2,2],6)
)
win02 = MPI.Win.Allocate_shared(len_key,s_int,comm=shm_comm)
buf02,s_int = win02.Shared_query(0)
key_info = np.ndarray(
    buffer=buf02,dtype=np.int32,shape=(key_num[ncell2,3],2)
)
win03 = MPI.Win.Allocate_shared(len_key1,s_int,comm=shm_comm)
buf03,s_int = win03.Shared_query(0)
key_info1 = np.ndarray(
    buffer=buf03,dtype=np.int32,shape=(key_num[ncell2,3],2)
)
shm_comm.Barrier()
if (shm_id==0):
    readhamilsparse.GetSparseIdx(
        inDir.encode('utf-8'),
        (H5HamName if IsH5 else 'None').encode('utf-8'),norbital,
        ncell,ncell2,key_num,pub_key,key_info,key_info1,
        atom_idx_all0,atom_idx_all,atomnum*ncell,IsH5
    )

# get sum sparse matrix info
if (shm_id==0):
    len_keynum_s = (ncell+1)*2*s_int
else:
    len_keynum_s = 0
win04 = MPI.Win.Allocate_shared(len_keynum_s,s_int,comm=shm_comm)
buf04,s_int = win04.Shared_query(0)
key_num_s = np.ndarray(
    buffer=buf04,dtype=np.int32,shape=(ncell+1,2)
)
shm_comm.Barrier()
if (shm_id==0):
    readhamilsparse.GetSparseNumSum(
        ncell,key_num,key_num_s,key_info
    )
shm_comm.Barrier()
if (shm_id==0):
    len_key = key_num_s[ncell,1]*s_int
else:
    len_key = 0
win05 = MPI.Win.Allocate_shared(len_key,s_int,comm=shm_comm)
buf05,s_int = win05.Shared_query(0)
key_info_s = np.ndarray(
    buffer=buf05,dtype=np.int32,shape=(key_num_s[ncell,1])
)
shm_comm.Barrier()
if (shm_id==0):
    readhamilsparse.GetSparseIdxSum(
        ncell,key_num,key_num_s,key_info,key_info_s
    )
shm_comm.Barrier()
# read H0
if Ispin == 2:
    if dhamil_method[0] != 'C':
        if (shm_id==0):
            len_hamil = 4*key_num[ncell2,3]*s_dcplx
        else:
            len_hamil = 0
        win06 = MPI.Win.Allocate_shared(len_hamil,s_dcplx,comm=shm_comm)
        buf06,s_dcplx = win06.Shared_query(0)
        hamil_buf = np.ndarray(
            buffer=buf06,dtype=np.complex128,shape=(4*key_num[ncell2,3])
        )
        readhamilsparse_nc.ReadHamil0(
            shm_comm,norbital,ncell,max(orbital),atomnum*ncell,
            key_num,pub_key,atom_idx_all0,atom_idx_all,
            hamil_buf,inDir.encode('utf-8'),
            (H5HamName if IsH5 else 'None').encode('utf-8'),IsH5
        )
    else:
        hamil_buf = np.zeros((0),dtype=np.complex128)
else:
    if dhamil_method[0] != 'C':
        if (shm_id==0):
            len_hamil = (Ispin+1)*key_num[ncell2,3]*s_d
        else:
            len_hamil = 0
        win06 = MPI.Win.Allocate_shared(len_hamil,s_d,comm=shm_comm)
        buf06,s_d = win06.Shared_query(0)
        hamil_buf = np.ndarray(
            buffer=buf06,dtype=np.float64,shape=((Ispin+1)*key_num[ncell2,3])
        )
        readhamilsparse.ReadHamil0(
            shm_comm,norbital,ncell,max(orbital),atomnum*ncell,
            key_num,pub_key,atom_idx_all0,atom_idx_all,
            hamil_buf,inDir.encode('utf-8'),
            (H5HamName if IsH5 else 'None').encode('utf-8'),Ispin,IsH5
        )
    else:
        hamil_buf = np.zeros((0),dtype=np.float64)
shm_comm.Barrier()

#if (myid == 0):
#    np.save('key_num.npy',key_num)
#    np.save('key_num_s.npy',key_num_s)
#    np.save('key_info.npy',key_info)
#    np.save('key_info_s.npy',key_info_s)

# create epcDir
if myid == 0:
    if not os.path.exists(inDir+epcDir):
        os.mkdir(inDir+epcDir)

if IsAllKlist:
    # read kgrid, split knum
    kproc_num = np.zeros((nprocs),dtype=np.int32)
    kproc = np.zeros((nprocs+1),dtype=np.int32)
    if EpcType != 'Q':
        nq_str = conf['epc']['nq']
        nq_list = nq_str[1:-1].split(',')
        nq = np.array([int(i) for i in nq_list],dtype=np.int32)
        knum = nq[0]*nq[1]*nq[2]
        if EpcType == 'K':
            Kpoint_str = conf['epc']['Kpoint']
            nkpath, kpath = GenKlist(Kpoint_str,nq)
            if myid == 0:
                np.save(inDir+epcDir+'kpath.npy',kpath)
        else:
            nkpath = knum; kpath = np.empty((0,0))
        knum2 = nkpath*knum
        for i in range(nprocs):
            knum_min = (knum2*i)//nprocs
            knum_max = (knum2*(i+1))//nprocs
            kproc_num[i] = knum_max - knum_min
            kproc[i+1] = knum_max
    else:
        nq_str = conf['epc']['nq']
        nq_list = nq_str[1:-1].split(',')
        nq = np.array([int(i) for i in nq_list],dtype=np.int32)
        knum = nq[0]*nq[1]*nq[2]
        Kpoint_str = conf['epc']['Kpoint']
        nkpath, kpath = GenKlist(Kpoint_str,nq)
        if myid == 0:
            np.save(inDir+epcDir+'qpath.npy',kpath)
        for i in range(nprocs):
            knum_min = (knum*i)//nprocs
            knum_max = (knum*(i+1))//nprocs
            kproc_num[i] = knum_max - knum_min
            kproc[i+1] = knum_max
    
    # create shm buffer
    if (shm_id==0):
        #print(nm_buffer,key_num[ncell2,3])
        if Ispin != 2:
            len_epcr = nm_buffer*(Ispin+1)*int(key_num[ncell2,3])*s_d
            len_bandvec = (Ispin+1)*int(knum)*norbital*nbands*s_dcplx
        else:
            len_epcr = nm_buffer*4*int(key_num[ncell2,3])*s_dcplx
            len_bandvec = int(knum)*norb*nbands*s_dcplx
        len_phvecval = int(knum)*nm_buffer*nmodes*s_dcplx
    else:
        len_epcr = 0
        len_phvecval = 0
        len_bandvec = 0
    
    win = MPI.Win.Allocate_shared(len_epcr,s_d,comm=shm_comm)
    buf,s_d = win.Shared_query(0)
    win2 = MPI.Win.Allocate_shared(len_bandvec,s_dcplx,comm=shm_comm)
    buf2,s_dcplx = win2.Shared_query(0)
    if Ispin != 2:
        dhamil = np.ndarray(
            buffer=buf,dtype=float,
            shape=(nm_buffer,Ispin+1,key_num[ncell2,3])
        )
        bandveck = np.ndarray(
            buffer=buf2,dtype=complex,
            shape=(Ispin+1,knum,norbital,nbands)
        )
    else:
        dhamil = np.ndarray(
            buffer=buf,dtype=complex,
            shape=(nm_buffer,4,key_num[ncell2,3])
        )
        bandveck = np.ndarray(
            buffer=buf2,dtype=complex,
            shape=(knum,norb,nbands)
        )
    win1 = MPI.Win.Allocate_shared(len_phvecval,s_dcplx,comm=shm_comm)
    buf1,s_dcplx = win1.Shared_query(0)
    phvecval = np.ndarray(
        buffer=buf1,dtype=complex,
        shape=(knum,nm_buffer,nmodes)
    )
    nmnb2 = nmodes*nbands*nbands
    if EpcType != 'Q':
        epc_t = np.empty((kproc_num[myid],nmnb2),dtype=complex)
    else:
        epc_t = np.empty((kproc_num[myid]*nkpath,nmnb2),dtype=complex)
   
    mass = get_mass(inDir+infile)
    # read bandvec & phval
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

    if EpcType == 'Q':
        kfactor = nkpath*nmnb2
    else:
        kfactor = nmnb2

    dhamiltime = 0.0
    epctime = 0.0
    for spin in range(spin_loop):
        epc_t[:] = 0.0
        for i in range(nm_loop):
            nm_min = nmodes_split[i]
            nm_max = nmodes_split[i+1]
            nm_num = nm_max - nm_min
            if (shm_id==0): dhamil[:] = 0.0
            # preprocess phvecval
            start = time.time()
            epcfuncsparse.epc_preprocess(
                shm_comm,myid,shm_id,nprocs_shm,knum,nmodes,nm_num,
                nm_min,atomnum,factor1,mass,phval,phvecval,
                (inDir+phononDir+phvecname).encode('utf-8')
            )
            # deltahamil
            if Ispin != 2:
                readhamilsparse.deltahamil_b(
                    comm,shm_comm,nm_num,nm_min,dH_block,norbital,ncell,
                    max(orbital),atomnum*ncell,atom_idx_all0,atom_idx_all,
                    catom,key_num,pub_key,key_info1,1.0/dQ,hamil_buf,dhamil,
                    inDir.encode('utf-8'),dhamilDir.encode('utf-8'),
                    (H5HamName if IsH5 else 'None').encode('utf-8'),
                    dhamil_method.encode('utf-8'),Ispin,IsH5
                )
            else:
                readhamilsparse_nc.deltahamil_b(
                    comm,shm_comm,nm_num,nm_min,dH_block,norbital,ncell,
                    max(orbital),atomnum*ncell,atom_idx_all0,atom_idx_all,
                    catom,key_num,pub_key,key_info1,1.0/dQ,hamil_buf,dhamil,
                    inDir.encode('utf-8'),dhamilDir.encode('utf-8'),
                    (H5HamName if IsH5 else 'None').encode('utf-8'),
                    dhamil_method.encode('utf-8'),IsH5
                )
            end = time.time()
            dhamiltime += end - start
            # epc calculation
            start = time.time()
            if Ispin != 2:
                if EpcType != 'Q':
                    epcfuncsparse.MPIepc(
                        comm,spin,nmodes,nm_num,nm_min,norbital,nbands,ncell,
                        knum,nq,R_list,nkpath,kpath,key_num,key_num_s,key_info,
                        key_info_s,dhamil,bandveck[spin],phvecval,kproc,kproc_num,epc_t
                    )
                else:
                    epcfuncsparse.MPIepc_q(
                        comm,spin,nmodes,nm_num,nm_min,norbital,nbands,ncell,
                        knum,nq,R_list,nkpath,kpath,key_num,key_num_s,key_info,
                        key_info_s,dhamil,bandveck[spin],phvecval,kproc,kproc_num,epc_t
                    )
            else:
                if EpcType != 'Q':
                    epcfuncsparse_nc.MPIepc(
                        comm,2,nmodes,nm_num,nm_min,norbital,nbands,ncell,
                        knum,nq,R_list,nkpath,kpath,key_num,key_num_s,key_info,
                        key_info_s,dhamil,bandveck,phvecval,kproc,kproc_num,epc_t
                    )
                else:
                    epcfuncsparse_nc.MPIepc_q(
                        comm,2,nmodes,nm_num,nm_min,norbital,nbands,ncell,
                        knum,nq,R_list,nkpath,kpath,key_num,key_num_s,key_info,
                        key_info_s,dhamil,bandveck,phvecval,kproc,kproc_num,epc_t
                    )
            end = time.time()
            epctime += end - start
        # output epc
        epcfuncsparse.MPIepc_write(
            comm,kfactor,kproc,kproc_num,epc_t,
            filename[spin].encode('utf-8')
        )
    
    if (myid==0):
        print("dhamil time:%.6fs"%(dhamiltime),flush=True)
        print("epc time:%.6fs"%(epctime),flush=True)
   
    MPI.Win.Free(win)
    MPI.Win.Free(win1)
    MPI.Win.Free(win2)
else:
    # read kgrid, split knum
    nq_str = conf['epc']['nq']
    nq_list = nq_str[1:-1].split(',')
    nq = np.array([int(i) for i in nq_list],dtype=np.int32)
    knum = nq[0]*nq[1]*nq[2]

    if Ispin == 1: name_ex = ['_up','_dn']
    else: name_ex = ['']
    basselname = inDir+bandDir+conf['epc']['basselname'].split('.')[0]
    bassel = []
    knum_p = np.zeros((spin_loop),dtype=np.int32)
    kproc_num = np.zeros((spin_loop,nprocs),dtype=np.int32)
    kproc = np.zeros((spin_loop,nprocs+1),dtype=np.int32)
    for spin in range(spin_loop):
        bassel_t = np.load(basselname+'%s.npy'%name_ex[spin])
        bassel.append(bassel_t)
        knum_p[spin] = bassel_t.shape[0]
        knum_p2 = knum_p[spin]*knum_p[spin]
        for i in range(nprocs):
            knum_min = (knum_p2*i)//nprocs
            knum_max = (knum_p2*(i+1))//nprocs
            kproc_num[spin,i] = knum_max - knum_min
            kproc[spin,i+1] = knum_max
    knum_buf = int(max(knum_p))
    kproc_num_buf = np.max(kproc_num,axis=0)

    # create shm buffer
    s_d = 8
    s_dcplx = 16
    if (shm_id==0):
        #print(nm_buffer,key_num[ncell2,3])
        if Ispin != 2:
            len_epcr = nm_buffer*(Ispin+1)*int(key_num[ncell2,3])*s_d
            len_bandvec = (Ispin+1)*knum_buf*norbital*s_dcplx
        else:
            len_epcr = nm_buffer*4*int(key_num[ncell2,3])*s_dcplx
            len_bandvec = knum_buf*norb*s_dcplx
        len_phvecval = int(knum)*nm_buffer*nmodes*s_dcplx
    else:
        len_epcr = 0
        len_phvecval = 0
        len_bandvec = 0

    win = MPI.Win.Allocate_shared(len_epcr,s_d,comm=shm_comm)
    buf,s_d = win.Shared_query(0)
    win2 = MPI.Win.Allocate_shared(len_bandvec,s_dcplx,comm=shm_comm)
    buf2,s_dcplx = win2.Shared_query(0)
    if Ispin != 2:
        dhamil = np.ndarray(
            buffer=buf,dtype=float,
            shape=(nm_buffer,Ispin+1,key_num[ncell2,3])
        )
        bandveck = np.ndarray(
            buffer=buf2,dtype=complex,
            shape=(Ispin+1,knum_buf,norbital)
        )
    else:
        dhamil = np.ndarray(
            buffer=buf,dtype=complex,
            shape=(nm_buffer,4,key_num[ncell2,3])
        )
        bandveck = np.ndarray(
            buffer=buf2,dtype=complex,
            shape=(knum_buf,norb)
        )
    win1 = MPI.Win.Allocate_shared(len_phvecval,s_dcplx,comm=shm_comm)
    buf1,s_dcplx = win1.Shared_query(0)
    phvecval = np.ndarray(
        buffer=buf1,dtype=complex,
        shape=(knum,nm_buffer,nmodes)
    )
    epc_t = np.empty((kproc_num_buf[myid],nmodes),dtype=complex)

    mass = get_mass(inDir+infile)
    # read bandvec & phval
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

    dhamiltime = 0.0
    epctime = 0.0
    for spin in range(spin_loop):
        epc_t[:] = 0.0
        for i in range(nm_loop):
            nm_min = nmodes_split[i]
            nm_max = nmodes_split[i+1]
            nm_num = nm_max - nm_min
            if (shm_id==0): dhamil[:] = 0.0
            # preprocess phvecval
            start = time.time()
            epcfuncsparse.epc_preprocess(
                shm_comm,myid,shm_id,nprocs_shm,knum,nmodes,nm_num,
                nm_min,atomnum,factor1,mass,phval,phvecval,
                (inDir+phononDir+phvecname).encode('utf-8')
            )
            # deltahamil
            if Ispin != 2:
                readhamilsparse.deltahamil_b(
                    comm,shm_comm,nm_num,nm_min,dH_block,norbital,ncell,
                    max(orbital),atomnum*ncell,atom_idx_all0,atom_idx_all,
                    catom,key_num,pub_key,key_info1,1.0/dQ,hamil_buf,dhamil,
                    inDir.encode('utf-8'),dhamilDir.encode('utf-8'),
                    (H5HamName if IsH5 else 'None').encode('utf-8'),
                    dhamil_method.encode('utf-8'),Ispin,IsH5
                )
            else:
                readhamilsparse_nc.deltahamil_b(
                    comm,shm_comm,nm_num,nm_min,dH_block,norbital,ncell,
                    max(orbital),atomnum*ncell,atom_idx_all0,atom_idx_all,
                    catom,key_num,pub_key,key_info1,1.0/dQ,hamil_buf,dhamil,
                    inDir.encode('utf-8'),dhamilDir.encode('utf-8'),
                    (H5HamName if IsH5 else 'None').encode('utf-8'),
                    dhamil_method.encode('utf-8'),IsH5
                )
            end = time.time()
            dhamiltime += end - start
            # epc calculation
            start = time.time()
            if Ispin != 2:
                epcfuncsparse.MPIepc_p(
                    comm,spin,nmodes,nm_num,nm_min,norbital,ncell,
                    knum_p[spin],nq,R_list,key_num,key_num_s,key_info,
                    key_info_s,dhamil,bandveck[spin],phvecval,
                    kproc[spin],kproc_num[spin],bassel[spin],epc_t
                )
            else:
                epcfuncsparse_nc.MPIepc_p(
                    comm,2,nmodes,nm_num,nm_min,norbital,ncell,
                    knum_p[0],nq,R_list,key_num,key_num_s,
                    key_info,key_info_s,dhamil,bandveck,phvecval,
                    kproc[0],kproc_num[0],bassel[0],epc_t
                )
            end = time.time()
            epctime += end - start
        # output epc
        epcfuncsparse.MPIepc_write(
            comm,nmodes,kproc[spin],kproc_num[spin],
            epc_t,filename[spin].encode('utf-8')
        )

    if (myid==0):
        print("dhamil time:%.6fs"%(dhamiltime))
        print("epc time:%.6fs"%(epctime))

    MPI.Win.Free(win)
    MPI.Win.Free(win1)
    MPI.Win.Free(win2)


MPI.Win.Free(win00)
MPI.Win.Free(win01)
MPI.Win.Free(win02)
MPI.Win.Free(win03)
MPI.Win.Free(win04)
MPI.Win.Free(win05)
if dhamil_method[0] != 'C':
    MPI.Win.Free(win06)
