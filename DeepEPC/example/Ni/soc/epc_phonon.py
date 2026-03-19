import os
import time
import numpy as np
from scipy.linalg import eig, eigh
import ase.units as units
import configparser
from mpi4py import MPI

conf = configparser.ConfigParser()
conf.read('config.ini',encoding='utf-8')

inDir = conf['epc']['inDir']+'/'
phononDir = conf['epc']['phononDir']+'/'
ucellidx_str = conf['epc']['ucellidx']
ucellidx_list = ucellidx_str[1:-1].split(',')
ucellidx = [int(i) for i in ucellidx_list]
poscar_ucell = inDir+conf['epc']['poscar_ucell']
infile = conf['epc']['infile_out']
ifcname = conf['epc']['ifcname']
phvalname = conf['epc']['phvalname']
phvecname = conf['epc']['phvecname']

nq_str = conf['epc']['nq']
nq_list = nq_str[1:-1].split(',')
nq = [int(i) for i in nq_list]

atom_str = conf['epc']['atom']
atom_list = atom_str[1:-1].split(',')
atom = [int(i) for i in atom_list]
atomnum = sum(atom)
nmodes = atomnum*3

center = np.array([i//2 for i in ucellidx],dtype=int)
ucellnum = np.array([ucellidx]*3).T
icell = (center[0]*ucellidx[1]+center[1])*ucellidx[2]+center[2]
catom = icell*atomnum+np.arange(atomnum)

R_list = np.array(
    [
        [i,j,k] \
        for i in range(ucellidx[0]) \
        for j in range(ucellidx[1]) \
        for k in range(ucellidx[2]) \
    ],
    dtype=int
)-center
R_num = ucellidx[0]*ucellidx[1]*ucellidx[2]

factor = units._hbar * 1e10 / np.sqrt(units._e * units._amu)

def eig_k(fc,R_list,kpoint):
    
    phase = np.exp(-1j*2*np.pi*np.dot(R_list,kpoint))[:,np.newaxis,np.newaxis]
    fc_k = np.sum(fc*phase,axis=0)

    #val,vec = eigh(fc_k,eigvals_only=False)
    #val = val+0j
    #val,vec = eig(fc_k)
    val,vec = np.linalg.eigh(fc_k, UPLO='U')
    val = val + 0j
    nag = np.where(val.real<0)
    val_sqrt = np.sqrt(val).real
    val_sqrt[nag] = -np.sqrt(-val[nag].real)
    idx = np.argsort(val_sqrt)

    return val_sqrt[idx]*factor,vec[:,idx]


def phonon(inDir,phononDir,infile,qnum,q_list):
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    nprocs = comm.Get_size()

    start = time.time()
    shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    nprocs_shm = shm_comm.Get_size()
    shm_id = shm_comm.Get_rank()
    nnodes = nprocs//nprocs_shm

    if q_list.shape[0] == 0: IsQlist = False
    else: IsQlist = True

    # create shm buffer
    s_d = 8
    if (shm_id==0):
        len_fc = R_num*nmodes*nmodes*s_d
    else:
        len_fc = 0

    win = MPI.Win.Allocate_shared(len_fc,s_d,comm=shm_comm)
    buf,s_d = win.Shared_query(0)
    fc = np.ndarray(
        buffer=buf,dtype=float,shape=(R_num,nmodes,nmodes)
    )
    # read data
    if shm_id == 0:
        fc[:] = np.load(inDir+phononDir+ifcname)
    
    # alloc val/vec
    qproc_num = np.zeros((nprocs),dtype=np.int32)
    qproc = np.zeros((nprocs+1),dtype=np.int32)
    for i in range(nprocs):
        qnum_min = (qnum*i)//nprocs
        qnum_max = (qnum*(i+1))//nprocs
        qproc_num[i] = qnum_max - qnum_min
        qproc[i+1] = qnum_max
    val_p = np.zeros((qproc_num[myid],nmodes))
    vec_p = np.zeros((qproc_num[myid],nmodes,nmodes),dtype=complex)
    if myid == 0:
        val = np.zeros((qnum,nmodes))
        vec = np.zeros((qnum,nmodes,nmodes),dtype=complex)
    else:
        val = np.zeros((0,0))
        vec = np.zeros((0,0,0),dtype=complex)
  
    comm.Barrier()
    qpoint = np.zeros((3),dtype=float)
    qproc_min = qproc[myid]
    qproc_max = qproc[myid+1]
    for i in range(qproc_min,qproc_max):
        if IsQlist:
            qpoint[:] = q_list[i]
        else:
            qpoint[2] = i%nq[2]; qidx_xy = i//nq[2]
            qpoint[1] = qidx_xy%nq[1]; qpoint[0] = qidx_xy//nq[1]
            qpoint /= nq
        val_p[i-qproc_min],vec_p[i-qproc_min] = eig_k(fc,R_list,qpoint)

    comm.Gatherv(val_p,[val,qproc_num*nmodes,\
                 qproc[0:nprocs]*nmodes,MPI.DOUBLE],root=0)
    comm.Gatherv(vec_p,[vec,qproc_num*nmodes**2,\
                 qproc[0:nprocs]*nmodes**2,MPI.DOUBLE_COMPLEX],root=0)

    if myid == 0:
        np.save(inDir+phononDir+phvalname,val)
        vec.tofile(inDir+phononDir+phvecname)

    MPI.Win.Free(win)
    end = time.time()
    if myid == 0:
        print('Running time: %.2fs'%(end-start))


########################################

# get comm and shm_comm
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nprocs = comm.Get_size()

shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
nprocs_shm = shm_comm.Get_size()
shm_id = shm_comm.Get_rank()
nnodes = nprocs//nprocs_shm

s_d = 8
qnum = nq[0]*nq[1]*nq[2]
#qnum = 600
#
#if (shm_id==0): len_qlist = qnum*3*s_d
#else: len_qlist = 0
#win03 = MPI.Win.Allocate_shared(len_qlist,s_d,comm=shm_comm)
#buf03,s_d = win03.Shared_query(0)
#q_list = np.ndarray(buffer=buf03,dtype=float,shape=(qnum,3))
#if shm_id == 0:
#    q_list[0:100] = np.linspace([0,0,0],[1/2,0,1/2],100,endpoint=False)
#    q_list[100:200] = np.linspace([1/2,0,1/2],[1/2,1/4,3/4],100,endpoint=False)
#    q_list[200:300] = np.linspace([1/2,1/4,3/4],[1/2,0,1/2],100,endpoint=False)
#    q_list[300:400] = np.linspace([1/2,0,1/2],[5/8,1/4,5/8],100,endpoint=False)
#    q_list[400:500] = np.linspace([3/8,3/4,3/8],[0,0,0],100,endpoint=False)
#    q_list[500:600] = np.linspace([0,0,0],[1/2,1/2,1/2],100,endpoint=False)

q_list = np.zeros((0,3))
shm_comm.Barrier()
phonon(inDir,phononDir,infile,qnum,q_list)

#MPI.Win.Free(win03)
