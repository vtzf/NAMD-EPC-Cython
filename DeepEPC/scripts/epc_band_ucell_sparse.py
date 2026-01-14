import os
import time
from glob import glob
import numpy as np
import h5py
from scipy.linalg import eigh, eig
import json
import configparser
from mpi4py import MPI

conf = configparser.ConfigParser()
conf.read('config.ini',encoding='utf-8')

inDir = conf['epc']['inDir']+'/'
bandDir = conf['epc']['bandDir']+'/'
infile = conf['epc']['infile_out']
IsH5 = True if conf['epc']['IsH5']=='True' else False
valname = conf['epc']['valname']
vecname = conf['epc']['vecname']

nq_str = conf['epc']['nq']
nq_list = nq_str[1:-1].split(',')
nq = [int(i) for i in nq_list]

atom_str = conf['epc']['atom']
atom_list = atom_str[1:-1].split(',')
atom = [int(i) for i in atom_list]
orbital_str = conf['epc']['orbital']
orbital_list = orbital_str[1:-1].split(',')
orbital = [int(i) for i in orbital_list]

atomnum = sum(atom)
atom_type = [x*y for x,y in zip(atom,orbital)]
atom_idx0 = [y for x,y in zip(atom,orbital) for i in range(x)]
atom_idx = [sum(atom_idx0[0:i]) for i in range(len(atom_idx0)+1)]
norbital = atom_idx[-1]
atom_idx = np.array(atom_idx,dtype=int)
atom_idx0 = np.array(atom_idx0,dtype=int)

IsAllVec = True if conf['epc']['IsAllVec']=='True' else False
if IsAllVec:
    bmin = 0
    bmax = norbital
    nbands = norbital
else:
    bmin = int(conf['epc']['bmin'])
    bmax = int(conf['epc']['bmax'])
    nbands = bmax-bmin+1

IsAllKlist = True if conf['epc']['IsAllKlist']=='True' else False
if IsAllKlist:
    emin = -1.0e8
    emax = 1.0e8
else:
    emin = float(conf['epc']['emin'])
    emax = float(conf['epc']['emax'])

IsH5_u = True if conf['ucell']['IsH5_u']=='True' else False

Ry2eV = 13.605698065894
Hartree2eV = Ry2eV*2

# get comm and shm_comm
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
nprocs = comm.Get_size()

shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
nprocs_shm = shm_comm.Get_size()
shm_id = shm_comm.Get_rank()
nnodes = nprocs//nprocs_shm

# get R_num
def ReadRnumScfout(Name):
    fp = open(Name,'rb')
    fp.seek(0)
    i_vec = np.fromfile(fp,dtype=np.intc,count=6)
    atomnum = i_vec[0]
    TCpyCell = i_vec[5]
    fp.seek(4+(TCpyCell+1)*4*(8+4),1)
    fp.seek(atomnum*4,1)

    FNAN = np.zeros((atomnum+1),dtype=np.intc)
    FNAN[1:] = np.fromfile(fp,dtype=np.intc,count=atomnum)
    natn = [[]]
    n_num = 0
    for ct_AN in range(1,atomnum+1):
        natn.append(np.fromfile(fp,dtype=np.intc,count=FNAN[ct_AN]+1))
        n_num += FNAN[ct_AN]+1
    ncn_set = np.zeros((n_num),dtype=np.intc)
    n_count = 0
    for ct_AN in range(1,atomnum+1):
        tmp = np.fromfile(fp,dtype=np.intc,count=FNAN[ct_AN]+1)
        ncn_set[n_count:n_count+FNAN[ct_AN]+1] = tmp
        n_count += FNAN[ct_AN]+1

    ncn_set = list(set(ncn_set))

    return len(ncn_set)


def ReadRnumH5(h5Dir):
    f=h5py.File(h5Dir,'r')
    h_key = list(f.keys())
    h_key = np.array([json.loads(x) for x in h_key],dtype=int)
    R_list = np.unique(h_key[:,0:3],axis=0)

    return R_list.shape[0]


if myid == 0:
    if IsH5_u:
        R_num = ReadRnumH5(inDir+'ucell/out/hamiltonians.h5')
    else:
        R_num = ReadRnumScfout(glob(inDir+'ucell/*.scfout')[0])
else:
    R_num = 0
R_num = comm.bcast(R_num,root=0)


# get sparse matrix info
def read_key0(HamName,R_list,k_num,IsH5):
    if IsH5:
        f=h5py.File(HamName,'r')
        h_key = list(f.keys())
        h_key = np.array([json.loads(x) for x in h_key],dtype=int)
    else:
        fp = open(HamName,'rb')
        fp.seek(0)
        i_vec = np.fromfile(fp,dtype=np.intc,count=6)
        atomnum = i_vec[0]
        TCpyCell = i_vec[5]
        fp.seek(4+(TCpyCell+1)*4*8,1)
        
        atv_ijk = np.zeros((TCpyCell+1,4),dtype=np.intc)
        for Rn in range(TCpyCell+1):
            atv_ijk[Rn] = np.fromfile(fp,dtype=np.intc,count=4)
        fp.seek(atomnum*4,1)

        FNAN = np.zeros((atomnum+1),dtype=np.intc)
        FNAN[1:] = np.fromfile(fp,dtype=np.intc,count=atomnum)
        natn = [[]]
        n_num = 0
        for ct_AN in range(1,atomnum+1):
            natn.append(np.fromfile(fp,dtype=np.intc,count=FNAN[ct_AN]+1))
            n_num += FNAN[ct_AN]+1
        ncn = np.zeros((n_num),dtype=np.intc)
        n_count = 0
        for ct_AN in range(1,atomnum+1):
            tmp = np.fromfile(fp,dtype=np.intc,count=FNAN[ct_AN]+1)
            ncn[n_count:n_count+FNAN[ct_AN]+1] = tmp
            n_count += FNAN[ct_AN]+1

        h_key = np.zeros((n_count,5),dtype=int)
        h_key[:,0:3] = atv_ijk[ncn,1:]
        n_count = 0
        for ct_AN in range(1,atomnum+1):
            h_key[n_count:n_count+FNAN[ct_AN]+1,3] = ct_AN
            h_key[n_count:n_count+FNAN[ct_AN]+1,4] = natn[ct_AN]
            n_count += FNAN[ct_AN]+1

    R_list[:] = np.unique(h_key[:,0:3],axis=0)
    key_idx = []
    for i in range(R_num):
        key_idx0 = np.where(
            (h_key[:,0]==R_list[i,0])
            &(h_key[:,1]==R_list[i,1])
            &(h_key[:,2]==R_list[i,2])
        )[0]
        key_num[i,0] = key_idx0.shape[0]
        key_num[i,1] = np.sum(atom_idx0[h_key[key_idx0,3]-1]\
                            * atom_idx0[h_key[key_idx0,4]-1])
        key_idx.append(key_idx0)

    key_num[0,2:4] = 0
    key_num[1:,2] = np.cumsum(key_num[:-1,0])
    key_num[1:,3] = np.cumsum(key_num[:-1,1])

    return h_key, key_idx


def read_key1(h_key,key_idx,pub_key,key_info):
    for i in range(R_num):
        idx_s = key_num[i,2]
        idx_e = key_num[i+1,2]
        key_i = h_key[key_idx[i],3]-1
        key_j = h_key[key_idx[i],4]-1
        pub_key[idx_s:idx_e,0] = key_i
        pub_key[idx_s:idx_e,1] = key_j
        pub_key[idx_s:idx_e,2] = atom_idx0[key_i]
        pub_key[idx_s:idx_e,3] = atom_idx0[key_j]

    offset = 0
    for h in range(key_num[R_num,2]):
        for i in range(pub_key[h,2]):
            for j in range(pub_key[h,3]):
                k = i*pub_key[h,3]+j
                key_info[k+offset,2] \
                = (i+atom_idx[pub_key[h,0]])*norbital\
                + (j+atom_idx[pub_key[h,1]])
        offset += pub_key[h,2]*pub_key[h,3]

    key_info[:,0] = key_info[:,2]//norbital
    key_info[:,1] = key_info[:,2]%norbital


def read_key0_sum(key_info):
    key_buf = key_info[:,2].copy()
    key_info_s = np.unique(key_buf)
    key_num_s = key_info_s.shape[0]
    mapidx = np.zeros((key_info_s[key_num_s-1]+1),dtype=int)
    mapidx[key_info_s] = np.arange(key_num_s)
    key_info[:,3] = mapidx[key_info[:,2]]

    return key_num_s


s_d = 8
s_int = 4
if (shm_id==0):
    len_Rlist = R_num*3*s_d
    len_keynum = (R_num+1)*4*s_int
else:
    len_Rlist = 0
    len_keynum = 0

win2 = MPI.Win.Allocate_shared(len_Rlist,s_d,comm=shm_comm)
buf2,s_d = win2.Shared_query(0)
R_list = np.ndarray(
    buffer=buf2,dtype=float,shape=(R_num,3)
)
win00 = MPI.Win.Allocate_shared(len_keynum,s_int,comm=shm_comm)
buf00,s_int = win00.Shared_query(0)
key_num = np.ndarray(
    buffer=buf00,dtype=np.int32,shape=(R_num+1,4)
)
shm_comm.Barrier()
if (shm_id==0):
    if IsH5_u:
        data_name = inDir+'ucell/out/hamiltonians.h5'
    else:
        data_name = glob(inDir+'ucell/*.scfout')[0]
    h_key, key_idx = read_key0(data_name,R_list,key_num,IsH5_u)
shm_comm.Barrier()
if (shm_id==0):
    len_ham = key_num[R_num,3]*s_d
    len_olp = key_num[R_num,3]*s_d
    len_pubkey = key_num[R_num,2]*4*s_int
    len_key = key_num[R_num,3]*4*s_int
else:
    len_ham = 0
    len_olp = 0
    len_pubkey = 0
    len_key = 0
win = MPI.Win.Allocate_shared(len_ham,s_d,comm=shm_comm)
buf,s_d = win.Shared_query(0)
ham = np.ndarray(
    buffer=buf,dtype=float,shape=(key_num[R_num,3])
)
win1 = MPI.Win.Allocate_shared(len_olp,s_d,comm=shm_comm)
buf1,s_d = win1.Shared_query(0)
olp = np.ndarray(
    buffer=buf1,dtype=float,shape=(key_num[R_num,3])
)
win01 = MPI.Win.Allocate_shared(len_pubkey,s_int,comm=shm_comm)
buf01,s_int = win01.Shared_query(0)
pub_key = np.ndarray(
    buffer=buf01,dtype=np.int32,shape=(key_num[R_num,2],4)
)
win02 = MPI.Win.Allocate_shared(len_key,s_int,comm=shm_comm)
buf02,s_int = win02.Shared_query(0)
key_info = np.ndarray(
    buffer=buf02,dtype=np.int32,shape=(key_num[R_num,3],4)
)
if (shm_id==0):
    read_key1(h_key,key_idx,pub_key,key_info)
shm_comm.Barrier()
#key_num_s = read_key0_sum(key_info)


def eig_k(ham,olp,ham_k,olp_k,val,vec,ik,R_list,kpoint):
    starttime = time.time()
    ham_k[:] = 0.0
    olp_k[:] = 0.0
    phase = np.exp(2j*np.pi*np.dot(R_list,kpoint))
    for i in range(R_num):
        idx_s = key_num[i,3]
        idx_e = key_num[i+1,3]
        key0 = key_info[idx_s:idx_e,0]
        key1 = key_info[idx_s:idx_e,1]
        ham_k[key0,key1] += ham[idx_s:idx_e]*phase[i]
        olp_k[key0,key1] += olp[idx_s:idx_e]*phase[i]

    if IsAllVec:
        val[ik],vec[ik] \
        = eigh(
            a=ham_k,b=olp_k,overwrite_a=True,overwrite_b=True,
            eigvals_only=False,driver='gvd'
        )
    else:
        val[ik],vec[ik] \
        = eigh(            
            a=ham_k,b=olp_k,overwrite_a=True,overwrite_b=True,
            eigvals_only=False,subset_by_index=[bmin,bmax],driver='gvx'
        )
    endtime = time.time()
    print("Eigenvalue problem in kpoint=[%.5f,%.5f,%.5f]: %.4fs."%(\
          kpoint[0],kpoint[1],kpoint[2],endtime-starttime),flush=True)


def ReadH5(HamName,mat,R_list,key_num,pub_key):
    f=h5py.File(HamName,'r')
    k = 0
    for i in range(R_num):
        for j in range(key_num[i,2],key_num[i+1,2]):
            key_buf = '[%d, %d, %d, %d, %d]'%(
                R_list[i,0],R_list[i,1],R_list[i,2],
                pub_key[j,0]+1,pub_key[j,1]+1
            )
            buflen = pub_key[j,2]*pub_key[j,3]
            mat[k:k+buflen] = f[key_buf][:].reshape(buflen)
            k += buflen


def ReadScfout(Name,hamil,olp,R_list,key_num):
    fp = open(Name,'rb')
    fp.seek(0)
    i_vec = np.fromfile(fp,dtype=np.intc,count=6)
    atomnum = i_vec[0]
    TCpyCell = i_vec[5]
    fp.seek(4+(TCpyCell+1)*8*4,1)

    atv_ijk = np.zeros((TCpyCell+1,4),dtype=np.intc)
    for Rn in range(TCpyCell+1):
        atv_ijk[Rn] = np.fromfile(fp,dtype=np.intc,count=4)

    Total_NumOrbs = np.ones((atomnum+1),dtype=np.intc)
    Total_NumOrbs[1:] = np.fromfile(fp,dtype=np.intc,count=atomnum)
    FNAN = np.zeros((atomnum+1),dtype=np.intc)
    FNAN[1:] = np.fromfile(fp,dtype=np.intc,count=atomnum)
    natn = [[]]
    for ct_AN in range(1,atomnum+1):
        natn.append(np.fromfile(fp,dtype=np.intc,count=FNAN[ct_AN]+1))
    ncn = [[]]
    for ct_AN in range(1,atomnum+1):
        tmp = np.fromfile(fp,dtype=np.intc,count=FNAN[ct_AN]+1)
        ncn.append(tmp)

    fp.seek((3+3+atomnum)*4*8,1)

    R_num = R_list.shape[0]
    R_key = {'%d,%d,%d'%(R_list[i,0],R_list[i,1],R_list[i,2]):i for i in range(R_num)}

    key_idx = key_num[:,3].copy()
    for ct_AN in range(1,atomnum+1):
        TNO1 = Total_NumOrbs[ct_AN]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            TNO2 = Total_NumOrbs[Gh_AN]
            R = atv_ijk[ncn[ct_AN][h_AN]]
            Ridx = R_key['%d,%d,%d'%(R[1],R[2],R[3])]
            Hks1 = np.fromfile(fp,dtype=float,count=TNO1*TNO2)
            offset = key_idx[Ridx]
            hamil[offset:offset+TNO1*TNO2] = Hks1*Hartree2eV
            key_idx[Ridx] += TNO1*TNO2

    key_idx = key_num[:,3].copy()
    for ct_AN in range(1,atomnum+1):
        TNO1 = Total_NumOrbs[ct_AN]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            TNO2 = Total_NumOrbs[Gh_AN]
            R = atv_ijk[ncn[ct_AN][h_AN]]
            Ridx = R_key['%d,%d,%d'%(R[1],R[2],R[3])]
            OLP1 = np.fromfile(fp,dtype=float,count=TNO1*TNO2)
            offset = key_idx[Ridx]
            olp[offset:offset+TNO1*TNO2] = OLP1
            key_idx[Ridx] += TNO1*TNO2

    fp.close()


def band_cal(inDir,bandDir,k_list):
    start = time.time()
    # create Dir
    if myid == 0:
        if not os.path.exists(inDir+bandDir):
            os.mkdir(inDir+bandDir)
    # read data
    if shm_id == 0:
        if IsH5_u:
            ReadH5(inDir+'ucell/out/hamiltonians.h5',ham,R_list,key_num,pub_key)
            ReadH5(inDir+'ucell/out/overlaps.h5',olp,R_list,key_num,pub_key)
        else:
            ReadScfout(glob(inDir+'ucell/*.scfout')[0],ham,olp,R_list,key_num)
    # alloc val/vec
    kproc_num = np.zeros((nprocs),dtype=np.int32)
    kproc = np.zeros((nprocs+1),dtype=np.int32)
    knum = k_list.shape[0]
    for i in range(nprocs):
        knum_min = (knum*i)//nprocs
        knum_max = (knum*(i+1))//nprocs
        kproc_num[i] = knum_max - knum_min
        kproc[i+1] = knum_max
    val_p = np.zeros((kproc_num[myid],nbands))
    vec_p = np.zeros((kproc_num[myid],norbital,nbands),dtype=complex)
    if myid == 0:
        val = np.zeros((knum,nbands))
        vec = np.zeros((knum,norbital,nbands),dtype=complex)
    else:
        val = np.zeros((0,0))
        vec = np.zeros((0,0,0),dtype=complex)
    comm.Barrier()
    # eigen
    ham_k = np.empty((norbital,norbital),dtype=complex)
    olp_k = np.empty((norbital,norbital),dtype=complex)
    kpoint = np.zeros((3),dtype=float)
    kproc_min = kproc[myid]
    kproc_max = kproc[myid+1]
    k_idx_list = []
    k_idx_num_p = 0
    for i in range(kproc_min,kproc_max):
        kpoint[:] = k_list[i]
        eig_k(
            ham,olp,ham_k,olp_k,val_p,
            vec_p,i-kproc_min,R_list,kpoint
        )
        for j in range(nbands):
#            if val_p[i-kproc_min,j]>=emin and val_p[i-kproc_min,j]<=emax \
#            and kpoint[0]<0.5 and kpoint[1]<0.5:
            if val_p[i-kproc_min,j]>=emin and val_p[i-kproc_min,j]<=emax:
                k_idx_list.append([i,j])
                k_idx_num_p += 1

    comm.Gatherv(val_p,[val,kproc_num*nbands,\
                 kproc[0:nprocs]*nbands,MPI.DOUBLE],root=0)
    comm.Gatherv(vec_p,[vec,kproc_num*norbital*nbands,\
                 kproc[0:nprocs]*norbital*nbands,MPI.DOUBLE_COMPLEX],root=0)

    if not IsAllKlist:
        if k_idx_num_p > 0:
            k_idx_p = np.array(k_idx_list,dtype=np.int32)
            val_idx_p = np.ascontiguousarray(val_p[k_idx_p[:,0]-kproc_min,k_idx_p[:,1]])
            vec_idx_p = np.ascontiguousarray(vec_p[k_idx_p[:,0]-kproc_min,:,k_idx_p[:,1]])
        else:
            k_idx_p = np.zeros((0,0),dtype=np.int32)
            val_idx_p = np.zeros((0))
            vec_idx_p = np.zeros((0,0),dtype=complex)
    
        k_idx_num = comm.allreduce(k_idx_num_p,op=MPI.SUM)
        k_idx_proc_num = np.zeros((nprocs),dtype=np.int32)
        k_idx_proc = np.zeros((nprocs+1),dtype=np.int32)
        k_idx_proc_num[myid] = k_idx_num_p
        comm.Allgather(MPI.IN_PLACE,k_idx_proc_num)
        k_idx_proc[1:] = np.cumsum(k_idx_proc_num)
        if myid == 0:
            k_idx = np.zeros((k_idx_num,2),dtype=np.int32)
            val_idx = np.zeros((k_idx_num))
            vec_idx = np.zeros((k_idx_num,norbital),dtype=complex)
        else:
            k_idx = np.zeros((0,0),dtype=np.int32)
            val_idx = np.zeros((0))
            vec_idx = np.zeros((0,0),dtype=complex)
        comm.Gatherv(k_idx_p,[k_idx,k_idx_proc_num*2,\
                     k_idx_proc[0:nprocs]*2,MPI.INT],root=0)
        comm.Gatherv(val_idx_p,[val_idx,k_idx_proc_num,\
                     k_idx_proc[0:nprocs],MPI.DOUBLE],root=0)
        comm.Gatherv(vec_idx_p,[vec_idx,k_idx_proc_num*norbital,\
                     k_idx_proc[0:nprocs]*norbital,MPI.DOUBLE_COMPLEX],root=0)
    
    if myid == 0:
        np.save(inDir+bandDir+valname,val)
        np.save(inDir+bandDir+vecname,vec)
#        np.save(valname,val)
#        np.save(vecname,vec)
        if not IsAllKlist:
            np.save(inDir+bandDir+'/bassel-%d.npy'%nq[0],k_idx)
#            np.save('bassel-%d.npy'%nq[0],k_idx)
            valpname = valname.split('.')[0]+'_p.npy'
            vecpname = vecname.split('.')[0]+'_p.npy'
            np.save(inDir+bandDir+valpname,val_idx)
            np.save(inDir+bandDir+vecpname,vec_idx)
#            np.save(valpname,val_idx)
#            np.save(vecpname,vec_idx)

    end = time.time()
    if myid == 0:
        print('Running time: %.2fs'%(end-start))

knum = nq[0]*nq[1]*nq[2]
#knum = 700

if (shm_id==0): len_klist = knum*3*s_d
else: len_klist = 0
win03 = MPI.Win.Allocate_shared(len_klist,s_d,comm=shm_comm)
buf03,s_d = win03.Shared_query(0)
k_list = np.ndarray(buffer=buf03,dtype=float,shape=(knum,3))
if shm_id == 0:
    kidx = np.zeros((knum,3),dtype=int)
    idx = np.arange(knum)
    kidx[:,2] = idx%nq[2]; kidx_xy = idx//nq[2]
    kidx[:,1] = kidx_xy%nq[1]; kidx[:,0] = kidx_xy//nq[1]
    k_list[:] = kidx/nq
#    k_list[0:100] = np.linspace([0,0,0],[0,1/2,0],100,endpoint=False)
#    k_list[100:200] = np.linspace([0,1/2,0],[1/2,1/2,0],100,endpoint=False)
#    k_list[200:300] = np.linspace([1/2,1/2,0],[0,0,0],100,endpoint=False)
#    k_list[300:400] = np.linspace([0,0,0],[0,0,1/2],100,endpoint=False)
#    k_list[400:500] = np.linspace([0,0,1/2],[0,1/2,1/2],100,endpoint=False)
#    k_list[500:600] = np.linspace([0,1/2,1/2],[1/2,1/2,1/2],100,endpoint=False)
#    k_list[600:700] = np.linspace([1/2,1/2,1/2],[0,0,1/2],100,endpoint=False)


shm_comm.Barrier()
band_cal(inDir,bandDir,k_list)
MPI.Win.Free(win)
MPI.Win.Free(win1)
MPI.Win.Free(win2)
MPI.Win.Free(win00)
MPI.Win.Free(win01)
MPI.Win.Free(win02)
MPI.Win.Free(win03)
