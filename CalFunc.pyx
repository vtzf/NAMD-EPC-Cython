#cython: language_level=3
#cython: cdivision=True

cimport cython
import numpy as np
import ctypes
cimport numpy as np
from mpi4py import MPI
from mpi4py cimport MPI
from mpi4py cimport libmpi as mpi
from libc.math cimport cos, sin, sqrt, exp, fabs, fmod, M_PI
from libc.stdio cimport sprintf, printf, FILE, \
     SEEK_SET, SEEK_CUR, SEEK_END, fopen, fseek, fread, fwrite, fclose
from libc.string cimport memcpy, memset, strcmp
from libc.stdlib cimport malloc, calloc, free, qsort
import os
from cython.parallel import prange

cdef extern from "complex.h" nogil:
    double complex conj(double complex)
    double creal(double complex)
    double cimag(double complex)
    double complex cexp(double complex)
    double complex ccos(double complex)
    double complex csin(double complex)
    double cabs(double complex)

cdef extern from "mkl.h" nogil:
    cdef enum CBLAS_LAYOUT:
        CblasRowMajor=101
        CblasColMajor=102
    cdef enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
    cdef enum CBLAS_UPLO:
        CblasUpper=121
        CblasLower=122

    cdef void cblas_dgemv(
        CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, int M, int N,
        double alpha, double *A, int lda, double *X, int incX,
        double beta, double *Y, int incY
    )
    cdef void cblas_zgemv(
        CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, int M, int N,
        void *alpha, void *A, int lda, void *X, int incX,
        void *beta, void *Y, int incY
    )
    cdef void cblas_zgemm(
        CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB, int M, int N, int K,
        void* alpha, void* A, int lda, void* B, int ldb,
        void* beta, void* C, int ldc
    )


# row-cache shm budget in bytes per node; override at build time
cdef extern from *:
    """
    #ifndef SHM_MAX_BYTES
    #define SHM_MAX_BYTES 53687091200L
    #endif
    """
    const long SHM_MAX_BYTES


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int Cmp_l(void * pa, void * pb) nogil:
    cdef long *pa1 = <long*>pa
    cdef long *pb1 = <long*>pb

    if pa1[0]>pb1[0]:
        return 1
    elif pa1[0]<pb1[0]:
        return -1
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int Cmp_s(void * pa, void * pb) nogil:
    cdef int *pa1 = <int*>pa
    cdef int *pb1 = <int*>pb

    if pa1[0]>pb1[0]:
        return 1
    elif pa1[0]<pb1[0]:
        return -1
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int Cmp_e(void * pa, void * pb) nogil:
    cdef int *pa1 = <int*>pa
    cdef int *pb1 = <int*>pb

    if pa1[3]>pb1[3]:
        return 1
    elif pa1[3]<pb1[3]:
        return -1
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
def GetIniIdx(
    int[::1] ekidx, int[::1] ebidx, 
    int k_state, int b_state, int nk_a
):
    cdef int i, idx

    idx = -1
    for i in range(nk_a):
        if (ekidx[i]==k_state and ebidx[i]==b_state):
            idx = i
            break
    
    return idx


@cython.boundscheck(False)
@cython.wraparound(False)
def GetPhQ(
    double[:,::1] phonon, int nmodes, int nq, int istep,
    double[:,::1] BEfactor, double dt, double hbar, double KbT,
    double complex[:,::1] phq0, double complex[:,::1] phq
):
    cdef int i, k
    cdef double KbT1 = 1.0/KbT
    cdef double factor0 = dt/hbar
    cdef double factor1 = factor0*istep
    cdef double bose

    for k in range(nmodes):
        for i in range(nq):
            bose = 1.0/(exp(phonon[k,i]*KbT1)-1.0)
            BEfactor[k,i] = sqrt(bose+0.5)
            phq0[k,i] = cexp(-1j*(phonon[k,i]*factor0))
            phq[k,i] = BEfactor[k,i]*cexp(-1j*(phonon[k,i]*factor1))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void GetH0(
    int nmodes, int nk_p, int nk_a, int nq,
    int nk_min, int k_p_m, int[:,::1] kqidx_p,
    double complex[::1] E_p, double complex factor, double np2,
    double complex[:,:,::1] epc_a, double complex[:,::1] phq_t, 
    double complex[:,::1] phq, double complex[:,::1] H
):
    cdef int i, j, k
    cdef double phq_tmp

    for i in range(nmodes):
        for j in range(nq):
            phq_t[i,j] = 2*phq[i,j]*factor*np2
        for j in range(nk_p):
            for k in range(nk_a):
                phq_tmp = cimag(phq_t[i,kqidx_p[j+k_p_m,k]])
                H[j,k] += creal(epc_a[i,j+k_p_m,k])*phq_tmp*1j \
                        - cimag(epc_a[i,j+k_p_m,k])*phq_tmp

    for i in range(nk_p):
        H[i,i+nk_min] += E_p[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void GetH(
    int nmodes, int nk_p, int nk_a, int nq,
    int nk_min, int k_p_m, int[:,::1] kqidx_p, 
    double wht, double complex[::1] E_p,
    double complex[:,:,::1] epc_a, double complex[:,::1] phq_t,
    double complex[:,::1] phq0_t, double complex[:,::1] H0_p, 
    double complex[:,::1] H1_p, double complex[:,::1] dH01
):
    cdef int i, j, k
    cdef double phq_tmp

    for i in range(nmodes):
        for j in range(nq):
            phq_t[i,j] *= phq0_t[i,j]
        for j in range(nk_p):
            for k in range(nk_a):
                phq_tmp = cimag(phq_t[i,kqidx_p[j+k_p_m,k]])
                H1_p[j,k] += creal(epc_a[i,j+k_p_m,k])*phq_tmp*1j \
                           - cimag(epc_a[i,j+k_p_m,k])*phq_tmp

    for i in range(nk_p):
        H1_p[i,i+nk_min] += E_p[i]
        for j in range(nk_a):
            dH01[i,j] = (H1_p[i,j]-H0_p[i,j])*wht
            H1_p[i,j] = 0


# integration function
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void TimeProp(
    mpi.MPI_Comm shm_comm, mpi.MPI_Comm remote_comm, mpi.MPI_Datatype CPLX_N,
    int shm_id, int nnode, int istep, int estep, int nel, int nk_a,
    int nk_p, int nk_min, int[::1] psi_p_remote, int[::1] psi_p_remote_num,
    double complex[:,::1] H0_p, double complex[:,::1] dH01,
    double complex * psi_a1, double complex * psi_a2, double complex * psi_buf
):
    cdef int i, j
    cdef long nel_l = nel
    cdef long nk_a_l = nk_a
    cdef double complex zero = 0.0
    cdef double complex psi_t, factor
    if istep==0 and estep==0: factor = 0.5
    else: factor = 1.0

    mpi.MPI_Barrier(shm_comm)
    cblas_zgemm(
        CblasRowMajor,CblasNoTrans,CblasNoTrans,
        nk_p,nel,nk_a,&factor,&H0_p[0,0],nk_a,
        psi_a1,nel,&zero,psi_buf,nel
    )
    for i in range(nk_p):
        for j in range(nel_l):
            psi_a2[(nk_min+i)*nel_l+j] += psi_buf[i*nel_l+j]
    mpi.MPI_Barrier(shm_comm)
    if remote_comm != mpi.MPI_COMM_NULL:
        if nnode > 1:
            mpi.MPI_Allgatherv(
                mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,
                psi_a2,&psi_p_remote_num[0],&psi_p_remote[0],
                CPLX_N,remote_comm
            )
    mpi.MPI_Barrier(shm_comm)
    if shm_id==0:
        for i in range(nk_a_l*nel_l):
            psi_t = psi_a1[i]
            psi_a1[i] = psi_a2[i]
            psi_a2[i] = psi_t
    mpi.MPI_Barrier(shm_comm)

    for i in range(nk_p):
        for j in range(nk_a):
            H0_p[i,j] += dH01[i,j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void WritePsi(
    mpi.MPI_Comm c_comm, mpi.MPI_Comm shm_comm, int myid, int shm_id,
    int shm_nprocs, char* namddir, int sample, int istep, int nel, int nk_a,
    int nel_p, int nel_min, int[::1] nel_proc, int[::1] nel_proc_num,
    double[::1] e_psi, double* energy, double* e_psi_buf,
    double* pop_buf, double complex * psi_a, double complex * psi_buf
):
    cdef int i, nk_a_min, nk_a_max
    cdef long j
    cdef long nel_l = nel
    cdef double complex psi_t
    cdef char data_name[500]
    cdef FILE * fp

    mpi.MPI_Barrier(shm_comm)
    nk_a_min = <int>((shm_id*nk_a)/shm_nprocs)
    nk_a_max = <int>(((shm_id+1)*nk_a)/shm_nprocs)
    for i in range(nk_a_min,nk_a_max):
        for j in range(nel_l):
            psi_buf[j*nk_a+i] = psi_a[i*nel_l+j]
    mpi.MPI_Barrier(shm_comm)
    for j in range(nel_p):
        for i in range(nk_a):
            psi_t = psi_buf[(j+nel_min)*nk_a+i]
            pop_buf[j*nk_a+i] = creal(psi_t)*creal(psi_t) \
                               +cimag(psi_t)*cimag(psi_t)
    cblas_dgemv(
        CblasRowMajor,CblasNoTrans,nel_p,nk_a,
        1.0,pop_buf,nk_a,energy,1,0.0,e_psi_buf,1
    )
    mpi.MPI_Gatherv(
        e_psi_buf,nel_p,mpi.MPI_DOUBLE,&e_psi[0],
        &nel_proc_num[0],&nel_proc[0],mpi.MPI_DOUBLE,0,c_comm
    )
    if myid == 0:
        sprintf(data_name,"%s/psi-%d.dat",namddir,sample)
        if istep == 0: fp = fopen(data_name,"wb")
        else: fp = fopen(data_name,"ab")
        fwrite(psi_buf,16,nel_l*nk_a,fp)
        fclose(fp)
    mpi.MPI_Barrier(c_comm)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fssh_psi(
    mpi.MPI_Comm c_comm, mpi.MPI_Comm shm_comm, int myid, int nprocs,
    int shm_id, int shm_nprocs, str namddir, int sample, int NSW, int NELM,
    int k_p_m, int nk_a, int nq, int n_p, int nmodes, int nel, double edt,
    double hbar, int[::1] state_s, int[::1] k_proc, int[::1] k_proc_num,
    int[:,::1] kqidx_p, double[::1] energy, double complex[:,:,::1] epc_a,
    double complex[:,::1] phq, double complex[:,::1] phq0
):
    cdef mpi.MPI_Comm remote_comm
    cdef mpi.MPI_Win win1, win2, win3
    cdef mpi.MPI_Aint len_psi_byte, len_psi_byte1, len_psi_byte2, len_own
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef char * own_qp
    cdef int[::1] psi_p_remote, psi_p_remote_num

    cdef int s_int = sizeof(int)
    cdef int s_d = sizeof(double)
    cdef int s_dcplx = sizeof(double complex)
    cdef long nel_l = nel

    cdef int i, j, k, shm_proc_s, shm_proc_e, ierr, nnode, \
             nk_a_min, nk_a_max, nfail, nfail_tot, s_own
    cdef int * nodelist
    cdef double wht = 1.0/NELM
    cdef double complex factor = 2*edt/(1j*hbar)
    cdef int nk_min = k_proc[myid]
    cdef int nk_proc = k_proc_num[myid]
    cdef double complex[::1] E_p = np.zeros((nk_proc),dtype=np.complex128)
    cdef double np2 = 1.0/sqrt(n_p)

    cdef double complex[:,::1] phq_t = np.zeros((nmodes,nq),dtype=np.complex128)
    cdef double complex[:,::1] H0_p = np.zeros((nk_proc,nk_a),dtype=np.complex128)
    cdef double complex[:,::1] H1_p = np.zeros((nk_proc,nk_a),dtype=np.complex128)
    cdef double complex[:,::1] dH01 = np.zeros((nk_proc,nk_a),dtype=np.complex128)

    cdef double complex * psi_a1
    cdef double complex * psi_a2
    cdef double complex * psi_a_buf = \
        <double complex*>malloc(nk_proc*nel_l*s_dcplx)
    cdef double complex * psi_buf
    cdef int[::1] nel_proc_num = np.zeros((nprocs),dtype=np.int32)
    cdef int[::1] nel_proc = np.zeros((nprocs+1),dtype=np.int32)
    for i in range(nprocs):
        nel_proc[i+1] = ((i+1)*nel)/nprocs
        nel_proc_num[i] = nel_proc[i+1]-nel_proc[i]
    cdef int nel_p = nel_proc_num[myid]
    cdef int nel_min = nel_proc[myid]
    cdef double * e_psi_buf = <double*>malloc(s_d*nel_p)
    cdef double * pop_buf = <double*>malloc(s_d*nk_a*nel_p)
    namddir_t = namddir.encode("utf-8")
    cdef char* namddir_c = namddir_t

    cdef double starttime, endtime

    cdef mpi.MPI_Datatype CPLX_N
    mpi.MPI_Type_contiguous(nel,mpi.MPI_DOUBLE_COMPLEX,&CPLX_N)
    mpi.MPI_Type_commit(&CPLX_N)

    cdef double[:,::1] e_psi
    if myid == 0:
        e_psi = np.zeros((NSW,nel),dtype=np.float64)
    else:
        e_psi = np.zeros((NSW,0),dtype=np.float64)

    nnode = nprocs/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&remote_comm)

    psi_p_remote = np.zeros((nnode+1),dtype=np.int32)
    psi_p_remote_num = np.zeros((nnode),dtype=np.int32)
    for i in range(nnode):
        shm_proc_s = i*shm_nprocs
        shm_proc_e = (i+1)*shm_nprocs
        for j in range(shm_proc_s,shm_proc_e):
            psi_p_remote_num[i] += k_proc_num[j]
        for j in range(i+1,nnode+1):
            psi_p_remote[j] += psi_p_remote_num[i]

    nk_a_min = <int>((shm_id*nk_a)/shm_nprocs)
    nk_a_max = <int>(((shm_id+1)*nk_a)/shm_nprocs)
    len_psi_byte = nel_l*(nk_a_max-nk_a_min)*s_dcplx
    len_psi_byte1 = nel_l*(nk_a_max-nk_a_min)*s_dcplx
    len_psi_byte2 = nel_l*(nk_a_max-nk_a_min)*s_dcplx

    mpi.MPI_Win_allocate_shared(
        len_psi_byte,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&psi_a1,&win1
    )
    mpi.MPI_Win_allocate_shared(
        len_psi_byte1,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&psi_a2,&win2
    )
    mpi.MPI_Win_allocate_shared(
        len_psi_byte2,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&psi_buf,&win3
    )

    if shm_id!=0:
        mpi.MPI_Win_shared_query(
            win1,0,&len_psi_byte,&s_dcplx,&psi_a1
        )
        mpi.MPI_Win_shared_query(
            win2,0,&len_psi_byte1,&s_dcplx,&psi_a2
        )
        mpi.MPI_Win_shared_query(
            win3,0,&len_psi_byte2,&s_dcplx,&psi_buf
        )
    nfail = 0
    mpi.MPI_Win_shared_query(win1,shm_id,&len_own,&s_own,&own_qp)
    if (<char*>own_qp-<char*>psi_a1)!=(nel_l*nk_a_min*s_dcplx): nfail += 1
    mpi.MPI_Win_shared_query(win2,shm_id,&len_own,&s_own,&own_qp)
    if (<char*>own_qp-<char*>psi_a2)!=(nel_l*nk_a_min*s_dcplx): nfail += 1
    mpi.MPI_Win_shared_query(win3,shm_id,&len_own,&s_own,&own_qp)
    if (<char*>own_qp-<char*>psi_buf)!=(nel_l*nk_a_min*s_dcplx): nfail += 1
    mpi.MPI_Allreduce(&nfail,&nfail_tot,1,mpi.MPI_INT,mpi.MPI_SUM,shm_comm)
    if nfail_tot>0:
        if myid==0: printf("shm contiguity FAIL nfail=%d\n",nfail_tot)
        mpi.MPI_Abort(shm_comm,1)
    else:
        if myid==0: printf("shm contiguity OK\n")
    memset(psi_a1+nel_l*nk_a_min,0,nel_l*(nk_a_max-nk_a_min)*s_dcplx)
    memset(psi_a2+nel_l*nk_a_min,0,nel_l*(nk_a_max-nk_a_min)*s_dcplx)
    mpi.MPI_Barrier(shm_comm)
    if shm_id==0:
        for j in range(nel):
            i = state_s[j]
            psi_a1[i*nel_l+j] = 1.0
            psi_a2[i*nel_l+j] = 1.0

    for i in range(nk_proc):
        j = nk_min+i
        E_p[i] = energy[j]*factor

    GetH0(
        nmodes,nk_proc,nk_a,nq,nk_min,k_p_m,
        kqidx_p,E_p,factor,np2,epc_a,phq_t,phq,H0_p
    )
    for j in range(NSW):
        starttime = mpi.MPI_Wtime()
        WritePsi(
            c_comm,shm_comm,myid,shm_id,shm_nprocs,namddir_c,
            sample,j,nel,nk_a,nel_p,nel_min,nel_proc,nel_proc_num,
            e_psi[j],&energy[0],e_psi_buf,pop_buf,psi_a1,psi_buf
        )
        GetH(
            nmodes,nk_proc,nk_a,nq,nk_min,k_p_m,kqidx_p,
            wht,E_p,epc_a,phq_t,phq0,H0_p,H1_p,dH01
        )
        for k in range(NELM):
            TimeProp(
                shm_comm,remote_comm,CPLX_N,shm_id,nnode,
                j,k,nel,nk_a,nk_proc,nk_min,psi_p_remote,
                psi_p_remote_num,H0_p,dH01,psi_a1,psi_a2,psi_a_buf
            )
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("TimeProp in %d fs: %.5fs.\n",j,endtime-starttime)

    if myid == 0:
        np.save(namddir+'/fssh_e_psi-%d.npy'%(sample),e_psi)

    mpi.MPI_Type_free(&CPLX_N)
    if remote_comm != mpi.MPI_COMM_NULL:
        mpi.MPI_Comm_free(&remote_comm)
    mpi.MPI_Group_free(&shm_gp)
    mpi.MPI_Group_free(&split_gp)
    mpi.MPI_Win_free(&win1)
    mpi.MPI_Win_free(&win2)
    mpi.MPI_Win_free(&win3)
    free(nodelist)
    free(psi_a_buf)
    free(pop_buf)
    free(e_psi_buf)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void GetState(
    int myid, int nprocs, int shm_nprocs, int state, 
    int * idx_all, int[::1] k_proc
):
    cdef int state_id0, idx_s, idx_e

    idx_s = 0
    idx_e = nprocs
    while 1:
        if ((idx_e-idx_s)<=1):
            if (state>=k_proc[idx_s]):
                state_id0 = idx_s
            else:
                state_id0 = idx_e
            break
        state_id0 = (idx_s+idx_e)/2
        if (state<k_proc[state_id0]):
            idx_e = state_id0
        else:
            idx_s = state_id0

    idx_all[myid*4] = state
    idx_all[myid*4+1] = state-k_proc[state_id0]
    idx_all[myid*4+2] = state_id0
    idx_all[myid*4+3] = myid


# FSSH functions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void GetHhop(
    int nmodes, int nk_a,
    long mstride, int * kqidx0, double[:,::1] phonon,
    double[:,::1] BE2, double complex * epc_p, 
    double f0, double[::1] H_hop, double *dE
):
    cdef int i, j, qidx
    cdef double ph, dE0, dE1, expdE, epc_p1
    cdef double complex epc_p0

    for i in range(nmodes):
        for j in range(nk_a):
            qidx = kqidx0[j]
            ph = phonon[i,qidx]+1e-8
            dE0 = dE[j]+ph
            dE1 = dE[j]-ph
            expdE = exp(dE0*dE0*f0)+exp(dE1*dE1*f0)
            epc_p0 = epc_p[i*mstride+j]
            epc_p1 = (creal(epc_p0)*creal(epc_p0) \
                     +cimag(epc_p0)*cimag(epc_p0))*BE2[i,qidx]
            H_hop[j] += epc_p1*expdE


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int fsshhop(
    int it, int nmodes, int nk_a,
    long mstride, double complex * psi_a, double[::1] H_hop, double *dE,
    double complex * epc_p, int * kqidx0, double[::1] energy, 
    double[:,::1] phonon, double[:,::1] BE2,
    int state, double f0, double f2, double KbT1,
    double *prop, double *prop_a, double *ph_pop_t, float[:,::1] ph_pop_p, 
    int[::1] n_occ, bint LHOLE, int BANDDEG
):
    cdef int i, j, k, idx_s, idx_e, qidx, state_t
    cdef int state_n = state
    cdef int state_t0 = <int>(state/BANDDEG)
    cdef double complex psi0 = psi_a[state_t0]
    cdef double densitydiag = \
                f2/(creal(psi0)*creal(psi0)+cimag(psi0)*cimag(psi0))
    cdef double dEt, prop_t, epc_p1, prop0, prop_sum, prop_c
    cdef double complex density, epc_p0

    cdef double ph_pop_sum, ph_pop0, ph_pop1, \
                dE0, dE1, expdE0, expdE1

    prop0 = np.random.rand()

    prop_sum = 0.0
    for i in range(nk_a):
        density = conj(psi_a[i])*psi0
        prop_t = cimag(density)*H_hop[i]*densitydiag
        if prop_t>0:
            prop[i] = prop_t
            prop_sum += prop_t
        else: prop[i] = 0.0

    if LHOLE:
        prop_c = 0.0
        for i in range(nk_a):
            prop_t = prop[i]
            if prop_t > 0:
                dEt = dE[i]
                if dEt<0: prop_t *= exp(dEt*KbT1)
#                prop_t /= prop_sum
                for j in range(BANDDEG):
                    prop_a[i*BANDDEG+j] = prop_c+prop_t/BANDDEG*j
                prop_c += prop_t
            else:
                for j in range(BANDDEG):
                    prop_a[i*BANDDEG+j] = prop_c
        prop_a[nk_a*BANDDEG] = prop_c
    else:
        prop_c = 0.0
        for i in range(nk_a):
            prop_t = prop[i]
            if prop_t > 0:
                dEt = dE[i]
                if dEt>0: prop_t *= exp(-dEt*KbT1)
#                prop_t /= prop_sum
                for j in range(BANDDEG):
                    prop_a[i*BANDDEG+j] = prop_c+prop_t/BANDDEG*j
                prop_c += prop_t
            else:
                for j in range(BANDDEG):
                    prop_a[i*BANDDEG+j] = prop_c
        prop_a[nk_a*BANDDEG] = prop_c

    if (prop0<=prop_a[nk_a*BANDDEG]):
        idx_s = 0
        idx_e = nk_a*BANDDEG
        while 1:
            if ((idx_e-idx_s)<=1):
                state_n = idx_s
                break
            state_n = (idx_s+idx_e)/2
            if (prop0<prop_a[state_n]):
                idx_e = state_n
            else:
                idx_s = state_n

    if (state_n != state):
        if (n_occ[state_n]>0):
            state_n = state
        else:
            state_t = <int>(state_n/BANDDEG)
            qidx = kqidx0[state_t]
            dEt = dE[state_t]
            ph_pop_sum = 0
            for i in range(nmodes):
                dE0 = dEt+phonon[i,qidx]+1e-8
                dE1 = dEt-phonon[i,qidx]-1e-8
                expdE0 = exp(dE0*dE0*f0)
                expdE1 = exp(dE1*dE1*f0)
                epc_p0 = epc_p[i*mstride+state_t]
                epc_p1 = (creal(epc_p0)*creal(epc_p0) \
                         +cimag(epc_p0)*cimag(epc_p0))*BE2[i,qidx]
                ph_pop0 = epc_p1*expdE0
                ph_pop1 = epc_p1*expdE1
                ph_pop_t[i] = ph_pop0 - ph_pop1
                ph_pop_sum += ph_pop0 + ph_pop1
            if ph_pop_sum>0:
                for i in range(nmodes):
                    ph_pop_p[i,qidx] += <float>(ph_pop_t[i]/ph_pop_sum)
            n_occ[state] = 0
            n_occ[state_n] = 1

    return state_n


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fssh_sh(
    mpi.MPI_Comm c_comm, mpi.MPI_Comm shm_comm, int myid, int nprocs,
    int shm_id, int shm_nprocs, str namddir, int sample, int NTRAJ, int NSW,
    int nk_proc, int nk_a, int nq, int n_p, int nmodes, int nel, double dt,
    double KbT, double hbar, double sigma, int[::1] state_s, int[::1] k_proc,
    int[:,::1] kqidx_p, double[::1] energy, double[:,::1] phonon,
    double[:,::1] BEfactor, double complex[:,:,::1] epc_a,
    float[:,::1] pop_sh, float[:,:,::1] ph_pop, int BANDDEG, bint LHOLE
):
    cdef double KbT1 = 1.0/KbT
    cdef double f0 = -0.5/(sigma*sigma)
    cdef double f1 = sqrt(2*M_PI)/sigma/n_p
    cdef double f2 = <double>(2.0*dt)/hbar
    cdef double E0

    cdef int i, j, k, l, m, n, p, state, k_state_p, state_t, \
             t0, t1, it, send_s, send_e, nsend, nsend_set, \
             ierr, nnode, node_id, inode, iproc, iel, recv_id, n_s_req, n_s_req_b, \
             nfail, nfail_tot, s_own

    cdef int * nodelist
    cdef int * idx_all
    cdef int * sendset
    cdef double complex * psi
    cdef double[:,::1] BE2 = np.zeros((nmodes,nq),dtype=np.float64)
    cdef double[::1] H_hop = np.zeros((nk_a),dtype=np.float64)
    cdef float[::1] pop_sh_p = np.zeros((nk_a),dtype=np.float32)

    cdef float[:,::1] ph_pop_p = np.zeros((nmodes,nq),dtype=np.float32)
    cdef int[::1] occ_m = np.zeros((nel),dtype=np.int32)
    cdef int[::1] state_num = np.zeros((nk_a),dtype=np.int32)
    cdef int[::1] state_idx = np.zeros((nel),dtype=np.int32)

    cdef int itraj_max, ntraj_max
    cdef int[:,::1] state_a, n_occ

    cdef int s_int = sizeof(int)
    cdef int s_float = sizeof(float)
    cdef int s_double = sizeof(double)
    cdef int s_dcplx = sizeof(double complex)
    cdef int s_req = sizeof(mpi.MPI_Request)
    cdef int s_void = 1

    cdef int epclen = nmodes*nk_a
    cdef long recvlen = epclen*s_dcplx+nk_a*s_int
    cdef long nel_l = nel
    cdef long nk_a_l = nk_a
    cdef char ** sendbuf
    cdef double complex *epc_buf
    cdef int * kqidx_buf
    cdef char * recvbuf
    cdef double complex * epc_p
    cdef int * kqidx
    cdef int * recvtag

    cdef double *ph_pop_t = <double*>malloc(nmodes*s_double)
    cdef double *prop = <double*>malloc(nk_a*s_double)
    cdef double *prop_a = <double*>malloc((nk_a*BANDDEG+1)*s_double)
    cdef double *dE = <double*>malloc(nk_a*s_double)
    cdef double starttime, endtime

    cdef mpi.MPI_Comm node_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Win win0, win1, win2, win3
    cdef mpi.MPI_Aint l_idx_all, l_recvbuf, l_recvtag, l_psi, l_recvbuf_lo, len_own
    cdef char * own_qp
    cdef long l_recvbuf_total
    cdef mpi.MPI_Request * send_req
    cdef mpi.MPI_Request * send_req_b
    cdef mpi.MPI_Request recv_req
    cdef char data_name[500]
    cdef FILE * fp
    namddir_t = namddir.encode("utf-8")
    cdef char* namddir_c = namddir_t

    # split ntraj
    if NTRAJ % nprocs == 0:
        itraj_max = <int>(NTRAJ/nprocs)
    else:
        itraj_max = <int>(NTRAJ/nprocs)+1
    ntraj_max = itraj_max*nprocs
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&node_comm)

    # alloc shared variables
    if (shm_id==0):
        l_idx_all = nprocs*4*s_int
        l_recvtag = nprocs*s_int*2
    else:
        l_idx_all = 0
        l_recvtag = 0
    l_recvbuf_total = shm_nprocs*recvlen
    l_recvbuf_lo = l_recvbuf_total*shm_id/shm_nprocs
    l_recvbuf = l_recvbuf_total*(shm_id+1)/shm_nprocs-l_recvbuf_lo
    l_psi = nel_l*(nk_a*(shm_id+1)/shm_nprocs-nk_a*shm_id/shm_nprocs)*s_dcplx
    mpi.MPI_Win_allocate_shared(
        l_idx_all,s_int,mpi.MPI_INFO_NULL,shm_comm,&idx_all,&win0
    )
    mpi.MPI_Win_allocate_shared(
        l_recvbuf,s_void,mpi.MPI_INFO_NULL,shm_comm,&recvbuf,&win1
    )
    mpi.MPI_Win_allocate_shared(
        l_recvtag,s_int,mpi.MPI_INFO_NULL,shm_comm,&recvtag,&win2
    )
    mpi.MPI_Win_allocate_shared(
        l_psi,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&psi,&win3
    )
    if (shm_id!=0):
        mpi.MPI_Win_shared_query(win0,0,&l_idx_all,&s_int,&idx_all)
        mpi.MPI_Win_shared_query(win1,0,&l_recvbuf,&s_void,&recvbuf)
        mpi.MPI_Win_shared_query(win2,0,&l_recvtag,&s_int,&recvtag)
        mpi.MPI_Win_shared_query(win3,0,&l_psi,&s_dcplx,&psi)
    nfail = 0
    mpi.MPI_Win_shared_query(win1,shm_id,&len_own,&s_own,&own_qp)
    if (own_qp-recvbuf)!=l_recvbuf_lo: nfail += 1
    mpi.MPI_Win_shared_query(win3,shm_id,&len_own,&s_own,&own_qp)
    if (<char*>own_qp-<char*>psi)!=(nel_l*(nk_a*shm_id/shm_nprocs)*s_dcplx): nfail += 1
    mpi.MPI_Allreduce(&nfail,&nfail_tot,1,mpi.MPI_INT,mpi.MPI_SUM,shm_comm)
    if nfail_tot>0:
        if myid==0: printf("shm contiguity FAIL nfail=%d\n",nfail_tot)
        mpi.MPI_Abort(shm_comm,1)
    else:
        if myid==0: printf("shm contiguity OK\n")
    memset(recvbuf+l_recvbuf_lo,0,l_recvbuf_total*(shm_id+1)/shm_nprocs-l_recvbuf_lo)
    memset(psi+nel_l*(nk_a*shm_id/shm_nprocs),0,\
        nel_l*(nk_a*(shm_id+1)/shm_nprocs-nk_a*shm_id/shm_nprocs)*s_dcplx)
    # init BEfactor
    for i in range(nmodes):
        for j in range(nq):
            BE2[i,j] = BEfactor[i,j]*BEfactor[i,j]*f1
    # init INICON
    state_a = np.zeros((itraj_max,nel),dtype=np.int32)
    n_occ = np.zeros((itraj_max,nk_a*BANDDEG),dtype=np.int32)
    for iel in range(nel):
        state_idx[iel] = state_num[state_s[iel]]
        state_num[state_s[iel]] += 1
    for k in range(itraj_max):
        for iel in range(nel):
            state_a[k,iel] = state_s[iel]*BANDDEG+state_idx[iel]
            n_occ[k,state_a[k,iel]] = 1
    mpi.MPI_Barrier(c_comm)

    if shm_id==0:
        sprintf(data_name,"%s/psi-%d.dat",namddir_c,sample)
        fp = fopen(data_name,"rb")
        fseek(fp,0,SEEK_SET)
    mpi.MPI_Barrier(shm_comm)

    for it in range(NSW):
        starttime = mpi.MPI_Wtime()
        if shm_id==0:
            fread(psi,16,nel_l*nk_a,fp)
        mpi.MPI_Barrier(shm_comm)
        memset(&pop_sh_p[0],0,nk_a*s_float)
        memset(&ph_pop_p[0,0],0,nmodes*nq*s_float)
        for i in range(itraj_max):
            for iel in range(nel):
                # get myid recv state belongs to, get send_id
                # idx_all has 4 tags
                # 0: state (send_state)
                # 1: state-k_proc[state_id] (send_id_loc)
                # 2: state_id (send_id)
                # 3: myid (recv_id)
                mpi.MPI_Barrier(shm_comm)
                GetState(
                    myid,nprocs,shm_nprocs,
                    <int>(state_a[i,iel]/BANDDEG),idx_all,k_proc
                )
                mpi.MPI_Barrier(shm_comm)
                if (nnode>1):
                    if (shm_id==0):
                        mpi.MPI_Allgather(
                            mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,
                            idx_all,shm_nprocs*4,mpi.MPI_INT,node_comm
                        )
                if (shm_id==0):
                    # sort by send_state
                    qsort(idx_all,nprocs,s_int*4,&Cmp_s)
                    memset(recvtag,0,nprocs*s_int*2)
                mpi.MPI_Barrier(c_comm)
                # get myid send range
                send_s = -1
                for j in range(nprocs):
                    if (idx_all[j*4+2]==myid):
                        send_s = j
                        break
                if (send_s<0):
                    send_s = 0
                    send_e = 0
                elif (idx_all[send_s*4+2]==idx_all[(nprocs-1)*4+2]):
                    send_e = nprocs
                else:
                    for j in range(send_s,nprocs):
                        if (idx_all[j*4+2]>myid):
                            send_e = j
                            break
                nsend = send_e-send_s
                # get nsend set, unique by send_state
                # send_id:   | 0 | 1 |3|6 |
                # send_state:|011|223|4|55|
                # recv_id:   |421|053|8|76|
                if nsend > 0:
                    sendset = <int*>calloc(nsend*2,s_int)
                    nsend_set = 1
                    k = idx_all[send_s*4]
                    sendset[(nsend_set-1)*2] = 1
                    for j in range(send_s+1,send_e):
                        if (idx_all[j*4]==k):
                            sendset[(nsend_set-1)*2] += 1
                        else:
                            sendset[nsend_set*2] = 1
                            k = idx_all[j*4]
                            nsend_set += 1
                    for j in range(nsend_set):
                        for k in range(j+1,nsend_set):
                            sendset[k*2+1] += sendset[j*2]
                    # sort recv_id in each send_state 
                    # id in the same node with same send_state 
                    # forms [] group, example below
                    # send_id:    |   0    |    1    | 3 |  6  |
                    # send_shm_id:|   0    |    0    | 0 |  1  |
                    # send_state: | 0   1  |  2    3 | 4 |  5  |
                    # recv_id:    | 4 {1 2}|{0  5} 3 | 8 |{6 7}|
                    # recv_shm_id:|[0][0 0]|[0][1][0]|[1]|[1 1]|
                    #             ------------------------------
                    # id:         | 0  1  2  3  4  5  6  7  8 |
                    # recv_tag:   | 3  3  1  3  3  2  3  1  2 |
                    # recv_id in the same send_id has 3 type of tag
                    # recvtag=3: recv_id in the same node with send_state
                    #            recv_id is the first id in the group
                    # recvtag=2: recv_id in different node with send_state
                    #            recv_id is the first id in the group
                    # recvtag=1: recv_id not the first id in the group
                    for j in range(nsend_set):
                        l = send_s+sendset[j*2+1]
                        qsort(&idx_all[l*4],sendset[j*2],s_int*4,&Cmp_e)
                        m = -1; n = -1
                        for k in range(sendset[j*2]):
                            l = k+sendset[j*2+1]+send_s
                            inode = <int>(idx_all[l*4+3]/shm_nprocs)
                            iproc = idx_all[l*4+3]
                            if m<inode:
                                m = inode; n = iproc
                                if inode==node_id:
                                    recvtag[iproc*2] = 3
                                else:
                                    recvtag[iproc*2] = 2
                                # for recvtag>=2, data send to recv_id
                                recvtag[iproc*2+1] = iproc%shm_nprocs
                            else:
                                recvtag[iproc*2] = 1
                                # for recvtag==1, data send to
                                # the id of first group member
                                recvtag[iproc*2+1] = n%shm_nprocs
                mpi.MPI_Barrier(shm_comm)
                # reduce shm part of recvtag to shm_id==0
                if (nnode>1):
                    for j in range(nnode):
                        if (shm_id==0):
                            if (node_id==j):
                                mpi.MPI_Reduce(
                                    mpi.MPI_IN_PLACE,
                                    &recvtag[shm_nprocs*2*j],shm_nprocs*2,
                                    mpi.MPI_INT,mpi.MPI_MAX,j,node_comm
                                )
                            else:
                                mpi.MPI_Reduce(
                                    &recvtag[shm_nprocs*2*j],
                                    &recvtag[shm_nprocs*2*j],shm_nprocs*2,
                                    mpi.MPI_INT,mpi.MPI_MAX,j,node_comm
                                )
                    mpi.MPI_Barrier(shm_comm)
                # prepare irecv for recvtag>=2
                if (recvtag[myid*2]==2):
                    mpi.MPI_Irecv(
                        &recvbuf[shm_id*recvlen],recvlen,mpi.MPI_BYTE,
                        mpi.MPI_ANY_SOURCE,myid,c_comm,&recv_req
                    )
                elif (recvtag[myid*2]==3):
                    mpi.MPI_Irecv(
                        &recv_id,0,mpi.MPI_INT,
                        mpi.MPI_ANY_SOURCE,myid,c_comm,&recv_req
                    )
                # prepare send
                if (nsend>0):
                    send_req = <mpi.MPI_Request*>malloc(nsend*s_req)
                    sendbuf = <char**>malloc(sizeof(char*)*nsend_set)
                    n_s_req = 0
                    for j in range(nsend_set):
                        sendbuf[j] = <char*>malloc(recvlen)
                        epc_buf = <double complex*>(sendbuf[j])
                        kqidx_buf = <int*>(epc_buf+epclen)
                        k_state_p = idx_all[(send_s+sendset[j*2+1])*4+1]
                        for t0 in range(nmodes):
                            for t1 in range(nk_a):
                                epc_buf[t0*nk_a+t1]=epc_a[t0,k_state_p,t1]
                        memcpy(kqidx_buf,&kqidx_p[k_state_p,0],nk_a*s_int)

                        for k in range(sendset[j*2]):
                            l = k+sendset[j*2+1]+send_s
                            iproc = idx_all[l*4+3]
                            # for recvtag==3: 
                            # data memcpy to shm_buffer[recv_shm_id]
                            # send sync tag to all group member
                            if (recvtag[iproc*2]==3):
                                m = iproc%shm_nprocs
                                memcpy(recvbuf+m*recvlen,sendbuf[j],recvlen)
                                n = (node_id+1)*shm_nprocs
                                for p in range(iproc,n):
                                    if (recvtag[p*2+1]==m):
                                        mpi.MPI_Isend(
                                            &p,0,mpi.MPI_INT,p,
                                            p,c_comm,&send_req[n_s_req]
                                        )
                                        n_s_req += 1
                            # for recvtag==2: data isend to recv_id
                            elif (recvtag[iproc*2]==2):
                                mpi.MPI_Isend(
                                    sendbuf[j],recvlen,mpi.MPI_BYTE,
                                    iproc,iproc,c_comm,&send_req[n_s_req]
                                )
                                n_s_req += 1

                # for recvtag>=2: wait irecv to confirm recv
                if (recvtag[myid*2]==3):
                    mpi.MPI_Wait(&recv_req,mpi.MPI_STATUSES_IGNORE)
                # for recvtag==2: after confirming recv
                # send sync tag to recvtag==1 in the same group
                elif (recvtag[myid*2]==2):
                    mpi.MPI_Wait(&recv_req,mpi.MPI_STATUSES_IGNORE)
                    n_s_req_b = 0
                    n = (node_id+1)*shm_nprocs
                    for j in range(myid+1,n):
                        if (recvtag[j*2+1]==shm_id):
                            n_s_req_b += 1
                    send_req_b = <mpi.MPI_Request*>malloc(n_s_req_b*s_req)
                    k = 0
                    for j in range(myid+1,n):
                        if (recvtag[j*2+1]==shm_id):
                            mpi.MPI_Isend(
                                &j,0,mpi.MPI_INT,j,j,c_comm,&send_req_b[k]
                            )
                            k += 1
                else:
                    # for recvtag==1: prepare recv sync tag
                    mpi.MPI_Recv(
                        &recv_id,0,mpi.MPI_INT,mpi.MPI_ANY_SOURCE,
                        myid,c_comm,mpi.MPI_STATUSES_IGNORE
                    )
                # calculate H_hop from (epc_p,kqidx)
                recv_id = recvtag[myid*2+1]
                epc_p = <double complex*>(recvbuf+recv_id*recvlen)
                kqidx = <int*>(epc_p+epclen)
                E0 = energy[<int>(state_a[i,iel]/BANDDEG)]
                for j in range(nk_a):
                    H_hop[j] = 0
                    dE[j] = energy[j]-E0
                memset(&prop_a[1],0,nk_a*BANDDEG*s_double)
                GetHhop(
                    nmodes,nk_a,nk_a,kqidx,
                    phonon,BE2,epc_p,f0,H_hop,dE
                )
                state_a[i,iel] = fsshhop(
                    it,nmodes,nk_a,nk_a,&psi[iel*nk_a_l],H_hop,dE,epc_p,
                    kqidx,energy,phonon,BE2,state_a[i,iel],f0,f2,KbT1,
                    prop,prop_a,ph_pop_t,ph_pop_p,n_occ[i],LHOLE,BANDDEG
                )
                # waitall send and send_sync & free
                if (nsend>0):
                    mpi.MPI_Waitall(n_s_req,send_req,mpi.MPI_STATUSES_IGNORE)
                    free(send_req)
                    for t0 in range(nsend_set):
                        free(sendbuf[t0])
                    free(sendbuf)
                    free(sendset)
                if (recvtag[myid*2]==2):
                    mpi.MPI_Waitall(n_s_req_b,send_req_b,mpi.MPI_STATUSES_IGNORE)
                    free(send_req_b)

                pop_sh_p[<int>(state_a[i,iel]/BANDDEG)] += 1

        mpi.MPI_Reduce(
            &pop_sh_p[0],&pop_sh[it,0],nk_a,
            mpi.MPI_FLOAT,mpi.MPI_SUM,0,c_comm
        )
        mpi.MPI_Reduce(
            &ph_pop_p[0,0],&ph_pop[it,0,0],nmodes*nq,
            mpi.MPI_FLOAT,mpi.MPI_SUM,0,c_comm
        )
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("FSSH in %d fs: %.5fs.\n",it,endtime-starttime)

    if shm_id==0: fclose(fp)

    free(ph_pop_t)
    free(prop)
    free(prop_a)
    free(dE)

    # free
    free(nodelist)
    if (node_comm != mpi.MPI_COMM_NULL):
        mpi.MPI_Comm_free(&node_comm)
    mpi.MPI_Group_free(&shm_gp)
    mpi.MPI_Group_free(&split_gp)
    mpi.MPI_Win_free(&win0)
    mpi.MPI_Win_free(&win1)
    mpi.MPI_Win_free(&win2)
    mpi.MPI_Win_free(&win3)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int GetOwner(int state_sel, int nprocs, int[::1] k_proc):
    cdef int k = 0
    cdef int l = nprocs
    cdef int p
    while l-k>1:
        p = <int>((k+l)/2)
        if state_sel<k_proc[p]:
            l = p
        else:
            k = p
    if state_sel<k_proc[k]:
        k = l
    return k


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void CacheFill(
    mpi.MPI_Comm c_comm, mpi.MPI_Comm shm_comm, mpi.MPI_Comm node_comm,
    int myid, int nprocs, int shm_id, int shm_nprocs, int node_id, int nnode,
    int fill_id, int nk_a, int nmodes, int n_slot,
    int BANDDEG, long recvlen,
    int[::1] state_a, int[::1] iel_arr, int n_iel,
    int[::1] k_proc,
    double complex[:,:,::1] epc_a, int[:,::1] kqidx_p,
    char * recvbuf, int * slot_of_state, int * cache_need, int * fill_tag,
    int * state_of_slot, int * lru, char * cmap
):
    # gather requested rows on the node, using free slots before LRU eviction
    cdef int i, j, k, l, m, n_own, n_r_req, n_s_req, k0, \
             state_t, t0, t1, n_free, n_cl, n_cr
    cdef int epclen = nmodes*nk_a
    cdef int s_int = sizeof(int)
    cdef int s_long = sizeof(long)
    cdef int s_req = sizeof(mpi.MPI_Request)
    cdef char ** sendbuf
    cdef double complex * epc_buf
    cdef int * kqidx_buf
    cdef int * free_s
    cdef long * cand_l
    cdef long * cand_r
    cdef mpi.MPI_Request * send_req
    cdef mpi.MPI_Request * recv_req

    for j in range(n_iel):
        cache_need[<int>(state_a[iel_arr[j]]/BANDDEG)] = fill_id
    mpi.MPI_Barrier(shm_comm)
    if shm_id==0:
        for j in range(nk_a):
            cmap[node_id*nk_a+j] = 0
            if cache_need[j]==fill_id:
                if slot_of_state[j]>=0:
                    lru[slot_of_state[j]] = fill_id
                else:
                    cmap[node_id*nk_a+j] = 1
        free_s = <int*>malloc(n_slot*s_int)
        cand_l = <long*>malloc(n_slot*s_long)
        cand_r = <long*>malloc(n_slot*s_long)
        n_free = 0
        n_cl = 0
        n_cr = 0
        for j in range(n_slot):
            k = state_of_slot[j]
            if k<0:
                free_s[n_free] = j
                n_free += 1
            elif cache_need[k]!=fill_id:
                if <int>(GetOwner(k,nprocs,k_proc)/shm_nprocs)==node_id:
                    cand_l[n_cl] = lru[j]*<long>n_slot+j
                    n_cl += 1
                else:
                    cand_r[n_cr] = lru[j]*<long>n_slot+j
                    n_cr += 1
        qsort(cand_l,n_cl,s_long,&Cmp_l)
        qsort(cand_r,n_cr,s_long,&Cmp_l)
        # prefer identity slots, then free slots, local rows, remote rows
        i = 0
        l = 0
        for j in range(nk_a):
            if cmap[node_id*nk_a+j]:
                k = -1
                if j<n_slot and state_of_slot[j]<0:
                    k = j
                else:
                    while n_free>0:
                        n_free -= 1
                        if state_of_slot[free_s[n_free]]<0:
                            k = free_s[n_free]
                            break
                    if k<0 and i<n_cl:
                        k = <int>(cand_l[i]%n_slot)
                        i += 1
                    if k<0 and l<n_cr:
                        k = <int>(cand_r[l]%n_slot)
                        l += 1
                if k>=0:
                    if state_of_slot[k]>=0:
                        slot_of_state[state_of_slot[k]] = -1
                    state_of_slot[k] = j
                    slot_of_state[j] = k
                    lru[k] = fill_id
                    fill_tag[j] = fill_id
                else:
                    cmap[node_id*nk_a+j] = 0
        free(free_s)
        free(cand_l)
        free(cand_r)
        mpi.MPI_Allgather(
            mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,
            cmap,nk_a,mpi.MPI_CHAR,node_comm
        )
    mpi.MPI_Barrier(shm_comm)
    # remote receives are strided over node ranks; local owners write directly
    recv_req = <mpi.MPI_Request*>malloc((<int>(nk_a/shm_nprocs)+1)*s_req)
    n_r_req = 0
    for j in range(shm_id,nk_a,shm_nprocs):
        if fill_tag[j]==fill_id:
            k = GetOwner(j,nprocs,k_proc)
            if <int>(k/shm_nprocs)!=node_id:
                mpi.MPI_Irecv(
                    recvbuf+slot_of_state[j]*recvlen,recvlen,mpi.MPI_BYTE,
                    k,j,c_comm,&recv_req[n_r_req]
                )
                n_r_req += 1
    k0 = k_proc[myid]
    n_own = k_proc[myid+1]-k0
    send_req = <mpi.MPI_Request*>malloc((n_own*nnode+1)*s_req)
    sendbuf = <char**>malloc(sizeof(char*)*(n_own+1))
    n_s_req = 0
    for j in range(n_own):
        sendbuf[j] = NULL
        epc_buf = NULL
        for m in range(nnode):
            if cmap[m*nk_a+k0+j] and m==node_id:
                epc_buf = <double complex*>(recvbuf+slot_of_state[k0+j]*recvlen)
                break
        if epc_buf==NULL:
            for m in range(nnode):
                if cmap[m*nk_a+k0+j]:
                    sendbuf[j] = <char*>malloc(recvlen)
                    epc_buf = <double complex*>(sendbuf[j])
                    break
        if epc_buf==NULL:
            continue
        kqidx_buf = <int*>(epc_buf+epclen)
        for t0 in range(nmodes):
            for t1 in range(nk_a):
                epc_buf[t0*nk_a+t1]=epc_a[t0,j,t1]
        memcpy(kqidx_buf,&kqidx_p[j,0],nk_a*s_int)
        for m in range(nnode):
            if cmap[m*nk_a+k0+j] and m!=node_id:
                mpi.MPI_Isend(
                    epc_buf,recvlen,mpi.MPI_BYTE,
                    m*shm_nprocs+(k0+j)%shm_nprocs,k0+j,
                    c_comm,&send_req[n_s_req]
                )
                n_s_req += 1
    mpi.MPI_Waitall(n_r_req,recv_req,mpi.MPI_STATUSES_IGNORE)
    mpi.MPI_Waitall(n_s_req,send_req,mpi.MPI_STATUSES_IGNORE)
    free(recv_req)
    for j in range(n_own):
        free(sendbuf[j])
    free(sendbuf)
    free(send_req)
    mpi.MPI_Barrier(shm_comm)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fssh_sh_cache(
    mpi.MPI_Comm c_comm, mpi.MPI_Comm shm_comm, int myid, int nprocs,
    int shm_id, int shm_nprocs, str namddir, int sample, int NTRAJ, int NSW,
    int nk_proc, int nk_a, int nq, int n_p, int nmodes, int nel, double dt,
    double KbT, double hbar, double sigma, int[::1] state_s, int[::1] k_proc,
    int[:,::1] kqidx_p, double[::1] energy, double[:,::1] phonon,
    double[:,::1] BEfactor, double complex[:,:,::1] epc_a,
    float[:,::1] pop_sh, float[:,:,::1] ph_pop, int BANDDEG, bint LHOLE,
    bint LPHSHM
):
    cdef double KbT1 = 1.0/KbT
    cdef double f0 = -0.5/(sigma*sigma)
    cdef double f1 = sqrt(2*M_PI)/sigma/n_p
    cdef double f2 = <double>(2.0*dt)/hbar
    cdef double E0

    cdef int i, j, k, it, ierr, node_id, nnode, \
             iel, state_t, slot, n_defer, n_hop, n_defer_all

    cdef double complex * psi
    cdef double[:,::1] BE2 = np.zeros((nmodes,nq),dtype=np.float64)
    cdef double[::1] H_hop = np.zeros((nk_a),dtype=np.float64)
    cdef float[::1] pop_sh_p = np.zeros((nk_a),dtype=np.float32)

    cdef float[:,::1] ph_pop_p = np.zeros((nmodes,nq),dtype=np.float32)
    cdef int[::1] iel_arr = np.arange(nel,dtype=np.int32)
    cdef int[::1] defer_iel = np.zeros((nel),dtype=np.int32)
    cdef int[::1] hop_arr
    cdef int[::1] state_num = np.zeros((nk_a),dtype=np.int32)
    cdef int[::1] state_idx = np.zeros((nel),dtype=np.int32)

    cdef int itraj_max
    cdef int[:,::1] state_a, n_occ

    cdef int s_int = sizeof(int)
    cdef int s_float = sizeof(float)
    cdef int s_double = sizeof(double)
    cdef int s_dcplx = sizeof(double complex)
    cdef int s_void = 1

    cdef int epclen = nmodes*nk_a
    cdef long recvlen = epclen*s_dcplx+nk_a*s_int
    cdef long nel_l = nel
    cdef long nk_a_l = nk_a
    cdef char * recvbuf
    cdef double complex * epc_p
    cdef int * kqidx
    cdef int n_slot
    cdef int * slot_of_state
    cdef int * cache_need
    cdef int * fill_tag
    cdef int * state_of_slot
    cdef int * lru
    cdef char * cmap

    cdef double *ph_pop_t = <double*>malloc(nmodes*s_double)
    cdef double *prop = <double*>malloc(nk_a*s_double)
    cdef double *prop_a = <double*>malloc((nk_a*BANDDEG+1)*s_double)
    cdef double *dE = <double*>malloc(nk_a*s_double)
    cdef double starttime, endtime

    cdef mpi.MPI_Win win1, win2
    cdef mpi.MPI_Comm node_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef int * nodelist
    cdef int nfail, nfail_tot, s_own
    cdef int fill_next = 1
    cdef mpi.MPI_Aint l_recvbuf, l_recvbuf_lo, l_psi, len_own
    cdef char * own_qp
    cdef long shm_fixed
    cdef char data_name[500]
    cdef FILE * fp
    namddir_t = namddir.encode("utf-8")
    cdef char* namddir_c = namddir_t

    if NTRAJ % nprocs == 0:
        itraj_max = <int>(NTRAJ/nprocs)
    else:
        itraj_max = <int>(NTRAJ/nprocs)+1
    node_id = myid/shm_nprocs

    # fixed shared arrays and all cache metadata count against the node budget
    shm_fixed = nel_l*nk_a*s_dcplx+nk_a*s_int
    if LPHSHM:
        shm_fixed += 2*nq*nmodes*s_dcplx+nq*nmodes*s_double
    nnode = nprocs/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&node_comm)
    free(nodelist)
    n_slot = <int>((SHM_MAX_BYTES-shm_fixed-3*nk_a*s_int-nnode*nk_a)/(recvlen+2*s_int))
    if n_slot>=nk_a:
        n_slot = nk_a
    elif n_slot<1:
        if myid==0:
            printf("FSSH cache: no slot fits the shm budget, abort\n")
        mpi.MPI_Abort(c_comm,1)
    elif myid==0:
        printf("FSSH cache: bounded to %d/%d slots/node\n",n_slot,nk_a)
    if myid==0:
        printf("FSSH row cache: %ld MB/node\n",n_slot*recvlen/1048576)
    # owner-slice allocation and first touch spread pages over NUMA domains
    l_recvbuf = n_slot*recvlen+(3*nk_a+2*n_slot)*s_int+nnode*nk_a
    l_recvbuf_lo = l_recvbuf*shm_id/shm_nprocs
    l_recvbuf = l_recvbuf*(shm_id+1)/shm_nprocs-l_recvbuf_lo
    l_psi = nel_l*(nk_a*(shm_id+1)/shm_nprocs-nk_a*shm_id/shm_nprocs)*s_dcplx
    mpi.MPI_Win_allocate_shared(
        l_recvbuf,s_void,mpi.MPI_INFO_NULL,shm_comm,&recvbuf,&win1
    )
    mpi.MPI_Win_allocate_shared(
        l_psi,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&psi,&win2
    )
    if (shm_id!=0):
        mpi.MPI_Win_shared_query(win1,0,&l_recvbuf,&s_void,&recvbuf)
        mpi.MPI_Win_shared_query(win2,0,&l_psi,&s_dcplx,&psi)
    nfail = 0
    mpi.MPI_Win_shared_query(win1,shm_id,&len_own,&s_own,&own_qp)
    if (own_qp-recvbuf)!=l_recvbuf_lo: nfail += 1
    mpi.MPI_Win_shared_query(win2,shm_id,&len_own,&s_own,&own_qp)
    if (<char*>own_qp-<char*>psi)!=(nel_l*(nk_a*shm_id/shm_nprocs)*s_dcplx): nfail += 1
    mpi.MPI_Allreduce(&nfail,&nfail_tot,1,mpi.MPI_INT,mpi.MPI_SUM,shm_comm)
    if (nfail_tot>0):
        if (myid==0): printf("shm contiguity FAIL nfail=%d\n",nfail_tot)
        mpi.MPI_Abort(shm_comm,1)
    else:
        if (myid==0): printf("shm contiguity OK\n")
    memset(recvbuf+l_recvbuf_lo,0,l_recvbuf)
    memset(psi+nel_l*(nk_a*shm_id/shm_nprocs),0,\
        nel_l*(nk_a*(shm_id+1)/shm_nprocs-nk_a*shm_id/shm_nprocs)*s_dcplx)
    mpi.MPI_Barrier(shm_comm)
    # window tail: state maps, round tags, slot states, LRU, per-node miss maps
    slot_of_state = <int*>(recvbuf+n_slot*recvlen)
    cache_need = slot_of_state+nk_a
    fill_tag = cache_need+nk_a
    state_of_slot = fill_tag+nk_a
    lru = state_of_slot+n_slot
    if shm_id==0:
        memset(slot_of_state,0xFF,nk_a*s_int)
        memset(cache_need,0,2*nk_a*s_int)
        memset(state_of_slot,0xFF,n_slot*s_int)
    cmap = <char*>(lru+n_slot)
    for i in range(nmodes):
        for j in range(nq):
            BE2[i,j] = BEfactor[i,j]*BEfactor[i,j]*f1
    state_a = np.zeros((itraj_max,nel),dtype=np.int32)
    n_occ = np.zeros((itraj_max,nk_a*BANDDEG),dtype=np.int32)
    for iel in range(nel):
        state_idx[iel] = state_num[state_s[iel]]
        state_num[state_s[iel]] += 1
    for k in range(itraj_max):
        for iel in range(nel):
            state_a[k,iel] = state_s[iel]*BANDDEG+state_idx[iel]
            n_occ[k,state_a[k,iel]] = 1
    mpi.MPI_Barrier(c_comm)

    if shm_id == 0:
        sprintf(data_name,"%s/psi-%d.dat",namddir_c,sample)
        fp = fopen(data_name,"rb")
        fseek(fp,0,SEEK_SET)

    for it in range(NSW):
        starttime = mpi.MPI_Wtime()
        if shm_id == 0:
            fread(psi,16,nel_l*nk_a,fp)
        mpi.MPI_Barrier(shm_comm)
        memset(&pop_sh_p[0],0,nk_a*s_float)
        memset(&ph_pop_p[0,0],0,nmodes*nq*s_float)
        for i in range(itraj_max):
            CacheFill(
                c_comm,shm_comm,node_comm,myid,nprocs,shm_id,shm_nprocs,
                node_id,nnode,fill_next,nk_a,nmodes,n_slot,BANDDEG,recvlen,
                state_a[i],iel_arr,nel,k_proc,epc_a,kqidx_p,
                recvbuf,slot_of_state,cache_need,fill_tag,state_of_slot,lru,
                cmap
            )
            fill_next += 1
            # fixed electron order preserves the fssh_sh random sequence
            n_hop = nel
            hop_arr = iel_arr
            for iel in range(nel+1):
                n_defer = 0
                for j in range(n_hop):
                    iel = hop_arr[j]
                    state_t = <int>(state_a[i,iel]/BANDDEG)
                    slot = slot_of_state[state_t]
                    if slot<0:
                        defer_iel[n_defer] = iel
                        n_defer += 1
                        continue
                    epc_p = <double complex*>(recvbuf+slot*recvlen)
                    kqidx = <int*>(epc_p+epclen)
                    E0 = energy[state_t]
                    for k in range(nk_a):
                        H_hop[k] = 0
                        dE[k] = energy[k]-E0
                    memset(&prop_a[1],0,nk_a*BANDDEG*s_double)
                    GetHhop(
                        nmodes,nk_a,nk_a,kqidx,
                        phonon,BE2,epc_p,f0,H_hop,dE
                    )
                    state_a[i,iel] = fsshhop(
                        it,nmodes,nk_a,nk_a,&psi[iel*nk_a_l],H_hop,dE,
                        epc_p,kqidx,energy,phonon,BE2,state_a[i,iel],f0,f2,
                        KbT1,prop,prop_a,ph_pop_t,ph_pop_p,n_occ[i],LHOLE,
                        BANDDEG
                    )
                    pop_sh_p[<int>(state_a[i,iel]/BANDDEG)] += 1
                # all nodes enter refill collectives together
                mpi.MPI_Allreduce(
                    &n_defer,&n_defer_all,1,mpi.MPI_INT,mpi.MPI_SUM,c_comm
                )
                if n_defer_all==0:
                    break
                CacheFill(
                    c_comm,shm_comm,node_comm,myid,nprocs,shm_id,shm_nprocs,
                    node_id,nnode,fill_next,nk_a,nmodes,n_slot,BANDDEG,recvlen,
                    state_a[i],defer_iel,n_defer,k_proc,epc_a,kqidx_p,
                    recvbuf,slot_of_state,cache_need,fill_tag,state_of_slot,lru,
                    cmap
                )
                fill_next += 1
                n_hop = n_defer
                hop_arr = defer_iel

        mpi.MPI_Reduce(
            &pop_sh_p[0],&pop_sh[it,0],nk_a,
            mpi.MPI_FLOAT,mpi.MPI_SUM,0,c_comm
        )
        mpi.MPI_Reduce(
            &ph_pop_p[0,0],&ph_pop[it,0,0],nmodes*nq,
            mpi.MPI_FLOAT,mpi.MPI_SUM,0,c_comm
        )
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("FSSH in %d fs: %.5fs.\n",it,endtime-starttime)

    if shm_id == 0: fclose(fp)

    free(ph_pop_t)
    free(prop)
    free(prop_a)
    free(dE)
    if (node_comm != mpi.MPI_COMM_NULL):
        mpi.MPI_Comm_free(&node_comm)
    mpi.MPI_Group_free(&shm_gp)
    mpi.MPI_Group_free(&split_gp)
    mpi.MPI_Win_free(&win1)
    mpi.MPI_Win_free(&win2)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fssh_sh_shm(
    mpi.MPI_Comm c_comm, mpi.MPI_Comm shm_comm, int myid, int nprocs,
    int shm_id, int shm_nprocs, str namddir, int sample, int NTRAJ, int NSW,
    int nk_proc, int nk_a, int nq, int n_p, int nmodes, int nel, double dt,
    double KbT, double hbar, double sigma, int[::1] state_s, int[::1] k_proc,
    int[:,::1] kqidx_p, double[::1] energy, double[:,::1] phonon,
    double[:,::1] BEfactor, double complex[:,:,::1] epc_a,
    float[:,::1] pop_sh, float[:,:,::1] ph_pop, int BANDDEG, bint LHOLE
):
    cdef double KbT1 = 1.0/KbT
    cdef double f0 = -0.5/(sigma*sigma)
    cdef double f1 = sqrt(2*M_PI)/sigma/n_p
    cdef double f2 = <double>(2.0*dt)/hbar
    cdef double E0

    cdef int i, j, k, l, it, ierr, nnode, node_id, iel, state_t, \
             nfail, nfail_tot, s_own
    cdef int * nodelist
    cdef double complex * psi

    cdef double[:,::1] BE2 = np.zeros((nmodes,nq),dtype=np.float64)
    cdef double[::1] H_hop = np.zeros((nk_a),dtype=np.float64)
    cdef float[::1] pop_sh_p = np.zeros((nk_a),dtype=np.float32)

    cdef float[:,::1] ph_pop_p = np.zeros((nmodes,nq),dtype=np.float32)
    cdef int[::1] occ_m = np.zeros((nel),dtype=np.int32)
    cdef int[::1] state_num = np.zeros((nk_a),dtype=np.int32)
    cdef int[::1] state_idx = np.zeros((nel),dtype=np.int32)

    cdef int itraj_max, ntraj_max
    cdef int[:,::1] state_a, n_occ

    cdef int s_int = sizeof(int)
    cdef int s_float = sizeof(float)
    cdef int s_double = sizeof(double)
    cdef int s_dcplx = sizeof(double complex)
    cdef long nel_l = nel
    cdef long nk_a_l = nk_a

    cdef int *kqidx_buf

    cdef double *ph_pop_t = <double*>malloc(nmodes*s_double)
    cdef double *prop = <double*>malloc(nk_a*s_double)
    cdef double *prop_a = <double*>malloc((nk_a*BANDDEG+1)*s_double)
    cdef double *dE = <double*>malloc(nk_a*s_double)
    cdef double starttime, endtime

    cdef mpi.MPI_Comm node_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Win win
    cdef mpi.MPI_Aint len_psi, len_own
    cdef char * own_qp
    cdef char data_name[500]
    cdef FILE * fp
    namddir_t = namddir.encode("utf-8")
    cdef char* namddir_c = namddir_t

    # split ntraj
    if NTRAJ % nprocs == 0:
        itraj_max = <int>(NTRAJ/nprocs)
    else:
        itraj_max = <int>(NTRAJ/nprocs)+1
    ntraj_max = itraj_max*nprocs
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&node_comm)
    # init BEfactor
    for i in range(nmodes):
        for j in range(nq):
            BE2[i,j] = BEfactor[i,j]*BEfactor[i,j]*f1
    # init INICON
    state_a = np.zeros((itraj_max,nel),dtype=np.int32)
    n_occ = np.zeros((itraj_max,nk_a*BANDDEG),dtype=np.int32)
    for iel in range(nel):
        state_idx[iel] = state_num[state_s[iel]]
        state_num[state_s[iel]] += 1
    for k in range(itraj_max):
        for iel in range(nel):
            state_a[k,iel] = state_s[iel]*BANDDEG+state_idx[iel]
            n_occ[k,state_a[k,iel]] = 1
    mpi.MPI_Barrier(c_comm)

    len_psi = nel_l*(nk_a*(shm_id+1)/shm_nprocs-nk_a*shm_id/shm_nprocs)*s_dcplx
    mpi.MPI_Win_allocate_shared(
        len_psi,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&psi,&win
    )
    if shm_id==0:
        sprintf(data_name,"%s/psi-%d.dat",namddir_c,sample)
        fp = fopen(data_name,"rb")
        fseek(fp,0,SEEK_SET)
    else:
        mpi.MPI_Win_shared_query(win,0,&len_psi,&s_dcplx,&psi)
    nfail = 0
    mpi.MPI_Win_shared_query(win,shm_id,&len_own,&s_own,&own_qp)
    if (<char*>own_qp-<char*>psi)!=(nel_l*(nk_a*shm_id/shm_nprocs)*s_dcplx): nfail += 1
    mpi.MPI_Allreduce(&nfail,&nfail_tot,1,mpi.MPI_INT,mpi.MPI_SUM,shm_comm)
    if nfail_tot>0:
        if myid==0: printf("shm contiguity FAIL nfail=%d\n",nfail_tot)
        mpi.MPI_Abort(shm_comm,1)
    else:
        if myid==0: printf("shm contiguity OK\n")
    memset(psi+nel_l*(nk_a*shm_id/shm_nprocs),0,\
        nel_l*(nk_a*(shm_id+1)/shm_nprocs-nk_a*shm_id/shm_nprocs)*s_dcplx)
    mpi.MPI_Barrier(shm_comm)

    for it in range(NSW):
        starttime = mpi.MPI_Wtime()
        if shm_id==0:
            fread(psi,16,nel_l*nk_a,fp)
        mpi.MPI_Barrier(shm_comm)
        memset(&pop_sh_p[0],0,nk_a*s_float)
        memset(&ph_pop_p[0,0],0,nmodes*nq*s_float)
        for i in range(itraj_max):
            for iel in range(nel):
                # calculate H_hop from (epc_a,kqidx)
                state_t = <int>(state_a[i,iel]/BANDDEG)
                E0 = energy[state_t]
                for j in range(nk_a):
                    H_hop[j] = 0
                    dE[j] = energy[j]-E0
                memset(&prop_a[1],0,nk_a*BANDDEG*s_double)
                kqidx_buf = &kqidx_p[state_t,0]
                GetHhop(
                    nmodes,nk_a,nk_a_l*nk_a,kqidx_buf,
                    phonon,BE2,&epc_a[0,state_t,0],f0,H_hop,dE
                )
                state_a[i,iel] = fsshhop(
                    it,nmodes,nk_a,nk_a_l*nk_a,&psi[iel*nk_a_l],H_hop,dE,
                    &epc_a[0,state_t,0],kqidx_buf,energy,phonon,BE2,
                    state_a[i,iel],f0,f2,KbT1,prop,prop_a,ph_pop_t,ph_pop_p,
                    n_occ[i],LHOLE,BANDDEG
                )
                pop_sh_p[<int>(state_a[i,iel]/BANDDEG)] += 1

        mpi.MPI_Reduce(
            &pop_sh_p[0],&pop_sh[it,0],nk_a,
            mpi.MPI_FLOAT,mpi.MPI_SUM,0,c_comm
        )
        mpi.MPI_Reduce(
            &ph_pop_p[0,0],&ph_pop[it,0,0],nmodes*nq,
            mpi.MPI_FLOAT,mpi.MPI_SUM,0,c_comm
        )
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("FSSH-SHM in %d fs: %.5fs.\n",it,endtime-starttime)

    if shm_id==0: fclose(fp)
    mpi.MPI_Win_free(&win)

    free(ph_pop_t)
    free(prop)
    free(prop_a)
    free(dE)

    # free
    free(nodelist)
    if (node_comm != mpi.MPI_COMM_NULL):
        mpi.MPI_Comm_free(&node_comm)
    mpi.MPI_Group_free(&shm_gp)
    mpi.MPI_Group_free(&split_gp)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void WriteEPC(
    mpi.MPI_Comm c_comm, int myid, int nprocs, str namddir, int sample, 
    int NSW, double KbT, double sigma, int[::1] k_proc, int[::1] k_proc_num,
    int k_p_m, int nk_a, int nq, int n_p, int nmodes, 
    int[:,::1] kqidx_p, double complex[:,:,::1] epc_a, double[::1] energy, 
    double[:,::1] phonon, double complex[:,::1] phq, double complex[:,::1] phq0
):
    cdef int NSW_epc = 1
    cdef int h, i, j, k, qidx, jj
    cdef int nk_min = k_proc[myid]
    cdef int nk_proc = k_proc_num[myid]
    cdef double ph, E0, dE, dE0, dE1, expdE
    cdef double np2 = 1.0/sqrt(n_p)
    cdef double KbT1 = 1.0/KbT
    cdef double f0 = -0.5/(sigma*sigma)
    cdef double f1 = sqrt(2*M_PI)/sigma
    cdef double dt1 = 1.0/NSW_epc
    cdef double[:,::1] epcec_t = np.zeros((nk_proc,nk_a),dtype=np.float64)
    cdef double[:,::1] epcec_b = np.empty((nk_proc,nk_a),dtype=np.float64)
    cdef double[:,::1] epcph_t = np.zeros((nmodes,nq),dtype=np.float64)
    cdef double[:,::1] epc_buf, epcph
    cdef double complex[:,::1] phq_t = \
        np.zeros((nmodes,nq),dtype=np.complex128)
    cdef double epc_tmp2, H_hop_tmp, H_hop, phq_t1
    cdef double complex epc_phq_tmp, epc_tmp, phq_tmp
    cdef mpi.MPI_Datatype DB_N

    for k in range(nmodes):
        for l in range(nq):
            phq_t[k,l] = phq[k,l]*np2

    for h in range(NSW_epc):
        memset(&epcec_b[0,0],0,nk_proc*nk_a*sizeof(double))
        for i in range(nmodes):
            for j in range(nk_proc):
                E0 = energy[nk_min+j]
                for k in range(nk_a):
                    qidx = kqidx_p[j+k_p_m,k]
                    dE = E0 - energy[k]

                    ph = phonon[i,qidx]+1e-8
                    dE0 = dE + ph
                    dE1 = dE - ph
                    expdE = exp(dE0*dE0*f0)+exp(dE1*dE1*f0)
                    epc_tmp = epc_a[i,j+k_p_m,k]
                    epc_tmp2 = creal(epc_tmp)*creal(epc_tmp) \
                             + cimag(epc_tmp)*cimag(epc_tmp)
                    phq_tmp = phq_t[i,qidx]
                    phq_t1 = creal(phq_tmp)*creal(phq_tmp) \
                           + cimag(phq_tmp)*cimag(phq_tmp)
                    H_hop_tmp = epc_tmp2*expdE*phq_t1
                    epcph_t[i,qidx] += H_hop_tmp
                    epcec_b[j,k] += H_hop_tmp

            for j in range(nq):
                phq_t[i,j] *= phq0[i,j]

        for j in range(nk_proc):
            for k in range(nk_a):
                epcec_t[j,k] += epcec_b[j,k]

    for i in range(nk_proc):
        for j in range(nk_a):
            epcec_t[i,j] *= (dt1*f1)
    for i in range(nmodes):
        for j in range(nq):
            epcph_t[i,j] *= (dt1*f1)

    if myid == 0:
        epc_buf = np.zeros((nk_a,nk_a),dtype=np.float64)
        epcph = np.zeros((nmodes,nq),dtype=np.float64)
    else:
        epc_buf = np.zeros((0,0),dtype=np.float64)
        epcph = np.zeros((0,0),dtype=np.float64)

    mpi.MPI_Type_contiguous(nk_a,mpi.MPI_DOUBLE,&DB_N)
    mpi.MPI_Type_commit(&DB_N)
    mpi.MPI_Gatherv(
        &epcec_t[0,0],k_proc_num[myid],DB_N,&epc_buf[0,0],
        &k_proc_num[0],&k_proc[0],DB_N,0,c_comm
    )
    mpi.MPI_Type_free(&DB_N)
    if myid == 0:
        np.save(namddir+'/epcec-%d.npy'%(sample),epc_buf)
    mpi.MPI_Reduce(
        &epcph_t[0,0],&epcph[0,0],nmodes*nq,
        mpi.MPI_DOUBLE,mpi.MPI_SUM,0,c_comm
    )
    if myid == 0:
        np.save(namddir+'/epcph-%d.npy'%(sample),epcph)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void FsshOut(
    int nmodes, int NSW, int nk_a, int nq, int ntraj, int nel,
    double[::1] energy, double[:,::1] phonon, float[:,::1] pop_sh,
    double[::1] e_sh, float[:,:,::1] ph_pop, double[:,::1] e_ph
):
    cdef int i, j, k

    for i in range(NSW):
        for j in range(nk_a):
            pop_sh[i,j] /= (ntraj*nel)
            e_sh[i] += pop_sh[i,j]*energy[j]
        for j in range(nmodes):
            for k in range(nq):
                ph_pop[i,j,k] = ph_pop[i,j,k]/ntraj+1e-30
                e_ph[i,j] += ph_pop[i,j,k]*phonon[j,k]


@cython.boundscheck(False)
@cython.wraparound(False)
def fssh(
    MPI.Comm comm, str namddir, int sample, int step_s, int[::1] state_s,
    int NTRAJ, int NSW, int NELM, double KbT, double edt, double hbar,
    double sigma, double dt, int[::1] k_proc, int[::1] k_proc_num, 
    int nk_a, int nq, int n_p, int nmodes, int nel, int[:,::1] kqidx_p, 
    double complex[:,:,::1] epc_a, double[::1] energy, 
    double[:,::1] phonon, bint LHOLE, bint LSPLIT, bint LEPCSHM, 
    bint LPHSHM, int BANDDEG
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm shm_comm
    cdef int myid, nprocs, ierr, shm_id, shm_nprocs
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    cdef long len_phq, len_phq0, len_BE
    cdef long nq_l = nq
    cdef int s_double = sizeof(double)
    cdef int s_dcplx = sizeof(double complex)
    cdef int itraj_max, ntraj_max, k_proc_min
    cdef int nk_proc = k_proc_num[myid]
    cdef double[:,::1] BEfactor
    cdef double complex[:,::1] phq0, phq
    cdef double[::1] e_sh
    cdef float[:,::1] pop_sh
    cdef double starttime, endtime, starttime1, endtime1

    # init
    if LEPCSHM: k_proc_min = k_proc[myid]
    else: k_proc_min = 0
    # split ntraj
    if NTRAJ % nprocs == 0:
        itraj_max = <int>(NTRAJ/nprocs)
    else:
        itraj_max = <int>(NTRAJ/nprocs)+1
    ntraj_max = itraj_max*nprocs
    # get shm_comm
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    # alloc phq0, phq, BEfactor
    shm_comm_py = comm.Split_type(MPI.COMM_TYPE_SHARED)
    if LPHSHM:
        n_phq = nq*nmodes*16
        n_BE = nq*nmodes*8
        len_phq0 = n_phq*(shm_id+1)//shm_nprocs-n_phq*shm_id//shm_nprocs
        len_phq = len_phq0
        len_BE = n_BE*(shm_id+1)//shm_nprocs-n_BE*shm_id//shm_nprocs
        win1 = MPI.Win.Allocate_shared(len_phq0,s_dcplx,comm=shm_comm_py)
        buf1,s_dcplx = win1.Shared_query(0)
        base1 = np.ndarray(buffer=buf1,dtype=np.int8,shape=(1,)).ctypes.data
        phq0 = np.ndarray(buffer=(ctypes.c_char*n_phq).from_address(base1),dtype=np.complex128,shape=(nmodes,nq))
        win2 = MPI.Win.Allocate_shared(len_phq,s_dcplx,comm=shm_comm_py)
        buf2,s_dcplx = win2.Shared_query(0)
        base2 = np.ndarray(buffer=buf2,dtype=np.int8,shape=(1,)).ctypes.data
        phq = np.ndarray(buffer=(ctypes.c_char*n_phq).from_address(base2),dtype=np.complex128,shape=(nmodes,nq))
        win3 = MPI.Win.Allocate_shared(len_BE,s_double,comm=shm_comm_py)
        buf3,s_dcplx = win3.Shared_query(0)
        base3 = np.ndarray(buffer=buf3,dtype=np.int8,shape=(1,)).ctypes.data
        BEfactor = np.ndarray(buffer=(ctypes.c_char*n_BE).from_address(base3),dtype=np.float64,shape=(nmodes,nq))
        nfail = 0
        own1,_ = win1.Shared_query(shm_id)
        if np.ndarray(buffer=own1,dtype=np.int8,shape=(1,)).ctypes.data-base1!=n_phq*shm_id//shm_nprocs: nfail += 1
        own2,_ = win2.Shared_query(shm_id)
        if np.ndarray(buffer=own2,dtype=np.int8,shape=(1,)).ctypes.data-base2!=n_phq*shm_id//shm_nprocs: nfail += 1
        own3,_ = win3.Shared_query(shm_id)
        if np.ndarray(buffer=own3,dtype=np.int8,shape=(1,)).ctypes.data-base3!=n_BE*shm_id//shm_nprocs: nfail += 1
        nfail_tot = shm_comm_py.allreduce(nfail,op=MPI.SUM)
        if nfail_tot>0:
            if myid==0: print("shm contiguity FAIL nfail=%d"%nfail_tot,flush=True)
            shm_comm_py.Abort(1)
        else:
            if myid==0: print("shm contiguity OK",flush=True)
        np.asarray(phq0).view(np.int8).reshape(n_phq)[n_phq*shm_id//shm_nprocs:n_phq*(shm_id+1)//shm_nprocs] = 0
        np.asarray(phq).view(np.int8).reshape(n_phq)[n_phq*shm_id//shm_nprocs:n_phq*(shm_id+1)//shm_nprocs] = 0
        np.asarray(BEfactor).view(np.int8).reshape(n_BE)[n_BE*shm_id//shm_nprocs:n_BE*(shm_id+1)//shm_nprocs] = 0

        # Init phq0[nmode,nq] and phq
        mpi.MPI_Barrier(shm_comm)
        if (shm_id==0):
            GetPhQ(
                phonon,nmodes,nq,step_s,
                BEfactor,dt,hbar,KbT,phq0,phq
            )
    else:
        phq0 = np.zeros((nmodes,nq),dtype=np.complex128)
        phq = np.zeros((nmodes,nq),dtype=np.complex128)
        BEfactor = np.zeros((nmodes,nq),dtype=np.float64)
        GetPhQ(
            phonon,nmodes,nq,step_s,
            BEfactor,dt,hbar,KbT,phq0,phq
        )
    mpi.MPI_Barrier(shm_comm)

    if not LSPLIT:
        # write EPTXT, EPECTXT, EPPHTXT
        starttime = mpi.MPI_Wtime()
        WriteEPC(
            c_comm,myid,nprocs,namddir,sample,NSW,KbT,sigma,
            k_proc,k_proc_num,k_proc_min,nk_a,nq,n_p,
            nmodes,kqidx_p,epc_a,energy,phonon,phq,phq0
        )
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("Sample %d, EPCTXT time: %.6fs.\n",sample,endtime-starttime)
        # run TDDFT, get expansion coefficients psi
        starttime0 = mpi.MPI_Wtime()
        fssh_psi(
            c_comm,shm_comm,myid,nprocs,shm_id,shm_nprocs,namddir,
            sample,NSW,NELM,k_proc_min,nk_a,nq,n_p,nmodes,nel,edt,hbar,
            state_s,k_proc,k_proc_num,kqidx_p,energy,epc_a,phq,phq0
        )
        endtime0 = mpi.MPI_Wtime()
        if myid == 0:
            printf("Sample %d, TimeProp time: %.6fs.\n",sample,endtime0-starttime0)
        # run FSSH
        if myid == 0:
            pop_sh = np.zeros((NSW,nk_a),dtype=np.float32)
            ph_pop = np.zeros((NSW,nmodes,nq),dtype=np.float32)
            e_sh = np.zeros((NSW),dtype=np.float64)
            e_ph = np.zeros((NSW,nmodes),dtype=np.float64)
        else:
            pop_sh = np.zeros((0,0),dtype=np.float32)
            ph_pop = np.zeros((0,0,0),dtype=np.float32)

        starttime1 = mpi.MPI_Wtime()
        if LEPCSHM:
            fssh_sh_shm(
                c_comm,shm_comm,myid,nprocs,shm_id,shm_nprocs,
                namddir,sample,NTRAJ,NSW,nk_proc,nk_a,nq,n_p,nmodes,
                nel,dt,KbT,hbar,sigma,state_s,k_proc,kqidx_p,energy,
                phonon,BEfactor,epc_a,pop_sh,ph_pop,BANDDEG,LHOLE
            )
        else:
            fssh_sh_cache(
                c_comm,shm_comm,myid,nprocs,shm_id,shm_nprocs,
                namddir,sample,NTRAJ,NSW,nk_proc,nk_a,nq,n_p,nmodes,
                nel,dt,KbT,hbar,sigma,state_s,k_proc,kqidx_p,energy,
                phonon,BEfactor,epc_a,pop_sh,ph_pop,BANDDEG,LHOLE,LPHSHM
            )
        endtime1 = mpi.MPI_Wtime()
        if myid == 0:
            FsshOut(
                nmodes,NSW,nk_a,nq,ntraj_max,nel,
                energy,phonon,pop_sh,e_sh,ph_pop,e_ph
            )
            np.save(namddir+'/fssh_pop_sh-%d.npy'%(sample),pop_sh)
            np.save(namddir+'/fssh_e_sh-%d.npy'%(sample),e_sh)
            np.save(namddir+'/fssh_pop_ph-%d.npy'%(sample),ph_pop)
            np.save(namddir+'/fssh_e_ph-%d.npy'%(sample),e_ph)
            printf("Sample %d, FSSH time: %.6fs.\n",sample,endtime1-starttime1)
    else:
        if not os.path.exists(namddir+"/RESTART-%d"%(sample)):
            # write EPTXT, EPECTXT, EPPHTXT
            starttime = mpi.MPI_Wtime()
            WriteEPC(
                c_comm,myid,nprocs,namddir,sample,NSW,KbT,sigma,
                k_proc,k_proc_num,k_proc_min,nk_a,nq,n_p,
                nmodes,kqidx_p,epc_a,energy,phonon,phq,phq0
            )
            endtime = mpi.MPI_Wtime()
            if myid == 0:
                printf("Sample %d, EPCTXT time: %.6fs.\n",sample,endtime-starttime)
            # run TDDFT, get expansion coefficients psi
            starttime0 = mpi.MPI_Wtime()
            fssh_psi(
                c_comm,shm_comm,myid,nprocs,shm_id,shm_nprocs,namddir,
                sample,NSW,NELM,k_proc_min,nk_a,nq,n_p,nmodes,nel,edt,hbar,
                state_s,k_proc,k_proc_num,kqidx_p,energy,epc_a,phq,phq0
            )
            endtime0 = mpi.MPI_Wtime()
            if myid == 0:
                os.system('touch %s'%(namddir+"/RESTART-%d"%(sample)))
                printf("Sample %d, TimeProp time: %.6fs.\n",sample,endtime0-starttime0)
        else:
            # run FSSH
            if myid == 0:
                pop_sh = np.zeros((NSW,nk_a),dtype=np.float32)
                ph_pop = np.zeros((NSW,nmodes,nq),dtype=np.float32)
                e_sh = np.zeros((NSW),dtype=np.float64)
                e_ph = np.zeros((NSW,nmodes),dtype=np.float64)
            else:
                pop_sh = np.zeros((0,0),dtype=np.float32)
                ph_pop = np.zeros((0,0,0),dtype=np.float32)

            starttime1 = mpi.MPI_Wtime()
            if LEPCSHM:
                fssh_sh_shm(
                    c_comm,shm_comm,myid,nprocs,shm_id,shm_nprocs,
                    namddir,sample,NTRAJ,NSW,nk_proc,nk_a,nq,n_p,nmodes,
                    nel,dt,KbT,hbar,sigma,state_s,k_proc,kqidx_p,energy,
                    phonon,BEfactor,epc_a,pop_sh,ph_pop,BANDDEG,LHOLE
                )
            else:
                fssh_sh_cache(
                    c_comm,shm_comm,myid,nprocs,shm_id,shm_nprocs,
                    namddir,sample,NTRAJ,NSW,nk_proc,nk_a,nq,n_p,nmodes,
                    nel,dt,KbT,hbar,sigma,state_s,k_proc,kqidx_p,energy,
                    phonon,BEfactor,epc_a,pop_sh,ph_pop,BANDDEG,LHOLE,LPHSHM
                )
            endtime1 = mpi.MPI_Wtime()
            if myid == 0:
               FsshOut(
                   nmodes,NSW,nk_a,nq,ntraj_max,nel,
                   energy,phonon,pop_sh,e_sh,ph_pop,e_ph
               )
               #os.system('rm -rf %s'%(namddir+"/RESTART-%d"%(sample)))
               np.save(namddir+'/fssh_pop_sh-%d.npy'%(sample),pop_sh)
               np.save(namddir+'/fssh_e_sh-%d.npy'%(sample),e_sh)
               np.save(namddir+'/fssh_pop_ph-%d.npy'%(sample),ph_pop)
               np.save(namddir+'/fssh_e_ph-%d.npy'%(sample),e_ph)
               printf("Sample %d, FSSH time: %.6fs.\n",sample,endtime1-starttime1)
    if LPHSHM:
        MPI.Win.Free(win1)
        MPI.Win.Free(win2)
        MPI.Win.Free(win3)
