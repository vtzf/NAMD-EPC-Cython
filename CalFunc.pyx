#cython: language_level=3
#cython: cdivision=True

cimport cython
import numpy as np
cimport numpy as np
from mpi4py import MPI
from mpi4py cimport MPI
from mpi4py cimport libmpi as mpi
from libc.math cimport cos, sin, sqrt, exp, fabs, fmod, M_PI
from libc.stdio cimport printf
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

    cdef void cblas_zgemv(
        CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, int M, int N,
        void *alpha, void *A, int lda, void *X, int incX,
        void *beta, void *Y, int incY
    )


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
                        + cimag(epc_a[i,j+k_p_m,k])*phq_tmp

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
                           + cimag(epc_a[i,j+k_p_m,k])*phq_tmp

    for i in range(nk_p):
        H1_p[i,i+nk_min] += E_p[i]
        for j in range(nk_a):
            dH01[i,j] = (H1_p[i,j]-H0_p[i,j])*wht
            H1_p[i,j] = 0


# integration function
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void TimeProp(
    mpi.MPI_Comm shm_comm, mpi.MPI_Comm remote_comm, int nnode,
    double complex[:,::1] H0_p, double complex[:,::1] dH01,
    double complex * psi_a1, double complex * psi_a2,
    double complex * psi_buf, int nk_a, int nk_p, int nk_min,
    int[::1] psi_p_remote, int[::1] psi_p_remote_num, 
    int istep, int estep
):
    cdef int i, j
    cdef double complex zero = 0.0
    cdef double complex half = 0.5
    cdef double complex one = 1.0
    cdef double complex psi_t

    mpi.MPI_Barrier(shm_comm)
    if istep==0 and estep==0:
        cblas_zgemv(
            CblasRowMajor,CblasNoTrans,nk_p,nk_a,
            &half,&H0_p[0,0],nk_a,psi_a2,1,&zero,psi_buf,1
        )
        for i in range(nk_p):
            psi_a1[nk_min+i] += psi_buf[i]
        mpi.MPI_Barrier(shm_comm)
        if nnode > 1:
            if remote_comm != mpi.MPI_COMM_NULL:
                mpi.MPI_Allgatherv(
                    mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,
                    psi_a1,&psi_p_remote_num[0],&psi_p_remote[0],
                    mpi.MPI_DOUBLE_COMPLEX,remote_comm
                )
    else:
        cblas_zgemv(
            CblasRowMajor,CblasNoTrans,nk_p,nk_a,
            &one,&H0_p[0,0],nk_a,psi_a1,1,&zero,psi_buf,1
        )
        for i in range(nk_p):
            psi_a2[nk_min+i] += psi_buf[i]
        mpi.MPI_Barrier(shm_comm)
        if remote_comm != mpi.MPI_COMM_NULL:
            if nnode > 1:
                mpi.MPI_Allgatherv(
                    mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,
                    psi_a2,&psi_p_remote_num[0],&psi_p_remote[0],
                    mpi.MPI_DOUBLE_COMPLEX,remote_comm
                )
            for i in range(nk_a):
                psi_t = psi_a1[i]
                psi_a1[i] = psi_a2[i]
                psi_a2[i] = psi_t

    for i in range(nk_p):
        for j in range(nk_a):
            H0_p[i,j] += dH01[i,j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fssh_psi(
    mpi.MPI_Comm c_comm, int nprocs, int myid, int NSW, int NELM, 
    double edt, double hbar, int[::1] state_s, 
    int k_p_m, int nk_a, int nq, int n_p, int nmodes, 
    int nel, int[::1] k_proc, int[::1] k_proc_num, int[:,::1] kqidx_p,
    double complex[:,:,::1] epc_a, double[::1] energy, 
    double complex[:,::1] phq, double complex[:,::1] phq0,
    double complex[:,::1] psi_t
):
    cdef mpi.MPI_Comm remote_comm, shm_comm
    cdef mpi.MPI_Win win1, win2
    cdef mpi.MPI_Aint len_psi_byte, len_psi_byte1
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef int[::1] psi_p_remote, psi_p_remote_num

    cdef int s_int = sizeof(int)
    cdef int s_dcplx = sizeof(double complex)

    cdef int i, j, k, iel, shm_proc_s, shm_proc_e, ierr, shm_id, shm_nprocs
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
    cdef double complex * psi_buf = <double complex*>malloc(nk_proc*s_dcplx)
    cdef double starttime, endtime

    # get shm_comm
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&remote_comm)

    # get psi_p range
    psi_p_remote = np.zeros((nnode+1),dtype=np.int32)
    psi_p_remote_num = np.zeros((nnode),dtype=np.int32)
    for i in range(nnode):
        shm_proc_s = i*shm_nprocs
        shm_proc_e = (i+1)*shm_nprocs
        for j in range(shm_proc_s,shm_proc_e):
            psi_p_remote_num[i] += k_proc_num[j]
        for j in range(i+1,nnode+1):
            psi_p_remote[j] += psi_p_remote_num[i]

    if (shm_id==0):
        len_psi_byte = nk_a*s_dcplx
        len_psi_byte1 = nk_a*s_dcplx
    else:
        len_psi_byte = 0
        len_psi_byte1 = 0

    mpi.MPI_Win_allocate_shared(
        len_psi_byte,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&psi_a1,&win1
    )
    mpi.MPI_Win_allocate_shared(
        len_psi_byte1,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&psi_a2,&win2
    )

    if (shm_id==0):
        memset(psi_a1,0,nk_a*s_dcplx)
        memset(psi_a2,0,nk_a*s_dcplx)
        for iel in range(nel):
            psi_a1[state_s[iel]] += 1.0/sqrt(nel)
            psi_a2[state_s[iel]] += 1.0/sqrt(nel)
    else:
        mpi.MPI_Win_shared_query(
            win1,0,&len_psi_byte,&s_dcplx,&psi_a1
        )
        mpi.MPI_Win_shared_query(
            win2,0,&len_psi_byte1,&s_dcplx,&psi_a2
        )

    for i in range(nk_proc):
        j = nk_min+i
        E_p[i] = energy[j]*factor

    GetH0(
        nmodes,nk_proc,nk_a,nq,nk_min,k_p_m,
        kqidx_p,E_p,factor,np2,epc_a,phq_t,phq,H0_p
    )
    for j in range(NSW):
        starttime = mpi.MPI_Wtime()
        if shm_id == 0:
            memcpy(&psi_t[j,0],psi_a1,nk_a*s_dcplx)
        GetH(
            nmodes,nk_proc,nk_a,nq,nk_min,k_p_m,kqidx_p,
            wht,E_p,epc_a,phq_t,phq0,H0_p,H1_p,dH01
        )
        for k in range(NELM):
            TimeProp(
                shm_comm,remote_comm,nnode,H0_p,dH01,
                psi_a1,psi_a2,psi_buf,nk_a,nk_proc,
                nk_min,psi_p_remote,psi_p_remote_num,j,k
            )
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("TimeProp in %d fs: %.5fs.\n",j,endtime-starttime)

    mpi.MPI_Comm_free(&shm_comm)
    mpi.MPI_Comm_free(&remote_comm)
    mpi.MPI_Group_free(&shm_gp)
    mpi.MPI_Group_free(&split_gp)
    mpi.MPI_Win_free(&win1)
    mpi.MPI_Win_free(&win2)
    free(nodelist)
    free(psi_buf)


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
    int * kqidx0, double[:,::1] phonon,
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
            epc_p0 = epc_p[i*nk_a+j]
            epc_p1 = (creal(epc_p0)*creal(epc_p0) \
                     +cimag(epc_p0)*cimag(epc_p0))*BE2[i,qidx]
            H_hop[j] += epc_p1*expdE


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int fsshhop(
    int it, int nmodes, int nk_a,
    double complex[::1] psi_a, double[::1] H_hop, double *dE, 
    double complex * epc_p, int * kqidx0, double[::1] energy, 
    double[:,::1] phonon, double[:,::1] BE2,
    int state, double f0, double f2, double KbT1,
    double *prop, double *prop_a, double *ph_pop_t, float[:,::1] ph_pop_p, 
    int[::1] n_occ, bint LHOLE, int BANDDEG
):
    cdef int i, j, k, idx_s, idx_e, qidx, state_t
    cdef int state_n = state
    cdef int state_t0 = <int>(state/BANDDEG)
    cdef double complex psi0 = conj(psi_a[state_t0])
    cdef double densitydiag = \
                f2/(creal(psi0)*creal(psi0)+cimag(psi0)*cimag(psi0))
    cdef double dEt, prop_t, epc_p1, prop0, prop_sum
    cdef double complex density, epc_p0

    cdef double ph_pop_sum, ph_pop0, ph_pop1, \
                dE0, dE1, expdE0, expdE1

    prop0 = np.random.rand()

    prop_sum = 0.0
    for i in range(nk_a):
        density = psi_a[i]*psi0
        prop_t = cimag(density)*H_hop[i]*densitydiag
        if prop_t>0:
            prop[i] = prop_t
            prop_sum += prop_t
        else: prop[i] = 0.0

    if LHOLE:
        for i in range(nk_a):
            prop_t = prop[i]
            if prop_t > 0:
                dEt = dE[i]
                if dEt<0: prop_t *= exp(dEt*KbT1)
#                prop_t /= prop_sum
                for j in range(BANDDEG):
                    prop_a[i*BANDDEG+j] += prop_t/BANDDEG*j
                for j in range((i+1)*BANDDEG,nk_a*BANDDEG+1):
                    prop_a[j] += prop_t
    else:
        for i in range(nk_a):
            prop_t = prop[i]
            if prop_t > 0:
                dEt = dE[i]
                if dEt>0: prop_t *= exp(-dEt*KbT1)
#                prop_t /= prop_sum
                for j in range(BANDDEG):
                    prop_a[i*BANDDEG+j] += prop_t/BANDDEG*j
                for j in range((i+1)*BANDDEG,nk_a*BANDDEG+1):
                    prop_a[j] += prop_t

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
                epc_p0 = epc_p[i*nk_a+state_t]
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
    mpi.MPI_Comm c_comm, int nprocs,
    int myid, int NTRAJ, int NSW, double KbT, double hbar, double sigma, 
    double dt, int[::1] state_s, int nk_proc, int nk_a, int nq, int n_p, 
    int nmodes, int nel, int[::1] k_proc, 
    int[:,::1] kqidx_p, double complex[:,:,::1] epc_a, double[::1] energy,
    double[:,::1] phonon, double[:,::1] BEfactor, 
    double complex[:,::1] psi_t, float[:,::1] pop_sh,
    float[:,:,::1] ph_pop, bint LHOLE, int BANDDEG 
):
    cdef double KbT1 = 1.0/KbT
    cdef double f0 = -0.5/(sigma*sigma)
    cdef double f1 = sqrt(2*M_PI)/sigma/n_p
    cdef double f2 = <double>(2.0*dt)/hbar
    cdef double E0

    cdef int i, j, k, l, m, n, p, state, k_state_p, state_t, \
             t0, t1, it, send_s, send_e, nsend, nsend_set, \
             shm_id, shm_nprocs, ierr, nnode, node_id, \
             inode, iproc, iel, recv_id, n_s_req, n_s_req_b

    cdef int * nodelist
    cdef int * idx_all
    cdef int * sendset
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

    cdef mpi.MPI_Comm shm_comm, node_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Win win0, win1, win2, win3
    cdef mpi.MPI_Aint l_idx_all, l_recvbuf, l_recvtag
    cdef mpi.MPI_Request * send_req 
    cdef mpi.MPI_Request * send_req_b
    cdef mpi.MPI_Request recv_req

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
        l_recvbuf = shm_nprocs*recvlen
        l_recvtag = nprocs*s_int*2
    else:
        l_idx_all = 0
        l_recvbuf = 0
        l_recvtag = 0
    mpi.MPI_Win_allocate_shared(
        l_idx_all,s_int,mpi.MPI_INFO_NULL,shm_comm,&idx_all,&win0
    )
    mpi.MPI_Win_allocate_shared(
        l_recvbuf,s_void,mpi.MPI_INFO_NULL,shm_comm,&recvbuf,&win1
    )
    mpi.MPI_Win_allocate_shared(
        l_recvtag,s_int,mpi.MPI_INFO_NULL,shm_comm,&recvtag,&win2
    )
    if (shm_id!=0):
        mpi.MPI_Win_shared_query(win0,0,&l_idx_all,&s_int,&idx_all)
        mpi.MPI_Win_shared_query(win1,0,&l_recvbuf,&s_void,&recvbuf)
        mpi.MPI_Win_shared_query(win2,0,&l_recvtag,&s_int,&recvtag)
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

    for it in range(NSW):
        starttime = mpi.MPI_Wtime()
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
                    nmodes,nk_a,kqidx,
                    phonon,BE2,epc_p,f0,H_hop,dE
                )
                state_a[i,iel] = fsshhop(
                    it,nmodes,nk_a,psi_t[it],H_hop,dE,epc_p,kqidx,
                    energy,phonon,BE2,state_a[i,iel],f0,f2,KbT1,
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

    free(ph_pop_t)
    free(prop)
    free(prop_a)
    free(dE)

    # free
    free(nodelist)
    mpi.MPI_Comm_free(&shm_comm)
    if (node_comm != mpi.MPI_COMM_NULL):
        mpi.MPI_Comm_free(&node_comm)
    mpi.MPI_Group_free(&shm_gp)
    mpi.MPI_Group_free(&split_gp)
    mpi.MPI_Win_free(&win0)
    mpi.MPI_Win_free(&win1)
    mpi.MPI_Win_free(&win2)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fssh_sh_shm(
    mpi.MPI_Comm c_comm, int nprocs,
    int myid, int NTRAJ, int NSW, double KbT, double hbar,
    double sigma, double dt, int[::1] state_s, int nk_proc,
    int nk_a, int nq, int n_p, int nmodes, int nel, int[::1] k_proc,
    int[:,::1] kqidx_p, double complex[:,:,::1] epc_a,
    double[::1] energy, double[:,::1] phonon, double[:,::1] BEfactor,
    double complex[:,::1] psi_t, float[:,::1] pop_sh,
    float[:,:,::1] ph_pop, bint LHOLE, int BANDDEG
):
    cdef double KbT1 = 1.0/KbT
    cdef double f0 = -0.5/(sigma*sigma)
    cdef double f1 = sqrt(2*M_PI)/sigma/n_p
    cdef double f2 = <double>(2.0*dt)/hbar
    cdef double E0

    cdef int i, j, k, l, it, t0, t1, shm_id, shm_nprocs, ierr,\
             nnode, node_id, iel, state_t
    cdef int * nodelist

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

    cdef double complex *epc_buf \
        = <double complex*>malloc(s_dcplx*nmodes*nk_a)
    cdef int *kqidx_buf

    cdef double *ph_pop_t = <double*>malloc(nmodes*s_double)
    cdef double *prop = <double*>malloc(nk_a*s_double)
    cdef double *prop_a = <double*>malloc((nk_a*BANDDEG+1)*s_double)
    cdef double *dE = <double*>malloc(nk_a*s_double)
    cdef double starttime, endtime

    cdef mpi.MPI_Comm shm_comm, node_comm
    cdef mpi.MPI_Group split_gp, shm_gp

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

    for it in range(NSW):
        starttime = mpi.MPI_Wtime()
        memset(&pop_sh_p[0],0,nk_a*s_float)
        memset(&ph_pop_p[0,0],0,nmodes*nq*s_float)
        for i in range(itraj_max):
            for iel in range(nel):
                # calculate H_hop from (epc_buf,kqidx)
                state_t = <int>(state_a[i,iel]/BANDDEG)
                for t0 in range(nmodes):
                    for t1 in range(nk_a):
                        epc_buf[t0*nk_a+t1] = epc_a[t0,state_t,t1]
                E0 = energy[state_t]
                for j in range(nk_a):
                    H_hop[j] = 0
                    dE[j] = energy[j]-E0
                memset(&prop_a[1],0,nk_a*BANDDEG*s_double)
                kqidx_buf = &kqidx_p[state_t,0]
                GetHhop(
                    nmodes,nk_a,kqidx_buf,
                    phonon,BE2,epc_buf,f0,H_hop,dE
                )
                state_a[i,iel] = fsshhop(
                    it,nmodes,nk_a,psi_t[it],H_hop,dE,epc_buf,kqidx_buf,
                    energy,phonon,BE2,state_a[i,iel],f0,f2,KbT1,
                    prop,prop_a,ph_pop_t,ph_pop_p,n_occ[i],LHOLE,BANDDEG
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

    free(ph_pop_t)
    free(prop)
    free(prop_a)
    free(dE)

    # free
    free(epc_buf)
    free(nodelist)
    mpi.MPI_Comm_free(&shm_comm)
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
    int nmodes, int NSW, int nk_a, int nq, int ntraj, 
    int nel, double complex[:,::1] psi_t,
    double[::1] energy, double[:,::1] phonon, 
    double[::1] e_psi, float[:,::1] pop_sh,
    double[::1] e_sh,
    float[:,:,::1] ph_pop, double[:,::1] e_ph
):
    cdef int i, j, k
    cdef double psi2
    cdef double complex psi1

    for i in range(NSW):
        for j in range(nk_a):
            psi1 = psi_t[i,j]
            psi2 = creal(psi1)*creal(psi1) \
                 + cimag(psi1)*cimag(psi1)
            e_psi[i] += psi2*energy[j]
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
    #cdef mpi.MPI_Win win
    cdef mpi.MPI_Comm shm_comm
    #cdef mpi.MPI_Aint len_psi
    cdef int myid, nprocs, ierr, shm_id
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    cdef long len_psi, len_phq, len_phq0, len_BE
    cdef long NSW_l = NSW
    cdef long nq_l = nq
    cdef int s_double = sizeof(double)
    cdef int s_dcplx = sizeof(double complex)
    cdef int itraj_max, ntraj_max, k_proc_min
    cdef int nk_proc = k_proc_num[myid]
    cdef double[:,::1] BEfactor
    cdef double complex[:,::1] phq0, phq
    cdef double complex[:,::1] psi_t, psi_tmp
    cdef double[::1] e_psi, e_sh 
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
    # alloc psi_t, phq0, phq, BEfactor
    shm_comm_py = comm.Split_type(MPI.COMM_TYPE_SHARED)
    if LPHSHM:
        if (shm_id==0):
            len_psi = NSW_l*nk_a*s_dcplx
            len_phq0 = nq_l*nmodes*s_dcplx
            len_phq = nq_l*nmodes*s_dcplx
            len_BE = nq_l*nmodes*s_double
        else:
            len_psi = 0
            len_phq0 = 0
            len_phq = 0
            len_BE = 0
        win = MPI.Win.Allocate_shared(len_psi,s_dcplx,comm=shm_comm_py)
        buf,s_dcplx = win.Shared_query(0)
        psi_t = np.ndarray(buffer=buf,dtype=np.complex128,shape=(NSW,nk_a))
        win1 = MPI.Win.Allocate_shared(len_phq0,s_dcplx,comm=shm_comm_py)
        buf1,s_dcplx = win1.Shared_query(0)
        phq0 = np.ndarray(buffer=buf1,dtype=np.complex128,shape=(nmodes,nq))
        win2 = MPI.Win.Allocate_shared(len_phq,s_dcplx,comm=shm_comm_py)
        buf2,s_dcplx = win2.Shared_query(0)
        phq = np.ndarray(buffer=buf2,dtype=np.complex128,shape=(nmodes,nq))
        win3 = MPI.Win.Allocate_shared(len_BE,s_double,comm=shm_comm_py)
        buf3,s_dcplx = win3.Shared_query(0)
        BEfactor = np.ndarray(buffer=buf3,dtype=np.float64,shape=(nmodes,nq))
    
        # Init phq0[nmode,nq] and phq
        mpi.MPI_Barrier(shm_comm)
        if (shm_id==0):
            GetPhQ(
                phonon,nmodes,nq,step_s,
                BEfactor,dt,hbar,KbT,phq0,phq
            )
    else:
        if (shm_id==0):
            len_psi = NSW_l*nk_a*s_dcplx
        else:
            len_psi = 0
        win = MPI.Win.Allocate_shared(len_psi,s_dcplx,comm=shm_comm_py)
        buf,s_dcplx = win.Shared_query(0)
        psi_t = np.ndarray(buffer=buf,dtype=np.complex128,shape=(NSW,nk_a))
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
        # run TDDFT, get expansion coefficients psi_t
        starttime0 = mpi.MPI_Wtime()
        fssh_psi(
            c_comm,nprocs,myid,NSW,NELM,edt,hbar,state_s,
            k_proc_min,nk_a,nq,n_p,nmodes,nel,k_proc,
            k_proc_num,kqidx_p,epc_a,energy,phq,phq0,psi_t
        )
        endtime0 = mpi.MPI_Wtime()
        if myid == 0:
            np.save(namddir+'/psi-%d.npy'%(sample),psi_t)
            printf("Sample %d, TimeProp time: %.6fs.\n",sample,endtime0-starttime0)
        # run FSSH
        #psi_t = np.load(namddir+'/psi-%d.npy'%(sample))
        if myid == 0:
            pop_sh = np.zeros((NSW,nk_a),dtype=np.float32)
            ph_pop = np.zeros((NSW,nmodes,nq),dtype=np.float32)
            e_psi = np.zeros((NSW),dtype=np.float64)
            e_sh = np.zeros((NSW),dtype=np.float64)
            e_ph = np.zeros((NSW,nmodes),dtype=np.float64)
        else:
            pop_sh = np.zeros((0,0),dtype=np.float32)
            ph_pop = np.zeros((0,0,0),dtype=np.float32)
    
        starttime1 = mpi.MPI_Wtime()
        if LEPCSHM:
            fssh_sh_shm(
                c_comm,nprocs,myid,NTRAJ,NSW,KbT,hbar,sigma,dt,
                state_s,nk_proc,nk_a,nq,n_p,nmodes,nel,k_proc,
                kqidx_p,epc_a,energy,phonon,BEfactor,psi_t,pop_sh,
                ph_pop,LHOLE,BANDDEG
            )
        else:
            fssh_sh(
                c_comm,nprocs,myid,NTRAJ,NSW,KbT,hbar,sigma,dt,
                state_s,nk_proc,nk_a,nq,n_p,nmodes,nel,k_proc,
                kqidx_p,epc_a,energy,phonon,BEfactor,psi_t,pop_sh,
                ph_pop,LHOLE,BANDDEG
            )
        endtime1 = mpi.MPI_Wtime()
        if myid == 0:
            FsshOut(
                nmodes,NSW,nk_a,nq,ntraj_max,nel,psi_t,energy,
                phonon,e_psi,pop_sh,e_sh,ph_pop,e_ph
            )
            np.save(namddir+'/fssh_e_psi-%d.npy'%(sample),e_psi)
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
            # run TDDFT, get expansion coefficients psi_t
            starttime0 = mpi.MPI_Wtime()
            fssh_psi(
                c_comm,nprocs,myid,NSW,NELM,edt,hbar,state_s,
                k_proc_min,nk_a,nq,n_p,nmodes,nel,k_proc,
                k_proc_num,kqidx_p,epc_a,energy,phq,phq0,psi_t
            )
            endtime0 = mpi.MPI_Wtime()
            if myid == 0:
                os.system('touch %s'%(namddir+"/RESTART-%d"%(sample)))
                np.save(namddir+'/psi-%d.npy'%(sample),psi_t)
                printf("Sample %d, TimeProp time: %.6fs.\n",sample,endtime0-starttime0)
        else:
            # load psi
            if (shm_id==0):
                psi_tmp = np.load(namddir+'/psi-%d.npy'%(sample))
                memcpy(&psi_t[0,0],&psi_tmp[0,0],len_psi)
                psi_tmp = None
            # run FSSH
            if myid == 0:
                pop_sh = np.zeros((NSW,nk_a),dtype=np.float32)
                ph_pop = np.zeros((NSW,nmodes,nq),dtype=np.float32)
                e_psi = np.zeros((NSW),dtype=np.float64)
                e_sh = np.zeros((NSW),dtype=np.float64)
                e_ph = np.zeros((NSW,nmodes),dtype=np.float64)
            else:
                pop_sh = np.zeros((0,0),dtype=np.float32)
                ph_pop = np.zeros((0,0,0),dtype=np.float32)

            starttime1 = mpi.MPI_Wtime()
            if LEPCSHM:
                fssh_sh_shm(
                    c_comm,nprocs,myid,NTRAJ,NSW,KbT,hbar,sigma,dt,
                    state_s,nk_proc,nk_a,nq,n_p,nmodes,nel,k_proc,
                    kqidx_p,epc_a,energy,phonon,BEfactor,psi_t,pop_sh,
                    ph_pop,LHOLE,BANDDEG
                )
            else:
                fssh_sh(
                    c_comm,nprocs,myid,NTRAJ,NSW,KbT,hbar,sigma,dt,
                    state_s,nk_proc,nk_a,nq,n_p,nmodes,nel,k_proc,
                    kqidx_p,epc_a,energy,phonon,BEfactor,psi_t,pop_sh,
                    ph_pop,LHOLE,BANDDEG
                )
            endtime1 = mpi.MPI_Wtime()
            if myid == 0:
               FsshOut(
                   nmodes,NSW,nk_a,nq,ntraj_max,nel,psi_t,energy,
                   phonon,e_psi,pop_sh,e_sh,ph_pop,e_ph
               )
               #os.system('rm -rf %s'%(namddir+"/RESTART-%d"%(sample)))
               np.save(namddir+'/fssh_e_psi-%d.npy'%(sample),e_psi)
               np.save(namddir+'/fssh_pop_sh-%d.npy'%(sample),pop_sh)
               np.save(namddir+'/fssh_e_sh-%d.npy'%(sample),e_sh)
               np.save(namddir+'/fssh_pop_ph-%d.npy'%(sample),ph_pop)
               np.save(namddir+'/fssh_e_ph-%d.npy'%(sample),e_ph)
               printf("Sample %d, FSSH time: %.6fs.\n",sample,endtime1-starttime1)
    MPI.Win.Free(win)
    if LPHSHM:
        MPI.Win.Free(win1)
        MPI.Win.Free(win2)
        MPI.Win.Free(win3)
