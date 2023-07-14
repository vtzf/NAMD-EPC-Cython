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
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, calloc, free, qsort

cdef extern from "complex.h":
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
cdef int Cmp(void * pa, void * pb) nogil:
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
    int nk_min, int[:,::1] kqidx_p,
    double complex[::1] E_p, double complex factor, double nq2,
    double complex[:,:,::1] epc_a, double complex[:,::1] phq_t, 
    double complex[:,::1] phq, double complex[:,::1] H
):
    cdef int i, j, k
    cdef double phq_tmp

    for i in range(nmodes):
        for j in range(nq):
            phq_t[i,j] = 2*phq[i,j]*factor*nq2
        for j in range(nk_p):
            for k in range(nk_a):
                phq_tmp = cimag(phq_t[i,kqidx_p[j,k]])
                H[j,k] += creal(epc_a[i,j,k])*phq_tmp*1j \
                        + cimag(epc_a[i,j,k])*phq_tmp

    for i in range(nk_p):
        H[i,i+nk_min] += E_p[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void GetH(
    int nmodes, int nk_p, int nk_a, int nq,
    int nk_min, int[:,::1] kqidx_p, 
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
                phq_tmp = cimag(phq_t[i,kqidx_p[j,k]])
                H1_p[j,k] += creal(epc_a[i,j,k])*phq_tmp*1j \
                           + cimag(epc_a[i,j,k])*phq_tmp

    for i in range(nk_p):
        H1_p[i,i+nk_min] += E_p[i]
        for j in range(nk_a):
            dH01[i,j] = (H1_p[i,j]-H0_p[i,j])*wht
            H1_p[i,j] = 0


# integration function
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void TimePropShm(
    mpi.MPI_Comm shm_comm, int shm_id,
    double complex[:,::1] H0_p, double complex[:,::1] dH01,
    double complex * psi_a1, double complex * psi_a2, 
    double complex * psi_buf, 
    int nk_a, int nk_p, int nk_min, int istep, int estep
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
    else:
        cblas_zgemv(
            CblasRowMajor,CblasNoTrans,nk_p,nk_a,
            &one,&H0_p[0,0],nk_a,psi_a1,1,&zero,psi_buf,1
        )
        for i in range(nk_p):
            psi_a2[nk_min+i] += psi_buf[i]
        mpi.MPI_Barrier(shm_comm)
        if shm_id == 0:
            for i in range(nk_a):
                psi_t = psi_a1[i]
                psi_a1[i] = psi_a2[i]
                psi_a2[i] = psi_t

    for i in range(nk_p):
        for j in range(nk_a):
            H0_p[i,j] += dH01[i,j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void TimeProp(
    mpi.MPI_Comm shm_comm, mpi.MPI_Comm remote_comm,
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
        if remote_comm != mpi.MPI_COMM_NULL:
            mpi.MPI_Allgatherv(
                mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,psi_a1,
                &psi_p_remote_num[0],&psi_p_remote[0],
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
            mpi.MPI_Allgatherv(
                mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,psi_a2,
                &psi_p_remote_num[0],&psi_p_remote[0],
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
    mpi.MPI_Comm c_comm, int nprocs,
    int myid, int NSW, int NELM, double edt, double hbar,
    int state_s, int nk_proc, int nk_a, int nq, int nmodes, 
    int[::1] k_proc, int[::1] k_proc_num, int[:,::1] kqidx_p,
    double complex[:,:,::1] epc_a, double[::1] energy, 
    double complex[:,::1] phq, double complex[:,::1] phq0,
    double complex[:,::1] psi_t
):
    cdef mpi.MPI_Comm remote_comm, shm_comm
    cdef mpi.MPI_Win win1, win2
    cdef mpi.MPI_Aint len_psi_byte
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef int[::1] psi_p_remote, psi_p_remote_num

    cdef int s_int = sizeof(int)
    cdef int s_dcplx = sizeof(double complex)

    cdef int i, j, k, shm_proc_s, shm_proc_e, ierr, shm_id, shm_nprocs
    cdef int * nodelist
    cdef double wht = 1.0/NELM
    cdef double complex factor = 2*edt/(1j*hbar)
    cdef int nk_min = k_proc[myid]
    cdef double complex[::1] E_p = np.zeros((nk_proc),dtype=np.complex128)
    cdef double nq2 = 1.0/sqrt(nq)

    cdef double complex[:,::1] phq_t = np.zeros((nmodes,nq),dtype=np.complex128)
    cdef double complex[:,::1] H0_p = np.zeros((nk_proc,nk_a),dtype=np.complex128)
    cdef double complex[:,::1] H1_p = np.zeros((nk_proc,nk_a),dtype=np.complex128)
    cdef double complex[:,::1] dH01 = np.zeros((nk_proc,nk_a),dtype=np.complex128)

    cdef double complex * psi_a1
    cdef double complex * psi_a2
    cdef double complex * psi_buf = <double complex*>malloc(nk_proc*s_dcplx)

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
    else:
        len_psi_byte = 0

    mpi.MPI_Win_allocate_shared(
        len_psi_byte,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&psi_a1,&win1
    )
    mpi.MPI_Win_allocate_shared(
        len_psi_byte,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&psi_a2,&win2
    )

    if (shm_id==0):
        memset(psi_a1,0,nk_a*s_dcplx)
        memset(psi_a2,0,nk_a*s_dcplx)
        psi_a1[state_s] = 1
        psi_a2[state_s] = 1
    else:
        mpi.MPI_Win_shared_query(
            win1,0,&len_psi_byte,&s_dcplx,&psi_a1
        )
        mpi.MPI_Win_shared_query(
            win2,0,&len_psi_byte,&s_dcplx,&psi_a2
        )

    for i in range(nk_proc):
        j = nk_min+i
        E_p[i] = energy[j]*factor

    GetH0(
        nmodes,nk_proc,nk_a,nq,nk_min,kqidx_p,
        E_p,factor,nq2,epc_a,phq_t,phq,H0_p
    )
    if shm_nprocs == nprocs:
        for j in range(NSW):
            if myid == 0:
                memcpy(&psi_t[j,0],psi_a1,nk_a*s_dcplx)
            GetH(
                nmodes,nk_proc,nk_a,nq,nk_min,kqidx_p,
                wht,E_p,epc_a,phq_t,phq0,H0_p,H1_p,dH01
            )
            for k in range(NELM):
                TimePropShm(
                    shm_comm,shm_id,H0_p,dH01,psi_a1,psi_a2,
                    psi_buf,nk_a,nk_proc,nk_min,j,k
                )
    else:
        for j in range(NSW):
            if shm_id == 0:
                memcpy(&psi_t[j,0],psi_a1,nk_a*s_dcplx)
            GetH(
                nmodes,nk_proc,nk_a,nq,nk_min,kqidx_p,
                wht,E_p,epc_a,phq_t,phq0,H0_p,H1_p,dH01
            )
            for k in range(NELM):
                TimeProp(
                    shm_comm,remote_comm,H0_p,dH01,psi_a1,psi_a2,
                    psi_buf,nk_a,nk_proc,nk_min,psi_p_remote,
                    psi_p_remote_num,j,k
                )

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
    int myid, int shm_id, int node_id, 
    int nprocs, int shm_nprocs, int state, 
    int * idx_all, int[::1] k_proc, int * nowait
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

    idx_all[shm_id*7] = state
    idx_all[shm_id*7+1] = state-k_proc[state_id0]
    idx_all[shm_id*7+2] = node_id
    idx_all[shm_id*7+3] = state_id0
    idx_all[shm_id*7+4] = state_id0/shm_nprocs
    idx_all[shm_id*7+5] = shm_id
    idx_all[shm_id*7+6] = myid

    if (idx_all[shm_id*7+3]==myid):
        nowait[0] = 1
    else:
        nowait[0] = 0


# FSSH functions
@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void GetHhop(
    int nmodes, int nk_a,
    int * kqidx0, double[::1] energy, double[:,::1] phonon,
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
            epc_p1 = creal(conj(epc_p0)*epc_p0)*BE2[i,qidx]
            H_hop[j] += epc_p1*expdE


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int fsshhop(
    int it, int nmodes, int nk_a,
    double complex[::1] psi_a, double[::1] H_hop, double *dE, 
    double complex * epc_p, int * kqidx0, double[::1] energy, 
    double[:,::1] phonon, double[:,::1] BE2,
    int state, double f0, double f2, double KbT1,
    double *prop, double *ph_pop_t, double[:,::1] ph_pop_p,
    bint LHOLE
):
    cdef int i, j, idx_s, idx_e, qidx
    cdef int state_n = state
    cdef double complex psi0 = conj(psi_a[state])
    cdef double densitydiag = \
                f2/creal(conj(psi0)*psi0)
    cdef double dEt, prop_t, epc_p1
    cdef double complex density, epc_p0
    cdef double prop0 = np.random.rand()

    cdef double ph_pop_sum, ph_pop0, ph_pop1, \
                dE0, dE1, expdE0, expdE1

    if LHOLE:
        for i in range(nk_a):
            density = psi_a[i]*psi0
            prop_t = cimag(density)*H_hop[i]*densitydiag
            if prop_t>0:
                dEt = dE[i]
                if dEt<0:
                    prop_t *= exp(dEt*KbT1)
                for j in range(i+1,nk_a+1):
                    prop[j] += prop_t
    else:
        for i in range(nk_a):
            density = psi_a[i]*psi0
            prop_t = cimag(density)*H_hop[i]*densitydiag
            if prop_t>0:
                dEt = dE[i]
                if dEt>0:
                    prop_t *= exp(-dEt*KbT1)
                for j in range(i+1,nk_a+1):
                    prop[j] += prop_t

    if (prop0<=prop[nk_a]):
        idx_s = 0
        idx_e = nk_a
        while 1:
            if ((idx_e-idx_s)<=1):
                state_n = idx_s
                break
            state_n = (idx_s+idx_e)/2
            if (prop0<prop[state_n]):
                idx_e = state_n
            else:
                idx_s = state_n

    if (state_n != state):
        qidx = kqidx0[state_n]
        dEt = dE[state_n]
        ph_pop_sum = 0
        for i in range(nmodes):
            dE0 = dEt+phonon[i,qidx]+1e-8
            dE1 = dEt-phonon[i,qidx]-1e-8
            expdE0 = exp(dE0*dE0*f0)
            expdE1 = exp(dE1*dE1*f0)
            epc_p0 = epc_p[i*nk_a+state_n]
            epc_p1 = creal(conj(epc_p0)*epc_p0)*BE2[i,qidx]
            ph_pop0 = epc_p1*expdE0
            ph_pop1 = epc_p1*expdE1
            ph_pop_t[i] = ph_pop1 - ph_pop0
            ph_pop_sum += ph_pop0 + ph_pop1
        if ph_pop_sum>0:
            for i in range(nmodes):
                ph_pop_p[i,qidx] += ph_pop_t[i]/ph_pop_sum

    return state_n


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void fssh_sh(
    mpi.MPI_Comm c_comm, int nprocs,
    int myid, int NTRAJ, int NSW, double KbT, double hbar,
    double sigma, double dt, int state_s, int nk_proc, 
    int nk_a, int nq, int nmodes, int[::1] k_proc,
    int[:,::1] kqidx_p, double complex[:,:,::1] epc_a, 
    double[::1] energy, double[:,::1] phonon, double[:,::1] BEfactor, 
    double complex[:,::1] psi_t, float[:,::1] pop_sh, double[:,:,::1] ph_pop,
    bint LHOLE
):
    cdef double KbT1 = 1.0/KbT
    cdef double f0 = -0.5/(sigma*sigma)
    cdef double f1 = sqrt(2*M_PI)/sigma/nq
    cdef double f2 = <double>(2.0*dt)/hbar
    cdef double E0

    cdef int i, j, k, l, m, n, p, state, sd_id, sd_num,\
             k_state_p, b_state_p, idx,\
             idx0, idx1, idx2, idx3, idx4, idx5,\
             idx_t, idx_t0, idx_t1, idx_t2, t0, t1, t2, it,\
             send_s, send_e, nsend, shm_id, shm_nprocs, ierr,\
             nnode, node_id, nowait

    cdef int * setlen
    cdef int * setlist
    cdef int * setlistr
    cdef int * nodelist
    cdef int * idx_all
    cdef int * idx_set
    cdef int * idx_set_all
    cdef int * idx_recv
    
    cdef double[:,::1] BE2 = np.zeros((nmodes,nq),dtype=np.float64)
    cdef double[::1] H_hop = np.zeros((nk_a),dtype=np.float64)
    cdef float[::1] pop_sh_p = np.zeros((nk_a),dtype=np.float32)
    cdef double[:,::1] ph_pop_p = np.zeros((nmodes,nq),dtype=np.float64)

    cdef int itraj_max = <int>(NTRAJ/nprocs+1)
    cdef int ntraj_max = itraj_max*nprocs
    cdef int[::1] state_a = np.zeros((itraj_max),dtype=np.int32)

    cdef int s_int = sizeof(int)
    cdef int s_float = sizeof(float)
    cdef int s_double = sizeof(double)
    cdef int s_dcplx = sizeof(double complex)
    cdef int s_req = sizeof(mpi.MPI_Request)
    cdef int s_void = 1

    cdef int epclen = nmodes*nk_a
    cdef int recvlen = epclen*s_dcplx+nk_a*s_int
    cdef char ** sendbuf
    cdef double complex *epc_buf
    cdef int *kqidx0_buf
    cdef char * recvbuf
    cdef double complex * epc_p
    cdef int * kqidx0
    
    cdef double *ph_pop_t = <double*>malloc(nmodes*s_double)
    cdef double *prop = <double*>calloc(nk_a+1,s_double)
    cdef double *dE = <double*>malloc(nk_a*s_double)

    cdef mpi.MPI_Comm shm_comm, node_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Win win0, win1, win2, win3, win4, win5
    cdef mpi.MPI_Aint l_idx_all, l_idx_set_all, l_recvlen, l_setlen, l_idx_recv
    cdef mpi.MPI_Request * barrier_send_req
    cdef mpi.MPI_Request * send_req 
    cdef mpi.MPI_Request recv_req

    # get shm_comm
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    barrier_send_req = <mpi.MPI_Request*>malloc(2*shm_nprocs*s_req)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&node_comm)

    # Bcast psi_t
    mpi.MPI_Bcast(
        &psi_t[0,0],NSW*nk_a,
        mpi.MPI_DOUBLE_COMPLEX,0,node_comm
    )

    # alloc shared variables
    if (shm_id==0):
        l_idx_all = shm_nprocs*7*s_int
        l_idx_set_all = nprocs*6*s_int
        l_recvbuf = shm_nprocs*recvlen
        l_setlen = 2*s_int
        l_idx_recv = 2*shm_nprocs*s_int
    else:
        l_idx_all = 0
        l_idx_set_all = 0
        l_recvbuf = 0
        l_setlen = 0
        l_idx_recv = 0
    
    mpi.MPI_Win_allocate_shared(
        l_idx_all,s_int,mpi.MPI_INFO_NULL,shm_comm,&idx_all,&win0
    )
    mpi.MPI_Win_allocate_shared(
        l_idx_set_all,s_int,mpi.MPI_INFO_NULL,shm_comm,&idx_set_all,&win1
    )
    mpi.MPI_Win_allocate_shared(
        l_recvbuf,s_void,mpi.MPI_INFO_NULL,shm_comm,&recvbuf,&win2
    )
    mpi.MPI_Win_allocate_shared(
        l_setlen,s_int,mpi.MPI_INFO_NULL,shm_comm,&setlen,&win3
    )
    mpi.MPI_Win_allocate_shared(
        l_idx_recv,s_int,mpi.MPI_INFO_NULL,shm_comm,&idx_recv,&win4
    )
    if (shm_id==0):
        idx_set = <int*>malloc(s_int*shm_nprocs*6)
        setlist = <int*>malloc(s_int*nnode)
        setlistr = <int*>malloc(s_int*nnode)
    else:
        mpi.MPI_Win_shared_query(win0,0,&l_idx_all,&s_int,&idx_all)
        mpi.MPI_Win_shared_query(win1,0,&l_idx_set_all,&s_int,&idx_set_all)
        mpi.MPI_Win_shared_query(win2,0,&l_recvbuf,&s_void,&recvbuf)
        mpi.MPI_Win_shared_query(win3,0,&l_setlen,&s_int,&setlen)
        mpi.MPI_Win_shared_query(win4,0,&l_idx_recv,&s_int,&idx_recv)

    for i in range(nmodes):
        for j in range(nq):
            BE2[i,j] = BEfactor[i,j]*BEfactor[i,j]*f1

    for k in range(itraj_max):
        state_a[k] = state_s

    for it in range(NSW):
        for k in range(itraj_max):
            pop_sh_p[state_a[k]] += 1
        mpi.MPI_Reduce(
            &pop_sh_p[0],&pop_sh[it,0],nk_a,
            mpi.MPI_FLOAT,mpi.MPI_SUM,0,c_comm
        )
        mpi.MPI_Reduce(
            &ph_pop_p[0,0],&ph_pop[it,0,0],nmodes*nq,
            mpi.MPI_DOUBLE,mpi.MPI_SUM,0,c_comm
        )
        memset(&pop_sh_p[0],0,nk_a*s_float)
        memset(&ph_pop_p[0,0],0,nmodes*nq*s_double)

        for i in range(itraj_max):
            # get which state_id state belongs to
            GetState(
                myid,shm_id,node_id,nprocs,shm_nprocs,
                state_a[i],idx_all,k_proc,&nowait
            )
            mpi.MPI_Barrier(shm_comm)

            if (shm_id==0):
                memset(idx_recv,0,2*shm_nprocs*s_int)
                qsort(idx_all,shm_nprocs,s_int*7,&Cmp)
                # idx_all to idx_set
                idx_set[0] = idx_all[0]
                idx_set[1] = idx_all[1]
                idx_set[2] = idx_all[2]
                idx_set[3] = idx_all[3]
                idx_set[4] = idx_all[6]
                idx_set[5] = 0
                idx_t = idx_all[0]
                setlen[0] = 1 # number of idx_set
                if (idx_all[4] == node_id):
                    idx_recv[idx_all[5]*2] = -1
                else:
                    idx_recv[idx_all[5]*2] = 1
                idx_recv[idx_all[5]*2+1] = setlen[0]-1
                for j in range(shm_nprocs):
                    if (idx_all[j*7+4] == node_id):
                        idx_recv[idx_all[j*7+5]*2] = -1
                    if (idx_all[j*7] != idx_t):
                        idx_set[setlen[0]*6] = idx_all[j*7]
                        idx_set[setlen[0]*6+1] = idx_all[j*7+1]
                        idx_set[setlen[0]*6+2] = idx_all[j*7+2]
                        idx_set[setlen[0]*6+3] = idx_all[j*7+3]
                        idx_set[setlen[0]*6+4] = idx_all[j*7+6]
                        idx_set[setlen[0]*6+5] = setlen[0]
                        setlen[0] += 1
                        idx_t = idx_all[j*7]
                        if (idx_all[j*7+4] != node_id):
                            idx_recv[idx_all[j*7+5]*2] = 1
                    idx_recv[idx_all[j*7+5]*2+1] = setlen[0]-1
                
                # gather setlen
                if shm_nprocs != nprocs:
                    mpi.MPI_Allgather(
                        setlen,1,mpi.MPI_INT,setlist,
                        1,mpi.MPI_INT,node_comm
                    )
                    for j in range(nnode):
                        setlistr[j] = 0
                    setlen[1] = 0
                    for j in range(nnode):
                        l = setlist[j]
                        setlen[1] += l
                        l *= 6
                        for k in range(j+1,nnode):
                            setlistr[k] += l
                        setlist[j] = l
                    mpi.MPI_Allgatherv(
                        idx_set,setlist[node_id],mpi.MPI_INT,idx_set_all,
                        setlist,setlistr,mpi.MPI_INT,node_comm
                    )
                    qsort(idx_set_all,setlen[1],s_int*6,&Cmp)
                else:
                    setlen[1] = setlen[0]
                    memcpy(idx_set_all,idx_set,setlen[0]*6*s_int)

            mpi.MPI_Barrier(c_comm)

            # get myid send range
            send_s = -1
            for j in range(setlen[1]):
                if (idx_set_all[j*6+3]==myid):
                    send_s = j
                    break
            if (send_s<0):
                send_s = 0
                send_e = 0
            elif (idx_set_all[send_s*6+3]==idx_set_all[setlen[1]*6-3]):
                send_e = setlen[1]
            else:
                for j in range(send_s,setlen[1]):
                    if (idx_set_all[j*6+3]>myid):
                        send_e = j
                        break
            nsend = send_e-send_s

            # prepare recv
            if (idx_recv[shm_id*2]==1):
                k = idx_recv[shm_id*2+1]
                mpi.MPI_Irecv(
                    recvbuf+k*recvlen,recvlen,mpi.MPI_BYTE,
                    mpi.MPI_ANY_SOURCE,k,c_comm,&recv_req
                )
            # prepare send
            p = 0
            if (nsend>0):
                send_req = <mpi.MPI_Request*>malloc(nsend*s_req)
                sendbuf = <char**>malloc(sizeof(char*)*nsend)
                for t0 in range(nsend):
                    sendbuf[t0] = <char*>malloc(recvlen)
                m = 0
                idx_t0 = 0
                idx_t = idx_set_all[send_s*6]
                idx_t1 = send_s
                while (1):
                    k = -1
                    idx_t2 = send_e
                    for j in range(idx_t1,send_e):
                        if (idx_set_all[j*6]>idx_t):
                            idx_t2 = j
                            break
                        else:
                            if (idx_set_all[j*6+2]==node_id):
                                l = idx_set_all[j*6+5]
                                epc_buf = <double complex*>(recvbuf+l*recvlen)
                                kqidx0_buf = <int*>(epc_buf+epclen)
                                k_state_p = idx_set_all[j*6+1]
                                for t0 in range(nmodes):
                                    for t1 in range(nk_a):
                                        epc_buf[t0*nk_a+t1] = epc_a[t0,k_state_p,t1]

                                memcpy(kqidx0_buf,&kqidx_p[k_state_p,0],nk_a*s_int)

                                k = j
                                for n in range(shm_nprocs):
                                    if (idx_recv[n*2+1]==l and n!=shm_id):
                                        mpi.MPI_Isend(&l,0,mpi.MPI_INT,
                                            n,0,shm_comm,&barrier_send_req[p]
                                        )
                                        p += 1
                    if (k>=0):
                        for j in range(idx_t1,idx_t2):
                            if (j!=k):
                                mpi.MPI_Isend(
                                    recvbuf+idx_set_all[k*6+5]*recvlen,recvlen,\
                                    mpi.MPI_BYTE,idx_set_all[j*6+4],\
                                    idx_set_all[j*6+5],c_comm,&send_req[idx_t0]
                                )
                                idx_t0 += 1
                    else:
                        epc_buf = <double complex*>(sendbuf[m])
                        kqidx0_buf = <int *>(epc_buf+epclen)
                        k_state_p = idx_set_all[idx_t1*6+1]
                        for t0 in range(nmodes):
                            for t1 in range(nk_a):
                                epc_buf[t0*nk_a+t1] = epc_a[t0,k_state_p,t1]

                        memcpy(kqidx0_buf,&kqidx_p[k_state_p,0],nk_a*s_int)

                        for j in range(idx_t1,idx_t2):
                            mpi.MPI_Isend(
                                sendbuf[m],recvlen,\
                                mpi.MPI_BYTE,idx_set_all[j*6+4],\
                                idx_set_all[j*6+5],c_comm,&send_req[idx_t0]
                            )
                            idx_t0 += 1
                        m += 1
                    if (idx_t2==send_e):
                        break
                    else:
                        idx_t = idx_set_all[idx_t2*6]
                        idx_t1 = idx_t2

            # waitall
            if (idx_recv[shm_id*2]==1):
                l = idx_recv[shm_id*2+1]
                mpi.MPI_Wait(&recv_req,mpi.MPI_STATUSES_IGNORE)
                for j in range(shm_nprocs):
                    if ((idx_recv[j*2]==0) and (idx_recv[j*2+1]==l)):
                        mpi.MPI_Isend(&l,0,mpi.MPI_INT,
                            j,0,shm_comm,&barrier_send_req[p]
                        )
                        p += 1
            else:
                if (nowait==0):
                    mpi.MPI_Recv(
                        &l,0,mpi.MPI_INT,mpi.MPI_ANY_SOURCE,
                        0,shm_comm,mpi.MPI_STATUSES_IGNORE
                    )

            # calculate H_hop from (epc_p,kqidx0)
            epc_p = <double complex*>(recvbuf+idx_recv[shm_id*2+1]*recvlen)
            kqidx0 = <int *>(epc_p+epclen)
            E0 = energy[state_a[i]]
            for j in range(nk_a):
                H_hop[j] = 0
                prop[j+1] = 0
                dE[j] = energy[j]-E0
            GetHhop(
                nmodes,nk_a,kqidx0,
                energy,phonon,BE2,epc_p,f0,H_hop,dE
            )
            state_a[i] = fsshhop(
                it,nmodes,nk_a,psi_t[it],H_hop,dE,
                epc_p,kqidx0,energy,phonon,BE2,state_a[i],
                f0,f2,KbT1,prop,ph_pop_t,ph_pop_p,LHOLE
            )
            if (p>0):
                mpi.MPI_Waitall(p,barrier_send_req,mpi.MPI_STATUSES_IGNORE)
            if (nsend>0):
                mpi.MPI_Waitall(idx_t0,send_req,mpi.MPI_STATUSES_IGNORE)
                free(send_req)
                for t0 in range(nsend):
                    free(sendbuf[t0])
                free(sendbuf)

    free(ph_pop_t)
    free(prop)
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
    mpi.MPI_Win_free(&win3)
    mpi.MPI_Win_free(&win4)
    free(barrier_send_req)
    if (shm_id==0):
        free(idx_set)
        free(setlist)
        free(setlistr)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void WriteEPC(
    mpi.MPI_Comm c_comm, int myid, int nprocs, str namddir,
    int NSW, double KbT, double sigma, int[::1] k_proc, int[::1] k_proc_num,
    int nk_proc, int nk_a, int nq, int nmodes, int[:,::1] kqidx_p, 
    double complex[:,:,::1] epc_a, double[::1] energy, double[:,::1] phonon,
    double complex[:,::1] phq, double complex[:,::1] phq0, 
    double[:,::1] epc, double[:,::1] epcec, double[:,::1] epcph
):
    cdef int h, i, j, k, qidx, jj
    cdef int nk_min = k_proc[myid]
    cdef double ph, E0, dE, dE0, dE1, expdE
    cdef double nq2 = 1.0/sqrt(nq)
    cdef double KbT1 = 1.0/KbT
    cdef double f0 = -0.5/(sigma*sigma)
    cdef double f1 = sqrt(2*M_PI)/sigma
    cdef double dt1 = 1.0/NSW
    cdef int[::1] epccount = np.zeros((nprocs),dtype=np.int32)
    cdef int[::1] epcdispl = np.zeros((nprocs),dtype=np.int32)
    cdef double[:,::1] epc_t = np.zeros((nk_proc,nk_a),dtype=np.float64)
    cdef double[:,::1] epcec_t = np.zeros((nk_proc,nk_a),dtype=np.float64)
    cdef double[:,::1] epcph_t = np.zeros((nmodes,nq),dtype=np.float64)
    cdef double complex[:,::1] phq_t = \
        np.zeros((nmodes,nq),dtype=np.complex128)
    cdef double epc_tmp2, H_hop_tmp, H_hop
    cdef double complex epc_phq_tmp, epc_tmp, phq_tmp

    for i in range(nprocs):
        epccount[i] = k_proc_num[i]*nk_a
        epcdispl[i] = k_proc[i]*nk_a

    for k in range(nmodes):
        for l in range(nq):
            phq_t[k,l] = phq[k,l]*nq2

    for h in range(NSW):
        for j in range(nk_proc):
            E0 = energy[nk_min+j]
            for k in range(nk_a):
                qidx = kqidx_p[j,k]
                dE = E0 - energy[k]

                epc_phq_tmp = 0
                H_hop = 0
                for i in range(nmodes):
                    ph = phonon[i,qidx]+1e-8
                    dE0 = dE + ph
                    dE1 = dE - ph
                    expdE = exp(dE0*dE0*f0)+exp(dE1*dE1*f0)
                    epc_tmp = epc_a[i,j,k]
                    epc_tmp2 = creal(conj(epc_tmp)*epc_tmp)
                    phq_tmp = phq_t[i,qidx]
                    phq_t1 = creal(conj(phq_tmp)*phq_tmp)
                    epc_phq_tmp += epc_tmp*creal(phq_tmp)
                    H_hop_tmp = epc_tmp2*expdE*phq_t1
                    epcph_t[i,qidx] += H_hop_tmp
                    H_hop += H_hop_tmp

                epc_t[j,k] += cabs(epc_phq_tmp)
                epcec_t[j,k] += H_hop

        for i in range(nmodes):
            for j in range(nq):
                phq_t[i,j] *= phq0[i,j]

    for i in range(nk_proc):
        for j in range(nk_a):
            epc_t[i,j] *= (2*dt1)
            epcec_t[i,j] *= (dt1*f1)
    for i in range(nmodes):
        for j in range(nq):
            epcph_t[i,j] *= (dt1*f1)

    mpi.MPI_Gatherv(
        &epc_t[0,0],epccount[myid],mpi.MPI_DOUBLE,&epc[0,0],
        &epccount[0],&epcdispl[0],mpi.MPI_DOUBLE,0,c_comm
    )
    mpi.MPI_Gatherv(
        &epcec_t[0,0],epccount[myid],mpi.MPI_DOUBLE,&epcec[0,0],
        &epccount[0],&epcdispl[0],mpi.MPI_DOUBLE,0,c_comm
    )
    mpi.MPI_Reduce(
        &epcph_t[0,0],&epcph[0,0],nmodes*nq,
        mpi.MPI_DOUBLE,mpi.MPI_SUM,0,c_comm
    )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void FsshOut(
    int nmodes, int NSW, int nk_a, int nq, int ntraj, 
    double complex[:,::1] psi_t,
    double[::1] energy, double[:,::1] phonon, 
    double[::1] e_psi, float[:,::1] pop_sh,
    double[::1] e_sh,
    double[:,:,::1] ph_pop, double[:,::1] e_ph
):
    cdef int i, j, k
    cdef double psi2

    for i in range(NSW):
        for j in range(nk_a):
            psi2 = creal(conj(psi_t[i,j])*psi_t[i,j])
            e_psi[i] += psi2*energy[j]
            pop_sh[i,j] /= ntraj
            e_sh[i] += pop_sh[i,j]*energy[j]
        for j in range(nmodes):
            for k in range(nq):
                ph_pop[i,j,k] = ph_pop[i,j,k]/ntraj+1e-30
                e_ph[i,j] += ph_pop[i,j,k]*phonon[j,k]


@cython.boundscheck(False)
@cython.wraparound(False)
def fssh(
    MPI.Comm comm, str namddir, int sample, int step_s, int state_s,
    int NTRAJ, int NSW, int NELM, double KbT, double edt, double hbar,
    double sigma, double dt, int[::1] k_proc, int[::1] k_proc_num, 
    int nk_a, int nq, int nmodes, int[:,::1] kqidx_p,
    double complex[:,:,::1] epc_a, double[::1] energy, double[:,::1] phonon,
    bint LHOLE
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    #cdef mpi.MPI_Win win
    cdef mpi.MPI_Comm shm_comm
    #cdef mpi.MPI_Aint len_psi
    cdef int myid, nprocs, ierr, shm_id
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    cdef int len_psi
    cdef int s_dcplx = sizeof(double complex)
    cdef int ntraj_max = <int>(NTRAJ/nprocs+1)*nprocs
    cdef int nk_proc = k_proc_num[myid]
    cdef double[:,::1] epc, epcec, epcph
    cdef double[:,::1] BEfactor = \
                np.zeros((nmodes,nq),dtype=np.float64)
    cdef double complex[:,::1] phq0 = \
                np.zeros((nmodes,nq),dtype=np.complex128)
    cdef double complex[:,::1] phq = \
                np.zeros((nmodes,nq),dtype=np.complex128)
    cdef double complex[:,::1] psi_t
    cdef double[::1] e_psi, e_sh 
    cdef float[:,::1] pop_sh
    cdef double starttime, endtime, starttime1, endtime1

    # init
    # get shm_comm
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    # alloc psi_t
    shm_comm_py = comm.Split_type(MPI.COMM_TYPE_SHARED)
    if (shm_id==0):
        len_psi = NSW*nk_a*s_dcplx
    else:
        len_psi = 0
    win = MPI.Win.Allocate_shared(len_psi,s_dcplx,comm=shm_comm_py)
    buf,s_dcplx = win.Shared_query(0)
    psi_t = np.ndarray(buffer=buf,dtype=np.complex128,shape=(NSW,nk_a))

    # write EPTXT, EPECTXT, EPPHTXT
    if myid == 0:
        epc = np.zeros((nk_a,nk_a),dtype=np.float64)
        epcec = np.zeros((nk_a,nk_a),dtype=np.float64)
        epcph = np.zeros((nmodes,nq),dtype=np.float64)
    else:
        epc = np.zeros((0,0),dtype=np.float64)
        epcec = np.zeros((0,0),dtype=np.float64)
        epcph = np.zeros((0,0),dtype=np.float64)

    starttime = mpi.MPI_Wtime()
    # Init phq0[nmode,nq] and phq
    GetPhQ(
        phonon,nmodes,nq,step_s,
        BEfactor,dt,hbar,KbT,phq0,phq
    )
    WriteEPC(
        c_comm,myid,nprocs,namddir,NSW,KbT,sigma,
        k_proc,k_proc_num,nk_proc,nk_a,nq,nmodes,kqidx_p,
        epc_a,energy,phonon,phq,phq0,epc,epcec,epcph
    )
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        np.save(namddir+'/epc-%d.npy'%(sample),epc)
        np.save(namddir+'/epcec-%d.npy'%(sample),epcec)
        np.save(namddir+'/epcph-%d.npy'%(sample),epcph)
        printf("Sample %d, EPCTXT time: %.6fs.\n",sample,endtime-starttime)
    # run TDDFT, get expansion coefficients psi_t
    starttime0 = mpi.MPI_Wtime()
    fssh_psi(
        c_comm,nprocs,myid,NSW,NELM,edt,hbar,state_s,
        nk_proc,nk_a,nq,nmodes,k_proc,k_proc_num,
        kqidx_p,epc_a,energy,phq,phq0,psi_t
    )
    endtime0 = mpi.MPI_Wtime()
    if myid == 0:
        np.save(namddir+'/psi-%d.npy'%(sample),psi_t)
        printf("Sample %d, TimeProp time: %.6fs.\n",sample,endtime0-starttime0)
    # run FSSH
    #psi_t = np.load(namddir+'/psi-%d.npy'%(sample))
    if myid == 0:
        pop_sh = np.zeros((NSW,nk_a),dtype=np.float32)
        ph_pop = np.zeros((NSW,nmodes,nq),dtype=np.float64)
        e_psi = np.zeros((NSW),dtype=np.float64)
        e_sh = np.zeros((NSW),dtype=np.float64)
        e_ph = np.zeros((NSW,nmodes),dtype=np.float64)
    else:
        pop_sh = np.zeros((0,0),dtype=np.float32)
        ph_pop = np.zeros((0,0,0),dtype=np.float64)

    starttime1 = mpi.MPI_Wtime()
    fssh_sh(
        c_comm,nprocs,myid,NTRAJ,NSW,KbT,
        hbar,sigma,dt,state_s,nk_proc,nk_a,nq,
        nmodes,k_proc,kqidx_p,epc_a,energy,
        phonon,BEfactor,psi_t,pop_sh,ph_pop,LHOLE
    )
    endtime1 = mpi.MPI_Wtime()
    if myid == 0:
        FsshOut(
            nmodes,NSW,nk_a,nq,ntraj_max,psi_t,energy,phonon,
            e_psi,pop_sh,e_sh,ph_pop,e_ph
        )
        np.save(namddir+'/fssh_e_psi-%d.npy'%(sample),e_psi)
        np.save(namddir+'/fssh_pop_sh-%d.npy'%(sample),pop_sh)
        np.save(namddir+'/fssh_e_sh-%d.npy'%(sample),e_sh)
        np.save(namddir+'/fssh_pop_ph-%d.npy'%(sample),ph_pop)
        np.save(namddir+'/fssh_e_ph-%d.npy'%(sample),e_ph)
        printf("Sample %d, FSSH time: %.6fs.\n",sample,endtime1-starttime1)

    MPI.Win.Free(win)
