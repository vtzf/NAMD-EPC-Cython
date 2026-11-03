#cython: language_level=3
#cython: cdivision=True

cimport cython
from epc cimport *
from mkl_cblas cimport *
from mkl_spblas cimport *


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void hamilveck_batch(
    bint use_perjk_flag, int nb_b1, int ncol_b1,
    int nm_num, int ncell, int nspin, int nspin2, int norbital, int nbands,
    int norb2, int norbnb, long norbnb_l, long hvlen_b1,
    int* kidx_b1, double complex* expikR_b1,
    double complex[:,:,::1] dhamil, double complex[:,:,::1] bandveck,
    int[:,::1] key_num, int[:,::1] key_num_s, int[:,::1] key_info,
    sparse_matrix_t* csrH_jk, int* tag, MKL_INT* csr_rowptr, MKL_INT* csr_cidx,
    double complex* bandveck_b1, double complex* dhveck_b1, double complex* dhexpikR,
    double complex* hamilveck, matrix_descr descrH
):
    cdef int i, j, k, l, m, n, r, h, jk, spin, s_r, s_c, kidx, dhlen
    cdef long hbase, len_hv = sizeof(double complex)*hvlen_b1
    cdef MKL_Complex16* hamilveck_t
    cdef sparse_matrix_t csrH
    cdef double complex c1 = 1.0
    cdef double complex c0 = 0.0
    cdef double complex ph_b1, dhval
    cdef MKL_Complex16 c1mkl, c0mkl
    c1mkl.real = 1.0; c1mkl.imag = 0.0
    c0mkl.real = 0.0; c0mkl.imag = 0.0

    memset(hamilveck,0,len_hv*nb_b1)
    if use_perjk_flag:
        for i in range(nspin):
            for j in range(nb_b1):
                kidx = kidx_b1[j]
                for k in range(norbital):
                    for l in range(nbands):
                        bandveck_b1[i*norbnb_l*nb_b1+k*ncol_b1+j*nbands+l] \
                        = bandveck[kidx,i*norbital+k,l]
        for i in range(nm_num):
            for j in range(ncell):
                for k in range(ncell):
                    jk = j*ncell+k
                    n = key_num[jk,1]
                    if n == 0: continue
                    m = key_num[jk,3]
                    for spin in range(nspin2):
                        s_r = <int>(spin/nspin)
                        s_c = spin%nspin
                        if n == norb2:
                            cblas_zgemm3m(
                                CblasRowMajor,CblasNoTrans,CblasNoTrans,
                                norbital,ncol_b1,norbital,
                                &c1,&dhamil[i,spin,m],norbital,
                                &bandveck_b1[s_r*norbnb_l*nb_b1],ncol_b1,
                                &c0,dhveck_b1,ncol_b1
                            )
                        else:
                            mkl_sparse_z_mm(
                                SPARSE_OPERATION_NON_TRANSPOSE,c1mkl,
                                csrH_jk[(jk*nm_num+i)*nspin2+spin],
                                descrH,SPARSE_LAYOUT_ROW_MAJOR,
                                <MKL_Complex16*>(&bandveck_b1[s_r*norbnb_l*nb_b1]),
                                ncol_b1,ncol_b1,c0mkl,
                                <MKL_Complex16*>dhveck_b1,ncol_b1
                            )
                        for h in range(nb_b1):
                            ph_b1 = expikR_b1[h*ncell+k]
                            hbase = h*hvlen_b1+((i*ncell+j)*nspin+s_c)*norbnb_l
                            for l in range(norbital):
                                for r in range(nbands):
                                    hamilveck[hbase+l*nbands+r] \
                                    += ph_b1*dhveck_b1[l*ncol_b1+h*nbands+r]
    else:
        for i in range(nm_num):
            for j in range(ncell):
                if tag[j] > 0:
                    dhlen = key_num_s[j,0]
                    memset(dhexpikR,0,nb_b1*nspin2*dhlen*sizeof(double complex))
                    if nb_b1 == 1:
                        for k in range(ncell):
                            jk = j*ncell+k
                            n = key_num[jk,1]
                            if n > 0:
                                m = key_num[jk,3]
                                ph_b1 = expikR_b1[k]
                                if n == norb2:
                                    for spin in range(nspin2):
                                        for l in range(norb2):
                                            dhexpikR[spin*dhlen+l] \
                                            += dhamil[i,spin,m+l]*ph_b1
                                else:
                                    for spin in range(nspin2):
                                        for l in range(n):
                                            dhexpikR[spin*dhlen+key_info[m+l,1]] \
                                            += dhamil[i,spin,m+l]*ph_b1
                    else:
                        for k in range(ncell):
                            jk = j*ncell+k
                            n = key_num[jk,1]
                            if n > 0:
                                m = key_num[jk,3]
                                if n == norb2:
                                    for spin in range(nspin2):
                                        for l in range(norb2):
                                            dhval = dhamil[i,spin,m+l]
                                            for h in range(nb_b1):
                                                dhexpikR[(h*nspin2+spin)*dhlen+l] \
                                                += dhval*expikR_b1[h*ncell+k]
                                else:
                                    for spin in range(nspin2):
                                        for l in range(n):
                                            dhval = dhamil[i,spin,m+l]
                                            r = key_info[m+l,1]
                                            for h in range(nb_b1):
                                                dhexpikR[(h*nspin2+spin)*dhlen+r] \
                                                += dhval*expikR_b1[h*ncell+k]
                    for h in range(nb_b1):
                        hamilveck_t = <MKL_Complex16*>(
                            &hamilveck[h*hvlen_b1+(i*ncell+j)*nspin*norbnb_l]
                        )
                        if tag[j] == 1:
                            for spin in range(nspin2):
                                s_r = <int>(spin/nspin)
                                s_c = spin%nspin
                                mkl_sparse_z_create_csr(
                                    &csrH,SPARSE_INDEX_BASE_ZERO,norbital,norbital,
                                    &csr_rowptr[j*(norbital+1)],
                                    &csr_rowptr[j*(norbital+1)+1],
                                    &csr_cidx[key_num_s[j,1]],
                                    <MKL_Complex16*>(&dhexpikR[(h*nspin2+spin)*dhlen])
                                )
                                mkl_sparse_optimize(csrH)
                                mkl_sparse_z_mm(
                                    SPARSE_OPERATION_NON_TRANSPOSE,c1mkl,
                                    csrH,descrH,SPARSE_LAYOUT_ROW_MAJOR,
                                    <MKL_Complex16*>(&bandveck[kidx_b1[h],s_r*norbital,0]),
                                    nbands,nbands,c1mkl,&hamilveck_t[s_c*norbnb],nbands
                                )
                                mkl_sparse_destroy(csrH)
                        else:
                            for spin in range(nspin2):
                                s_r = <int>(spin/nspin)
                                s_c = spin%nspin
                                cblas_zgemm3m(
                                    CblasRowMajor,CblasNoTrans,CblasNoTrans,norbital,
                                    nbands,norbital,&c1,
                                    &dhexpikR[(h*nspin2+spin)*dhlen],norbital,
                                    &bandveck[kidx_b1[h],s_r*norbital,0],nbands,&c1,
                                    &hamilveck_t[s_c*norbnb],nbands
                                )


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void epckq_batch(
    double complex* hamilveck, double complex[:,:,::1] bandveck,
    double complex[:,:,::1] phvecval, double complex* expikR,
    double complex* dhvexpikR, double complex* vdhvexpikR, double complex* epcq,
    int[:,::1] R_list, int[::1] nq, int nspin, int nmodes, int nm_num, int ncell,
    int nbands, int norbital, int nbands2, MKL_INT norbnb, int bsize,
    int* kpidx_arr, int* phidx_arr
):
    cdef int b, i, j, k, M = nspin*norbnb
    cdef int kpidx, kpidx_x, kpidx_y, kpidx_z, kpidx_xy
    cdef double RKp, kpx, kpy, kpz
    cdef double complex pi2j = M_PI*2j
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef double complex phvv

    for b in range(bsize):
        kpidx = kpidx_arr[b]
        kpidx_z = kpidx%nq[2]; kpidx_xy = kpidx/nq[2]
        kpidx_y = kpidx_xy%nq[1]; kpidx_x = kpidx_xy/nq[1]
        kpx = (<double>kpidx_x)/(<double>nq[0])
        kpy = (<double>kpidx_y)/(<double>nq[1])
        kpz = (<double>kpidx_z)/(<double>nq[2])
        for j in range(ncell):
            RKp = R_list[j,0]*kpx+R_list[j,1]*kpy+R_list[j,2]*kpz
            expikR[b*ncell+j] = cexp(-1.0*pi2j*RKp)
    for i in range(nm_num):
        cblas_zgemm(
            CblasRowMajor,CblasNoTrans,CblasNoTrans,
            bsize,M,ncell,&c1,expikR,ncell,
            &hamilveck[i*ncell*M],M,&c0,dhvexpikR,M
        )
        for b in range(bsize):
            cblas_zgemm3m(
                CblasRowMajor,CblasConjTrans,CblasNoTrans,
                nbands,nbands,nspin*norbital,&c1,
                &bandveck[kpidx_arr[b],0,0],nbands,
                &dhvexpikR[b*M],nbands,&c0,vdhvexpikR,nbands
            )
            for j in range(nmodes):
                phvv = phvecval[phidx_arr[b],i,j]
                for k in range(nbands2):
                    epcq[b*nmodes*nbands2+j*nbands2+k] += vdhvexpikR[k]*phvv


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void epckq_p_batch(
    double complex* hamilveck, double complex[:,::1] bandveck,
    double complex[:,:,::1] phvecval, double complex* expikR,
    double complex* dhvexpikR, double complex* epcq, int[:,::1] R_list,
    int[::1] nq, int nspin, int nmodes, int nm_num, int ncell, int norbital,
    int bsize, int* kpidx_arr, int* kgrid_arr, int* phidx_arr
):
    cdef int b, i, j, kg, kgx, kgy, kgz, kgxy, M = nspin*norbital
    cdef double RKp, kpx, kpy, kpz
    cdef double complex pi2j = M_PI*2j
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef double complex vdot

    for b in range(bsize):
        kg = kgrid_arr[b]
        kgz = kg%nq[2]; kgxy = kg/nq[2]
        kgy = kgxy%nq[1]; kgx = kgxy/nq[1]
        kpx = (<double>kgx)/(<double>nq[0])
        kpy = (<double>kgy)/(<double>nq[1])
        kpz = (<double>kgz)/(<double>nq[2])
        for j in range(ncell):
            RKp = R_list[j,0]*kpx+R_list[j,1]*kpy+R_list[j,2]*kpz
            expikR[b*ncell+j] = cexp(-1.0*pi2j*RKp)
    for i in range(nm_num):
        cblas_zgemm(
            CblasRowMajor,CblasNoTrans,CblasNoTrans,
            bsize,M,ncell,&c1,expikR,ncell,
            &hamilveck[i*ncell*M],M,&c0,dhvexpikR,M
        )
        for b in range(bsize):
            cblas_zdotc_sub(
                M,&bandveck[kpidx_arr[b],0],1,&dhvexpikR[b*M],1,&vdot
            )
            for j in range(nmodes):
                epcq[b*nmodes+j] += vdot*phvecval[phidx_arr[b],i,j]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void hamilveck_batch_p(
    bint use_perjk_flag, int nb_b1,
    int nm_num, int ncell, int nspin, int nspin2, int norbital,
    int norb2, long hvlen_b1,
    int* kidx_b1, double complex* expikR_b1,
    double complex[:,:,::1] dhamil, double complex[:,::1] bandveck,
    int[:,::1] key_num, int[:,::1] key_num_s, int[:,::1] key_info,
    sparse_matrix_t* csrH_jk, int* tag, MKL_INT* csr_rowptr, MKL_INT* csr_cidx,
    double complex* bandveck_b1, double complex* dhveck_b1, double complex* dhexpikR,
    double complex* hamilveck, matrix_descr descrH
):
    cdef int i, j, k, l, m, n, r, h, jk, spin, s_r, s_c, kidx, dhlen
    cdef long hbase, len_hv = sizeof(double complex)*hvlen_b1
    cdef MKL_Complex16* hamilveck_t
    cdef sparse_matrix_t csrH
    cdef double complex c1 = 1.0
    cdef double complex c0 = 0.0
    cdef double complex ph_b1, dhval
    cdef MKL_Complex16 c1mkl, c0mkl
    c1mkl.real = 1.0; c1mkl.imag = 0.0
    c0mkl.real = 0.0; c0mkl.imag = 0.0

    memset(hamilveck,0,len_hv*nb_b1)
    if use_perjk_flag:
        for i in range(nspin):
            for j in range(nb_b1):
                kidx = kidx_b1[j]
                for k in range(norbital):
                    bandveck_b1[i*norbital*nb_b1+k*nb_b1+j] \
                    = bandveck[kidx,i*norbital+k]
        for i in range(nm_num):
            for j in range(ncell):
                for k in range(ncell):
                    jk = j*ncell+k
                    n = key_num[jk,1]
                    if n == 0: continue
                    m = key_num[jk,3]
                    for spin in range(nspin2):
                        s_r = <int>(spin/nspin)
                        s_c = spin%nspin
                        if n == norb2:
                            cblas_zgemm3m(
                                CblasRowMajor,CblasNoTrans,CblasNoTrans,
                                norbital,nb_b1,norbital,
                                &c1,&dhamil[i,spin,m],norbital,
                                &bandveck_b1[s_r*norbital*nb_b1],nb_b1,
                                &c0,dhveck_b1,nb_b1
                            )
                        else:
                            mkl_sparse_z_mm(
                                SPARSE_OPERATION_NON_TRANSPOSE,c1mkl,
                                csrH_jk[(jk*nm_num+i)*nspin2+spin],
                                descrH,SPARSE_LAYOUT_ROW_MAJOR,
                                <MKL_Complex16*>(&bandveck_b1[s_r*norbital*nb_b1]),
                                nb_b1,nb_b1,c0mkl,
                                <MKL_Complex16*>dhveck_b1,nb_b1
                            )
                        for h in range(nb_b1):
                            ph_b1 = expikR_b1[h*ncell+k]
                            hbase = h*hvlen_b1+((i*ncell+j)*nspin+s_c)*norbital
                            for l in range(norbital):
                                hamilveck[hbase+l] += ph_b1*dhveck_b1[l*nb_b1+h]
    else:
        for i in range(nm_num):
            for j in range(ncell):
                if tag[j] > 0:
                    dhlen = key_num_s[j,0]
                    memset(dhexpikR,0,nb_b1*nspin2*dhlen*sizeof(double complex))
                    if nb_b1 == 1:
                        for k in range(ncell):
                            jk = j*ncell+k
                            n = key_num[jk,1]
                            if n > 0:
                                m = key_num[jk,3]
                                ph_b1 = expikR_b1[k]
                                if n == norb2:
                                    for spin in range(nspin2):
                                        for l in range(norb2):
                                            dhexpikR[spin*dhlen+l] \
                                            += dhamil[i,spin,m+l]*ph_b1
                                else:
                                    for spin in range(nspin2):
                                        for l in range(n):
                                            dhexpikR[spin*dhlen+key_info[m+l,1]] \
                                            += dhamil[i,spin,m+l]*ph_b1
                    else:
                        for k in range(ncell):
                            jk = j*ncell+k
                            n = key_num[jk,1]
                            if n > 0:
                                m = key_num[jk,3]
                                if n == norb2:
                                    for spin in range(nspin2):
                                        for l in range(norb2):
                                            dhval = dhamil[i,spin,m+l]
                                            for h in range(nb_b1):
                                                dhexpikR[(h*nspin2+spin)*dhlen+l] \
                                                += dhval*expikR_b1[h*ncell+k]
                                else:
                                    for spin in range(nspin2):
                                        for l in range(n):
                                            dhval = dhamil[i,spin,m+l]
                                            r = key_info[m+l,1]
                                            for h in range(nb_b1):
                                                dhexpikR[(h*nspin2+spin)*dhlen+r] \
                                                += dhval*expikR_b1[h*ncell+k]
                    for h in range(nb_b1):
                        hamilveck_t = <MKL_Complex16*>(
                            &hamilveck[h*hvlen_b1+(i*ncell+j)*nspin*norbital]
                        )
                        if tag[j] == 1:
                            for spin in range(nspin2):
                                s_r = <int>(spin/nspin)
                                s_c = spin%nspin
                                mkl_sparse_z_create_csr(
                                    &csrH,SPARSE_INDEX_BASE_ZERO,norbital,norbital,
                                    &csr_rowptr[j*(norbital+1)],
                                    &csr_rowptr[j*(norbital+1)+1],
                                    &csr_cidx[key_num_s[j,1]],
                                    <MKL_Complex16*>(&dhexpikR[(h*nspin2+spin)*dhlen])
                                )
                                mkl_sparse_optimize(csrH)
                                mkl_sparse_z_mv(
                                    SPARSE_OPERATION_NON_TRANSPOSE,c1mkl,csrH,
                                    descrH,<MKL_Complex16*>(&bandveck[kidx_b1[h],s_r*norbital]),
                                    c1mkl,&hamilveck_t[s_c*norbital]
                                )
                                mkl_sparse_destroy(csrH)
                        else:
                            for spin in range(nspin2):
                                s_r = <int>(spin/nspin)
                                s_c = spin%nspin
                                cblas_zgemv(
                                    CblasRowMajor,CblasNoTrans,norbital,
                                    norbital,&c1,&dhexpikR[(h*nspin2+spin)*dhlen],norbital,
                                    &bandveck[kidx_b1[h],s_r*norbital],1,&c1,
                                    &hamilveck_t[s_c*norbital],1
                                )


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepc(
    MPI.Comm comm,
    int nspin, int nmodes, int nm_num, int nm_min, int norbital, int nbands,
    int ncell, int knum, int[::1] nq, int[:,::1] R_list, int nkpath,
    double[:,::1] kpath, int[:,::1] key_num, int[:,::1] key_num_s,
    int[:,::1] key_info, int[::1] key_info_s, double complex[:,:,::1] dhamil,
    double complex[:,:,::1] bandveck, double complex[:,:,::1] phvecval,
    int[::1] kproc, int[::1] kproc_num, double complex[:,::1] epc_t,
    bint use_perjk_flag, int[::1] col_idx_jk, int[::1] row_ptr_jk, int NB_b1, int NB_b2
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int nprocs, myid, ierr, h, i, j, k, l, m, n, k1
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, kidx, \
             qidx_x, qidx_y, qidx_z, qidx_xy, \
             jk, knum_p, knum_s, spin, r, matlen, \
             nb_b1, nb_b2, k1_start, k1_hi, h_end, ncol_b1
    cdef int nbands2 = nbands*nbands
    cdef int norb2 = norbital*norbital
    cdef int nspin2 = nspin*nspin
    cdef int* tag = NULL
    cdef MKL_INT* csr_rowptr = NULL
    cdef MKL_INT* csr_cidx = NULL
    cdef sparse_matrix_t* csrH_jk = NULL
    cdef double kx, ky, kz, RK, starttime, endtime
    cdef double t0, build_time = 0.0, epckq_time = 0.0
    cdef double complex* hamilveck
    cdef double complex* expikR_b2
    cdef double complex* expikR_b1
    cdef double complex* dhexpikR = NULL
    cdef double complex* dhvexpikR_b2
    cdef double complex* vdhvexpikR
    cdef double complex* bandveck_b1 = NULL
    cdef double complex* dhveck_b1 = NULL
    cdef int* kpidx_b2
    cdef int* phidx_b2
    cdef int* kidxx_b1
    cdef int* kidxy_b1
    cdef int* kidxz_b1
    cdef int* kidx_b1
    cdef long len_hv, hvlen_b1
    cdef long norbnb_l = norbital*nbands
    cdef int norbnb = norbital*nbands
    cdef matrix_descr descrH
    cdef double complex pi2j = M_PI*2j

    starttime = mpi.MPI_Wtime()
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    descrH.type = SPARSE_MATRIX_TYPE_GENERAL

    if NB_b2>knum: NB_b2 = knum
    if NB_b1 < 1: NB_b1 = 1
    knum_p = kproc_num[myid]
    knum_s = kproc[myid]
    k1_hi = <int>((knum_s+knum_p-1)/knum)

    if use_perjk_flag:
        csrH_jk = <sparse_matrix_t*>malloc(sizeof(sparse_matrix_t)*ncell*ncell*nm_num*nspin2)
        for j in range(ncell):
            for k in range(ncell):
                jk = j*ncell+k
                n = key_num[jk,1]
                if n == 0 or n == norb2:
                    continue
                m = key_num[jk,3]
                for i in range(nm_num):
                    for spin in range(nspin2):
                        mkl_sparse_z_create_csr(
                            &csrH_jk[(jk*nm_num+i)*nspin2+spin],
                            SPARSE_INDEX_BASE_ZERO,norbital,norbital,
                            <MKL_INT*>(&row_ptr_jk[jk*(norbital+1)]),
                            <MKL_INT*>(&row_ptr_jk[jk*(norbital+1)+1]),
                            <MKL_INT*>(&col_idx_jk[m]),
                            <MKL_Complex16*>(&dhamil[i,spin,m])
                        )
                        mkl_sparse_optimize(csrH_jk[(jk*nm_num+i)*nspin2+spin])
    else:
        tag = <int*>calloc(sizeof(int),ncell)
        for i in range(ncell):
            if key_num_s[i,0]>0:
                tag[i] = 1
                if key_num_s[i,0] == norb2: tag[i] = 2
        csr_rowptr = <MKL_INT*>malloc(sizeof(MKL_INT)*ncell*(norbital+1))
        csr_cidx = <MKL_INT*>malloc(sizeof(MKL_INT)*key_num_s[ncell,1])
        for i in range(key_num_s[ncell,1]):
            csr_cidx[i] = key_info_s[i]%norbital
        for j in range(ncell):
            for r in range(norbital+1):
                csr_rowptr[j*(norbital+1)+r] = 0
            for i in range(key_num_s[j,0]):
                r = <int>(key_info_s[key_num_s[j,1]+i]/norbital)
                csr_rowptr[j*(norbital+1)+r+1] += 1
            for r in range(norbital):
                csr_rowptr[j*(norbital+1)+r+1] += csr_rowptr[j*(norbital+1)+r]
        matlen = key_num_s[0,0]
        for i in range(1,ncell):
            if matlen < key_num_s[i,0]:
                matlen = key_num_s[i,0]
        dhexpikR = <double complex*>malloc(sizeof(double complex)*NB_b1*nspin2*matlen)

    hvlen_b1 = nm_num*ncell*norbnb_l*nspin
    len_hv = sizeof(double complex)*hvlen_b1
    expikR_b2 = <double complex*>malloc(sizeof(double complex)*NB_b2*ncell)
    expikR_b1 = <double complex*>malloc(sizeof(double complex)*NB_b1*ncell)
    dhvexpikR_b2 = <double complex*>malloc(sizeof(double complex)*NB_b2*nspin*norbnb)
    vdhvexpikR = <double complex*>malloc(sizeof(double complex)*nbands2)
    hamilveck = <double complex*>malloc(len_hv*NB_b1)
    kpidx_b2 = <int*>malloc(sizeof(int)*NB_b2)
    phidx_b2 = <int*>malloc(sizeof(int)*NB_b2)
    kidxx_b1 = <int*>malloc(sizeof(int)*NB_b1)
    kidxy_b1 = <int*>malloc(sizeof(int)*NB_b1)
    kidxz_b1 = <int*>malloc(sizeof(int)*NB_b1)
    kidx_b1 = <int*>malloc(sizeof(int)*NB_b1)
    if use_perjk_flag:
        bandveck_b1 = <double complex*>malloc(sizeof(double complex)*nspin*norbnb_l*NB_b1)
        dhveck_b1 = <double complex*>malloc(sizeof(double complex)*norbnb_l*NB_b1)

    for k1_start in range(<int>(knum_s/knum),k1_hi+1,NB_b1):
        nb_b1 = NB_b1
        if k1_start+nb_b1-1 > k1_hi: nb_b1 = k1_hi-k1_start+1
        ncol_b1 = nbands*nb_b1
        t0 = mpi.MPI_Wtime()
        for i in range(nb_b1):
            k1 = k1_start+i
            if nkpath < knum:
                kidx_x = <int>round(nq[0]*kpath[k1,0])
                kidx_y = <int>round(nq[1]*kpath[k1,1])
                kidx_z = <int>round(nq[2]*kpath[k1,2])
            else:
                kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
                kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
            kidx = (kidx_x*nq[1]+kidx_y)*nq[2]+kidx_z
            kidxx_b1[i] = kidx_x; kidxy_b1[i] = kidx_y
            kidxz_b1[i] = kidx_z; kidx_b1[i] = kidx
            kx = (<double>(kidx_x))/(<double>(nq[0]))
            ky = (<double>(kidx_y))/(<double>(nq[1]))
            kz = (<double>(kidx_z))/(<double>(nq[2]))
            for k in range(ncell):
                RK = R_list[k,0]*kx+R_list[k,1]*ky+R_list[k,2]*kz
                expikR_b1[i*ncell+k] = cexp(pi2j*RK)
        hamilveck_batch(
            use_perjk_flag,nb_b1,ncol_b1,nm_num,ncell,nspin,nspin2,norbital,nbands,
            norb2,norbnb,norbnb_l,hvlen_b1,kidx_b1,expikR_b1,dhamil,bandveck,
            key_num,key_num_s,key_info,csrH_jk,tag,csr_rowptr,csr_cidx,
            bandveck_b1,dhveck_b1,dhexpikR,hamilveck,descrH
        )
        build_time += mpi.MPI_Wtime()-t0

        for i in range(nb_b1):
            k1 = k1_start+i
            kidx_x = kidxx_b1[i]; kidx_y = kidxy_b1[i]; kidx_z = kidxz_b1[i]
            h = k1*knum-knum_s
            if h < 0: h = 0
            h_end = (k1+1)*knum-knum_s
            if h_end > knum_p: h_end = knum_p
            for h in range(h,h_end,NB_b2):
                nb_b2 = NB_b2
                if h+nb_b2 > h_end: nb_b2 = h_end-h
                for j in range(nb_b2):
                    k = (h+knum_s+j)%knum
                    qidx_z = k%nq[2]; qidx_xy = k/nq[2]
                    qidx_y = qidx_xy%nq[1]; qidx_x = qidx_xy/nq[1]
                    kpidx_b2[j] = (((kidx_x+qidx_x)%nq[0])*nq[1] \
                                 +(kidx_y+qidx_y)%nq[1])*nq[2]+(kidx_z+qidx_z)%nq[2]
                    phidx_b2[j] = k
                t0 = mpi.MPI_Wtime()
                epckq_batch(
                    &hamilveck[i*hvlen_b1],bandveck,phvecval,
                    expikR_b2,dhvexpikR_b2,vdhvexpikR,&epc_t[h,0],R_list,nq,
                    nspin,nmodes,nm_num,ncell,nbands,norbital,nbands2,
                    norbnb,nb_b2,kpidx_b2,phidx_b2
                )
                epckq_time += mpi.MPI_Wtime()-t0

    if use_perjk_flag:
        for j in range(ncell):
            for k in range(ncell):
                jk = j*ncell+k
                n = key_num[jk,1]
                if n == 0 or n == norb2:
                    continue
                for i in range(nm_num):
                    for spin in range(nspin2):
                        mkl_sparse_destroy(csrH_jk[(jk*nm_num+i)*nspin2+spin])
        free(csrH_jk)
        free(bandveck_b1)
        free(dhveck_b1)
    else:
        free(dhexpikR)
        free(csr_rowptr)
        free(csr_cidx)
        free(tag)

    free(expikR_b2)
    free(expikR_b1)
    free(dhvexpikR_b2)
    free(vdhvexpikR)
    free(hamilveck)
    free(kpidx_b2)
    free(phidx_b2)
    free(kidxx_b1)
    free(kidxy_b1)
    free(kidxz_b1)
    free(kidx_b1)

    mpi.MPI_Barrier(c_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_s time in mode[%4d:%4d], NB_b1[%4d] NB_b2[%4d]:%12.4fs. (build:%9.4fs epckq:%9.4fs)\n",nm_min,nm_min+nm_num,NB_b1,NB_b2,endtime-starttime,build_time,epckq_time)


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepc_q(
    MPI.Comm comm,
    int nspin, int nmodes, int nm_num, int nm_min, int norbital, int nbands,
    int ncell, int knum, int[::1] nq, int[:,::1] R_list, int nqpath,
    double[:,::1] qpath, int[:,::1] key_num, int[:,::1] key_num_s,
    int[:,::1] key_info, int[::1] key_info_s, double complex[:,:,::1] dhamil,
    double complex[:,:,::1] bandveck, double complex[:,:,::1] phvecval,
    int[::1] kproc, int[::1] kproc_num, double complex[:,::1] epc_t,
    bint use_perjk_flag, int[::1] col_idx_jk, int[::1] row_ptr_jk, int NB_b1, int NB_b2
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int nprocs, myid, ierr, h, i, j, k, l, m, n, k1
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, \
             qidx_x, qidx_y, qidx_z, qidx_xy, \
             kpidx_x, kpidx_y, kpidx_z, \
             jk, knum_p, knum_s, spin, r, matlen, \
             nb_b1, nb_b2, q0, ncol_b1
    cdef int nbands2 = nbands*nbands
    cdef int norb2 = norbital*norbital
    cdef int nspin2 = nspin*nspin
    cdef int* tag = NULL
    cdef MKL_INT* csr_rowptr = NULL
    cdef MKL_INT* csr_cidx = NULL
    cdef sparse_matrix_t* csrH_jk = NULL
    cdef double kx, ky, kz, RK, starttime, endtime
    cdef double t0, build_time = 0.0, epckq_time = 0.0
    cdef double complex* hamilveck
    cdef double complex* expikR_b2
    cdef double complex* expikR_b1
    cdef double complex* dhexpikR = NULL
    cdef double complex* dhvexpikR_b2
    cdef double complex* vdhvexpikR
    cdef double complex* bandveck_b1 = NULL
    cdef double complex* dhveck_b1 = NULL
    cdef int* kpidx_b2
    cdef int* phidx_b2
    cdef int* kidxx_b1
    cdef int* kidxy_b1
    cdef int* kidxz_b1
    cdef int* kidx_b1
    cdef long len_hv, hvlen_b1
    cdef long norbnb_l = norbital*nbands
    cdef int norbnb = norbital*nbands
    cdef matrix_descr descrH
    cdef double complex pi2j = M_PI*2j

    starttime = mpi.MPI_Wtime()
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    descrH.type = SPARSE_MATRIX_TYPE_GENERAL

    if NB_b2>nqpath: NB_b2 = nqpath
    if NB_b1 < 1: NB_b1 = 1
    knum_p = kproc_num[myid]
    knum_s = kproc[myid]

    if use_perjk_flag:
        csrH_jk = <sparse_matrix_t*>malloc(sizeof(sparse_matrix_t)*ncell*ncell*nm_num*nspin2)
        for j in range(ncell):
            for k in range(ncell):
                jk = j*ncell+k
                n = key_num[jk,1]
                if n == 0 or n == norb2:
                    continue
                m = key_num[jk,3]
                for i in range(nm_num):
                    for spin in range(nspin2):
                        mkl_sparse_z_create_csr(
                            &csrH_jk[(jk*nm_num+i)*nspin2+spin],
                            SPARSE_INDEX_BASE_ZERO,norbital,norbital,
                            <MKL_INT*>(&row_ptr_jk[jk*(norbital+1)]),
                            <MKL_INT*>(&row_ptr_jk[jk*(norbital+1)+1]),
                            <MKL_INT*>(&col_idx_jk[m]),
                            <MKL_Complex16*>(&dhamil[i,spin,m])
                        )
                        mkl_sparse_optimize(csrH_jk[(jk*nm_num+i)*nspin2+spin])
    else:
        tag = <int*>calloc(sizeof(int),ncell)
        for i in range(ncell):
            if key_num_s[i,0]>0:
                tag[i] = 1
                if key_num_s[i,0] == norb2: tag[i] = 2
        csr_rowptr = <MKL_INT*>malloc(sizeof(MKL_INT)*ncell*(norbital+1))
        csr_cidx = <MKL_INT*>malloc(sizeof(MKL_INT)*key_num_s[ncell,1])
        for i in range(key_num_s[ncell,1]):
            csr_cidx[i] = key_info_s[i]%norbital
        for j in range(ncell):
            for r in range(norbital+1):
                csr_rowptr[j*(norbital+1)+r] = 0
            for i in range(key_num_s[j,0]):
                r = <int>(key_info_s[key_num_s[j,1]+i]/norbital)
                csr_rowptr[j*(norbital+1)+r+1] += 1
            for r in range(norbital):
                csr_rowptr[j*(norbital+1)+r+1] += csr_rowptr[j*(norbital+1)+r]
        matlen = key_num_s[0,0]
        for i in range(1,ncell):
            if matlen < key_num_s[i,0]:
                matlen = key_num_s[i,0]
        dhexpikR = <double complex*>malloc(sizeof(double complex)*NB_b1*nspin2*matlen)

    hvlen_b1 = nm_num*ncell*norbnb_l*nspin
    len_hv = sizeof(double complex)*hvlen_b1
    expikR_b2 = <double complex*>malloc(sizeof(double complex)*NB_b2*ncell)
    expikR_b1 = <double complex*>malloc(sizeof(double complex)*NB_b1*ncell)
    dhvexpikR_b2 = <double complex*>malloc(sizeof(double complex)*NB_b2*nspin*norbnb)
    vdhvexpikR = <double complex*>malloc(sizeof(double complex)*nbands2)
    hamilveck = <double complex*>malloc(len_hv*NB_b1)
    kpidx_b2 = <int*>malloc(sizeof(int)*NB_b2)
    phidx_b2 = <int*>malloc(sizeof(int)*NB_b2)
    kidxx_b1 = <int*>malloc(sizeof(int)*NB_b1)
    kidxy_b1 = <int*>malloc(sizeof(int)*NB_b1)
    kidxz_b1 = <int*>malloc(sizeof(int)*NB_b1)
    kidx_b1 = <int*>malloc(sizeof(int)*NB_b1)
    if use_perjk_flag:
        bandveck_b1 = <double complex*>malloc(sizeof(double complex)*nspin*norbnb_l*NB_b1)
        dhveck_b1 = <double complex*>malloc(sizeof(double complex)*norbnb_l*NB_b1)

    for h in range(0,knum_p,NB_b1):
        nb_b1 = NB_b1
        if h+nb_b1 > knum_p: nb_b1 = knum_p-h
        ncol_b1 = nbands*nb_b1
        t0 = mpi.MPI_Wtime()
        for i in range(nb_b1):
            k1 = knum_s+h+i
            kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
            kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
            kidxx_b1[i] = kidx_x; kidxy_b1[i] = kidx_y
            kidxz_b1[i] = kidx_z; kidx_b1[i] = k1
            kx = (<double>(kidx_x))/(<double>(nq[0]))
            ky = (<double>(kidx_y))/(<double>(nq[1]))
            kz = (<double>(kidx_z))/(<double>(nq[2]))
            for k in range(ncell):
                RK = R_list[k,0]*kx+R_list[k,1]*ky+R_list[k,2]*kz
                expikR_b1[i*ncell+k] = cexp(pi2j*RK)
        hamilveck_batch(
            use_perjk_flag,nb_b1,ncol_b1,nm_num,ncell,nspin,nspin2,norbital,nbands,
            norb2,norbnb,norbnb_l,hvlen_b1,kidx_b1,expikR_b1,dhamil,bandveck,
            key_num,key_num_s,key_info,csrH_jk,tag,csr_rowptr,csr_cidx,
            bandveck_b1,dhveck_b1,dhexpikR,hamilveck,descrH
        )
        build_time += mpi.MPI_Wtime()-t0

        for i in range(nb_b1):
            kidx_x = kidxx_b1[i]; kidx_y = kidxy_b1[i]; kidx_z = kidxz_b1[i]
            for q0 in range(0,nqpath,NB_b2):
                nb_b2 = NB_b2
                if q0+nb_b2 > nqpath: nb_b2 = nqpath-q0
                for j in range(nb_b2):
                    l = q0+j
                    if nqpath < knum:
                        qidx_x = <int>round(nq[0]*qpath[l,0])
                        qidx_y = <int>round(nq[1]*qpath[l,1])
                        qidx_z = <int>round(nq[2]*qpath[l,2])
                    else:
                        qidx_z = l%nq[2]; qidx_xy = l/nq[2]
                        qidx_y = qidx_xy%nq[1]; qidx_x = qidx_xy/nq[1]
                    phidx_b2[j] = (qidx_x*nq[1]+qidx_y)*nq[2]+qidx_z
                    kpidx_x = (kidx_x+qidx_x)%nq[0]
                    kpidx_y = (kidx_y+qidx_y)%nq[1]
                    kpidx_z = (kidx_z+qidx_z)%nq[2]
                    if kpidx_x<0: kpidx_x += nq[0]
                    if kpidx_y<0: kpidx_y += nq[1]
                    if kpidx_z<0: kpidx_z += nq[2]
                    kpidx_b2[j] = (kpidx_x*nq[1]+kpidx_y)*nq[2]+kpidx_z
                t0 = mpi.MPI_Wtime()
                epckq_batch(
                    &hamilveck[i*hvlen_b1],bandveck,phvecval,
                    expikR_b2,dhvexpikR_b2,vdhvexpikR,&epc_t[(h+i)*nqpath+q0,0],
                    R_list,nq,nspin,nmodes,nm_num,ncell,nbands,norbital,nbands2,
                    norbnb,nb_b2,kpidx_b2,phidx_b2
                )
                epckq_time += mpi.MPI_Wtime()-t0

    if use_perjk_flag:
        for j in range(ncell):
            for k in range(ncell):
                jk = j*ncell+k
                n = key_num[jk,1]
                if n == 0 or n == norb2:
                    continue
                for i in range(nm_num):
                    for spin in range(nspin2):
                        mkl_sparse_destroy(csrH_jk[(jk*nm_num+i)*nspin2+spin])
        free(csrH_jk)
        free(bandveck_b1)
        free(dhveck_b1)
    else:
        free(dhexpikR)
        free(csr_rowptr)
        free(csr_cidx)
        free(tag)

    free(expikR_b2)
    free(expikR_b1)
    free(dhvexpikR_b2)
    free(vdhvexpikR)
    free(hamilveck)
    free(kpidx_b2)
    free(phidx_b2)
    free(kidxx_b1)
    free(kidxy_b1)
    free(kidxz_b1)
    free(kidx_b1)

    mpi.MPI_Barrier(c_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_q_s time in mode[%4d:%4d], NB_b1[%4d] NB_b2[%4d]:%12.4fs. (build:%9.4fs epckq:%9.4fs)\n",nm_min,nm_min+nm_num,NB_b1,NB_b2,endtime-starttime,build_time,epckq_time)


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepc_p(
    MPI.Comm comm,
    int nspin, int nmodes, int nm_num, int nm_min, int norbital, int ncell,
    int knum, int[::1] nq, int[:,::1] R_list, int[:,::1] key_num,
    int[:,::1] key_num_s, int[:,::1] key_info, int[::1] key_info_s,
    double complex[:,:,::1] dhamil, double complex[:,::1] bandveck,
    double complex[:,:,::1] phvecval, int[::1] kproc, int[::1] kproc_num,
    int[:,::1] bassel, double complex[:,::1] epc_t,
    bint use_perjk_flag, int[::1] col_idx_jk, int[::1] row_ptr_jk, int NB_b1, int NB_b2
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int nprocs, myid, ierr, h, i, j, k, l, m, n, k1, k2
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, \
             qidx_x, qidx_y, qidx_z, \
             kpidx_x, kpidx_y, kpidx_z, kpidx_xy, \
             jk, knum_p, knum_s, spin, r, matlen, idx1, \
             nb_b1, nb_b2, idx1_start, idx1_hi, h_end
    cdef int norb2 = norbital*norbital
    cdef int nspin2 = nspin*nspin
    cdef int* tag = NULL
    cdef MKL_INT* csr_rowptr = NULL
    cdef MKL_INT* csr_cidx = NULL
    cdef sparse_matrix_t* csrH_jk = NULL
    cdef double kx, ky, kz, RK, starttime, endtime
    cdef double t0, build_time = 0.0, epckq_time = 0.0
    cdef double complex* hamilveck
    cdef double complex* expikR_b2
    cdef double complex* expikR_b1
    cdef double complex* dhexpikR = NULL
    cdef double complex* dhvexpikR_b2
    cdef double complex* bandveck_b1 = NULL
    cdef double complex* dhveck_b1 = NULL
    cdef int* kpidx_b2
    cdef int* kgrid_b2
    cdef int* phidx_b2
    cdef int* kidxx_b1
    cdef int* kidxy_b1
    cdef int* kidxz_b1
    cdef int* kidx_b1
    cdef long len_hv, hvlen_b1
    cdef matrix_descr descrH
    cdef double complex pi2j = M_PI*2j

    starttime = mpi.MPI_Wtime()
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    descrH.type = SPARSE_MATRIX_TYPE_GENERAL

    if NB_b2>knum: NB_b2 = knum
    if NB_b1 < 1: NB_b1 = 1
    knum_p = kproc_num[myid]
    knum_s = kproc[myid]
    idx1_hi = <int>((knum_s+knum_p-1)/knum)

    if use_perjk_flag:
        csrH_jk = <sparse_matrix_t*>malloc(sizeof(sparse_matrix_t)*ncell*ncell*nm_num*nspin2)
        for j in range(ncell):
            for k in range(ncell):
                jk = j*ncell+k
                n = key_num[jk,1]
                if n == 0 or n == norb2:
                    continue
                m = key_num[jk,3]
                for i in range(nm_num):
                    for spin in range(nspin2):
                        mkl_sparse_z_create_csr(
                            &csrH_jk[(jk*nm_num+i)*nspin2+spin],
                            SPARSE_INDEX_BASE_ZERO,norbital,norbital,
                            <MKL_INT*>(&row_ptr_jk[jk*(norbital+1)]),
                            <MKL_INT*>(&row_ptr_jk[jk*(norbital+1)+1]),
                            <MKL_INT*>(&col_idx_jk[m]),
                            <MKL_Complex16*>(&dhamil[i,spin,m])
                        )
                        mkl_sparse_optimize(csrH_jk[(jk*nm_num+i)*nspin2+spin])
    else:
        tag = <int*>calloc(sizeof(int),ncell)
        for i in range(ncell):
            if key_num_s[i,0]>0:
                tag[i] = 1
                if key_num_s[i,0] == norb2: tag[i] = 2
        csr_rowptr = <MKL_INT*>malloc(sizeof(MKL_INT)*ncell*(norbital+1))
        csr_cidx = <MKL_INT*>malloc(sizeof(MKL_INT)*key_num_s[ncell,1])
        for i in range(key_num_s[ncell,1]):
            csr_cidx[i] = key_info_s[i]%norbital
        for j in range(ncell):
            for r in range(norbital+1):
                csr_rowptr[j*(norbital+1)+r] = 0
            for i in range(key_num_s[j,0]):
                r = <int>(key_info_s[key_num_s[j,1]+i]/norbital)
                csr_rowptr[j*(norbital+1)+r+1] += 1
            for r in range(norbital):
                csr_rowptr[j*(norbital+1)+r+1] += csr_rowptr[j*(norbital+1)+r]
        matlen = key_num_s[0,0]
        for i in range(1,ncell):
            if matlen < key_num_s[i,0]:
                matlen = key_num_s[i,0]
        dhexpikR = <double complex*>malloc(sizeof(double complex)*NB_b1*nspin2*matlen)

    hvlen_b1 = nm_num*ncell*norbital*nspin
    len_hv = sizeof(double complex)*hvlen_b1
    expikR_b2 = <double complex*>malloc(sizeof(double complex)*NB_b2*ncell)
    expikR_b1 = <double complex*>malloc(sizeof(double complex)*NB_b1*ncell)
    dhvexpikR_b2 = <double complex*>malloc(sizeof(double complex)*NB_b2*nspin*norbital)
    hamilveck = <double complex*>malloc(len_hv*NB_b1)
    kpidx_b2 = <int*>malloc(sizeof(int)*NB_b2)
    kgrid_b2 = <int*>malloc(sizeof(int)*NB_b2)
    phidx_b2 = <int*>malloc(sizeof(int)*NB_b2)
    kidxx_b1 = <int*>malloc(sizeof(int)*NB_b1)
    kidxy_b1 = <int*>malloc(sizeof(int)*NB_b1)
    kidxz_b1 = <int*>malloc(sizeof(int)*NB_b1)
    kidx_b1 = <int*>malloc(sizeof(int)*NB_b1)
    if use_perjk_flag:
        bandveck_b1 = <double complex*>malloc(sizeof(double complex)*nspin*norbital*NB_b1)
        dhveck_b1 = <double complex*>malloc(sizeof(double complex)*norbital*NB_b1)

    for idx1_start in range(<int>(knum_s/knum),idx1_hi+1,NB_b1):
        nb_b1 = NB_b1
        if idx1_start+nb_b1-1 > idx1_hi: nb_b1 = idx1_hi-idx1_start+1
        t0 = mpi.MPI_Wtime()
        for i in range(nb_b1):
            idx1 = idx1_start+i
            k1 = bassel[idx1,0]
            kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
            kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
            kidxx_b1[i] = kidx_x; kidxy_b1[i] = kidx_y
            kidxz_b1[i] = kidx_z; kidx_b1[i] = idx1
            kx = (<double>(kidx_x))/(<double>(nq[0]))
            ky = (<double>(kidx_y))/(<double>(nq[1]))
            kz = (<double>(kidx_z))/(<double>(nq[2]))
            for k in range(ncell):
                RK = R_list[k,0]*kx+R_list[k,1]*ky+R_list[k,2]*kz
                expikR_b1[i*ncell+k] = cexp(pi2j*RK)
        hamilveck_batch_p(
            use_perjk_flag,nb_b1,nm_num,ncell,nspin,nspin2,norbital,
            norb2,hvlen_b1,kidx_b1,expikR_b1,dhamil,bandveck,
            key_num,key_num_s,key_info,csrH_jk,tag,csr_rowptr,csr_cidx,
            bandveck_b1,dhveck_b1,dhexpikR,hamilveck,descrH
        )
        build_time += mpi.MPI_Wtime()-t0

        for i in range(nb_b1):
            idx1 = idx1_start+i
            kidx_x = kidxx_b1[i]; kidx_y = kidxy_b1[i]; kidx_z = kidxz_b1[i]
            h = idx1*knum-knum_s
            if h < 0: h = 0
            h_end = (idx1+1)*knum-knum_s
            if h_end > knum_p: h_end = knum_p
            for h in range(h,h_end,NB_b2):
                nb_b2 = NB_b2
                if h+nb_b2 > h_end: nb_b2 = h_end-h
                for j in range(nb_b2):
                    l = (h+knum_s+j)%knum
                    k2 = bassel[l,0]
                    kpidx_z = k2%nq[2]; kpidx_xy = k2/nq[2]
                    kpidx_y = kpidx_xy%nq[1]; kpidx_x = kpidx_xy/nq[1]
                    qidx_x = (kpidx_x-kidx_x)%nq[0]
                    qidx_y = (kpidx_y-kidx_y)%nq[1]
                    qidx_z = (kpidx_z-kidx_z)%nq[2]
                    if qidx_x<0: qidx_x += nq[0]
                    if qidx_y<0: qidx_y += nq[1]
                    if qidx_z<0: qidx_z += nq[2]
                    kpidx_b2[j] = l
                    kgrid_b2[j] = k2
                    phidx_b2[j] = (qidx_x*nq[1]+qidx_y)*nq[2]+qidx_z
                t0 = mpi.MPI_Wtime()
                epckq_p_batch(
                    &hamilveck[i*hvlen_b1],bandveck,phvecval,
                    expikR_b2,dhvexpikR_b2,&epc_t[h,0],R_list,nq,
                    nspin,nmodes,nm_num,ncell,norbital,nb_b2,kpidx_b2,kgrid_b2,phidx_b2
                )
                epckq_time += mpi.MPI_Wtime()-t0

    if use_perjk_flag:
        for j in range(ncell):
            for k in range(ncell):
                jk = j*ncell+k
                n = key_num[jk,1]
                if n == 0 or n == norb2:
                    continue
                for i in range(nm_num):
                    for spin in range(nspin2):
                        mkl_sparse_destroy(csrH_jk[(jk*nm_num+i)*nspin2+spin])
        free(csrH_jk)
        free(bandveck_b1)
        free(dhveck_b1)
    else:
        free(dhexpikR)
        free(csr_rowptr)
        free(csr_cidx)
        free(tag)

    free(expikR_b2)
    free(expikR_b1)
    free(dhvexpikR_b2)
    free(hamilveck)
    free(kpidx_b2)
    free(kgrid_b2)
    free(phidx_b2)
    free(kidxx_b1)
    free(kidxy_b1)
    free(kidxz_b1)
    free(kidx_b1)

    mpi.MPI_Barrier(c_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_all_p_s time in mode[%4d:%4d], NB_b1[%4d] NB_b2[%4d]:%12.4fs. (build:%9.4fs epckq:%9.4fs)\n",nm_min,nm_min+nm_num,NB_b1,NB_b2,endtime-starttime,build_time,epckq_time)
