#cython: language_level=3
#cython: cdivision=True

cimport cython
from epc cimport *
from mkl_cblas cimport *
from mkl_spblas cimport *

@cython.boundscheck(False)
@cython.wraparound(False)
def epc_preprocess(
    MPI.Comm shm_comm, 
    int myid, int shm_id, int nprocs_shm, int knum, int nmodes, int nm_num, int nm_min, 
    int atomnum, double factor1, double[::1] mass, double[:,::1] phval, 
    double complex[:,:,::1] phvecval, char* phvecname
):
    cdef mpi.MPI_Comm c_shm_comm = shm_comm.ob_mpi
    cdef int h, i, j, k, l
    cdef double m, phvalmass, starttime, endtime
    cdef double* mass2 = <double*>malloc(sizeof(double)*atomnum)
    cdef int knum_min, knum_max
    cdef long nmodes_l = nmodes
    cdef FILE * fp

    starttime = mpi.MPI_Wtime()
    #mpi.MPI_Barrier(c_shm_comm)
    if (shm_id == 0):
        fp = fopen(phvecname,"rb")
        fseek(fp,0,SEEK_SET)
        for i in range(knum):
            fseek(fp,nm_min*nmodes_l*16,SEEK_CUR)
            fread(&phvecval[i,0,0],sizeof(double complex),nm_num*nmodes,fp)
            fseek(fp,(nmodes-nm_min-nm_num)*nmodes_l*16,SEEK_CUR)
        fclose(fp)
    mpi.MPI_Barrier(c_shm_comm)

    knum_min = <int>((knum*shm_id)/nprocs_shm)
    knum_max = <int>((knum*(shm_id+1))/nprocs_shm)
    for i in range(atomnum):
        mass2[i] = 1.0/sqrt(mass[i])*factor1
    for i in range(knum_min,knum_max):
        for j in range(nm_num):
            h = <int>((j+nm_min)/3)
            m = mass2[h]
            for l in range(nmodes):
                phvalmass = phval[i,l]*m
                phvecval[i,j,l] *= phvalmass

    free(mass2)
    mpi.MPI_Barrier(c_shm_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_preprocess time in mode[%4d:%4d]:%12.4fs.\n",nm_min,nm_min+nm_num,endtime-starttime)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void epckq(
    double Kpx, double Kpy, double Kpz, double complex* hamilveck,
    double complex[:,::1] bandveckp, double complex[:,::1] phvecval, 
    double complex* dhvexpikR, double complex* vdhvexpikR, 
    double complex* epcq, int* tag, int[:,::1] R_list, int nmodes, 
    int nm_num, int ncell, int nbands, int norbital, int nbands2, 
    MKL_INT norbnb
):
    cdef int i, j, k
    cdef double RKpx, RKpy, RKpz, RKp
    cdef double complex pi2j = M_PI*2j
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef double complex phvv, expikR

    for i in range(nm_num):
        memset(dhvexpikR,0,norbnb*sizeof(double complex))
        for j in range(ncell):
            if tag[j] > 0:
                RKpx = R_list[j,0]*Kpx
                RKpy = R_list[j,1]*Kpy
                RKpz = R_list[j,2]*Kpz
                RKp = RKpx + RKpy + RKpz
                expikR = cexp(-1.0*pi2j*RKp)
                for k in range(norbnb):
                    dhvexpikR[k] += hamilveck[(i*ncell+j)*norbnb+k]*expikR
        cblas_zgemm3m(
            CblasRowMajor,CblasConjTrans,CblasNoTrans,
            nbands,nbands,norbital,&c1,&bandveckp[0,0],
            nbands,dhvexpikR,nbands,&c0,vdhvexpikR,nbands
        )
        for j in range(nmodes):
            phvv = phvecval[i,j]
            for k in range(nbands2):
                epcq[j*nbands2+k] += vdhvexpikR[k]*phvv


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepc(
    MPI.Comm comm,
    int spin, int nmodes, int nm_num, int nm_min, int norbital, int nbands, 
    int ncell, int knum, int[::1] nq, int[:,::1] R_list, int nkpath, 
    double[:,::1] kpath, int[:,::1] key_num, int[:,::1] key_num_s, 
    int[:,::1] key_info, int[::1] key_info_s, double[:,:,::1] dhamil, 
    double complex[:,:,::1] bandveck, double complex[:,:,::1] phvecval, 
    int[::1] kproc, int[::1] kproc_num, double complex[:,::1] epc_t
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int nprocs, myid, ierr, h, i, j, k, l, m, n, k1, k2
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, kidx, \
             kpidx_x, kpidx_y, kpidx_z, kpidx_xy, kpidx, \
             qidx_x, qidx_y, qidx_z, qidx_xy, qidx, \
             qnum_p, qnum_s, matlen
    cdef int nbands2 = nbands*nbands
    cdef int norb2 = norbital*norbital
    cdef int * tag
    cdef MKL_INT* coo_ridx = <MKL_INT*>malloc(sizeof(MKL_INT)*key_num_s[ncell,1])
    cdef MKL_INT* coo_cidx = <MKL_INT*>malloc(sizeof(MKL_INT)*key_num_s[ncell,1])
    cdef double kx, ky, kz, kpx, kpy, kpz, RKx, RKy, RKz, RK, starttime, endtime
    cdef double complex* hamilveck
    cdef MKL_Complex16* hamilveck_t
    cdef double complex* dhexpikR
    cdef double complex* dhvexpikR
    cdef double complex* vdhvexpikR
    cdef double complex expikR
    cdef long len_hv
    cdef long norbnb_l = norbital*nbands
    cdef int norbnb = norbital*nbands
    cdef matrix_descr descrH
    cdef sparse_matrix_t cooH
    cdef double complex pi2j = M_PI*2j
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef MKL_Complex16 c0mkl
    cdef MKL_Complex16 c1mkl
    c0mkl.real = 0.0; c0mkl.imag = 0.0
    c1mkl.real = 1.0; c1mkl.imag = 0.0

    starttime = mpi.MPI_Wtime()
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    descrH.type = SPARSE_MATRIX_TYPE_GENERAL

    tag = <int*>calloc(sizeof(int),ncell)
    for i in range(ncell):
        if key_num_s[i,0]>0:
            tag[i] = 1
            if key_num_s[i,0] == norb2: tag[i] = 2
    for i in range(key_num_s[ncell,1]):
        coo_ridx[i] = <int>(key_info_s[i]/norbital)
        coo_cidx[i] = key_info_s[i]%norbital

    matlen = key_num_s[0,0]
    for i in range(1,ncell):
        if (matlen<key_num_s[i,0]):
            matlen = key_num_s[i,0]
    dhexpikR = <double complex*>malloc(sizeof(double complex)*matlen)

    dhvexpikR = <double complex*>malloc(sizeof(double complex)*norbnb)
    vdhvexpikR = <double complex*>malloc(sizeof(double complex)*nbands2)
    len_hv = sizeof(double complex)*nm_num*ncell*norbnb_l
    hamilveck = <double complex*>malloc(len_hv)

    knum_p = kproc_num[myid]
    knum_s = kproc[myid]

    kt = nkpath*knum
    for h in range(knum_p):
        k1 = <int>((h+knum_s)/knum)
        k2 = (h+knum_s)%knum
        if nkpath < knum:
            kidx_x = <int>round(nq[0]*kpath[k1,0])
            kidx_y = <int>round(nq[1]*kpath[k1,1])
            kidx_z = <int>round(nq[2]*kpath[k1,2])
        else:
            kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
            kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
        kidx = (kidx_x*nq[1]+kidx_y)*nq[2]+kidx_z

        qidx_z = k2%nq[2]; qidx_xy = k2/nq[2]
        qidx_y = qidx_xy%nq[1]; qidx_x = qidx_xy/nq[1]
        kpidx_x = (kidx_x+qidx_x)%nq[0]
        kpidx_y = (kidx_y+qidx_y)%nq[1]
        kpidx_z = (kidx_z+qidx_z)%nq[2]
        if kpidx_x<0: kpidx_x += nq[0]
        if kpidx_y<0: kpidx_y += nq[1]
        if kpidx_z<0: kpidx_z += nq[2]
        kpidx = (kpidx_x*nq[1]+kpidx_y)*nq[2]+kpidx_z
        kpx = (<double>(kpidx_x))/(<double>(nq[0]))
        kpy = (<double>(kpidx_y))/(<double>(nq[1]))
        kpz = (<double>(kpidx_z))/(<double>(nq[2]))

        if (kt != k1):
            kx = (<double>(kidx_x))/(<double>(nq[0]))
            ky = (<double>(kidx_y))/(<double>(nq[1]))
            kz = (<double>(kidx_z))/(<double>(nq[2]))
            hamilveck_t = <MKL_Complex16*>(&hamilveck[0])
            for i in range(nm_num):
                for j in range(ncell):
                    if tag[j] > 0:
                        memset(dhexpikR,0,key_num_s[j,0]*sizeof(double complex))
                        for k in range(ncell):
                            n = key_num[j*ncell+k,1]
                            if n > 0:
                                m = key_num[j*ncell+k,3]
                                RKx = R_list[k,0]*kx
                                RKy = R_list[k,1]*ky
                                RKz = R_list[k,2]*kz
                                RK = RKx + RKy + RKz
                                expikR = cexp(pi2j*RK)
                                if n == norb2:
                                    for l in range(norb2):
                                        dhexpikR[l] \
                                        += dhamil[i,spin,m+l]*expikR
                                else:
                                    for l in range(n):
                                        dhexpikR[key_info[m+l,1]] \
                                        += dhamil[i,spin,m+l]*expikR
                        if tag[j] == 1:
                            mkl_sparse_z_create_coo(
                                &cooH,SPARSE_INDEX_BASE_ZERO,norbital,norbital,
                                key_num_s[j,0],&coo_ridx[key_num_s[j,1]],
                                &coo_cidx[key_num_s[j,1]],<MKL_Complex16*>dhexpikR
                            )
                            mkl_sparse_optimize(cooH)
                            mkl_sparse_z_mm(
                                SPARSE_OPERATION_NON_TRANSPOSE,c1mkl,cooH,
                                descrH,SPARSE_LAYOUT_ROW_MAJOR,
                                <MKL_Complex16*>(&bandveck[kidx,0,0]),
                                nbands,nbands,c0mkl,hamilveck_t,nbands
                            )
                            mkl_sparse_destroy(cooH)
                        else:
                            cblas_zgemm(
                                CblasRowMajor,CblasNoTrans,CblasNoTrans,norbital,
                                nbands,norbital,&c1,dhexpikR,norbital,
                                &bandveck[kidx,0,0],nbands,&c0,hamilveck_t,nbands
                            )
                    hamilveck_t += norbnb
            kt = k1

        epckq(
            kpx,kpy,kpz,hamilveck,bandveck[kpidx],phvecval[k2],
            dhvexpikR,vdhvexpikR,&epc_t[h,0],tag,R_list,
            nmodes,nm_num,ncell,nbands,norbital,nbands2,norbnb
        )

    free(dhexpikR)
    free(dhvexpikR)
    free(vdhvexpikR)
    free(hamilveck)
    free(coo_ridx)
    free(coo_cidx)
    free(tag)

    mpi.MPI_Barrier(c_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_s time in mode[%4d:%4d]:%12.4fs.\n",nm_min,nm_min+nm_num,endtime-starttime)


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepc_q(
    MPI.Comm comm,
    int spin, int nmodes, int nm_num, int nm_min, int norbital, int nbands, 
    int ncell, int knum, int[::1] nq, int[:,::1] R_list, int nqpath, 
    double[:,::1] qpath, int[:,::1] key_num, int[:,::1] key_num_s, 
    int[:,::1] key_info, int[::1] key_info_s, double[:,:,::1] dhamil,
    double complex[:,:,::1] bandveck, double complex[:,:,::1] phvecval,
    int[::1] kproc, int[::1] kproc_num, double complex[:,::1] epc_t
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int nprocs, myid, ierr, h, i, j, k, l, m, n, k1
    cdef int qidx_x, qidx_y, qidx_z, qidx_xy, qidx, \
             kpidx_x, kpidx_y, kpidx_z, kpidx_xy, kpidx, \
             kidx_x, kidx_y, kidx_z, kidx_xy, kidx, \
             knum_p, knum_s, matlen
    cdef int nbands2 = nbands*nbands
    cdef int norb2 = norbital*norbital
    cdef int * tag
    cdef MKL_INT* coo_ridx = <MKL_INT*>malloc(sizeof(MKL_INT)*key_num_s[ncell,1])
    cdef MKL_INT* coo_cidx = <MKL_INT*>malloc(sizeof(MKL_INT)*key_num_s[ncell,1])
    cdef double kx, ky, kz, kpx, kpy, kpz, RKx, RKy, RKz, RK, starttime, endtime
    cdef double complex* hamilveck
    cdef MKL_Complex16* hamilveck_t
    cdef double complex* dhexpikR
    cdef double complex* dhvexpikR
    cdef double complex* vdhvexpikR
    cdef double complex expikR
    cdef long len_hv
    cdef long norbnb_l = norbital*nbands
    cdef int norbnb = norbital*nbands
    cdef matrix_descr descrH
    cdef sparse_matrix_t cooH
    cdef double complex pi2j = M_PI*2j
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef MKL_Complex16 c0mkl
    cdef MKL_Complex16 c1mkl
    c0mkl.real = 0.0; c0mkl.imag = 0.0
    c1mkl.real = 1.0; c1mkl.imag = 0.0

    starttime = mpi.MPI_Wtime()
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    descrH.type = SPARSE_MATRIX_TYPE_GENERAL

    tag = <int*>calloc(sizeof(int),ncell)
    for i in range(ncell):
        if key_num_s[i,0]>0:
            tag[i] = 1
            if key_num_s[i,0] == norb2: tag[i] = 2
    for i in range(key_num_s[ncell,1]):
        coo_ridx[i] = <int>(key_info_s[i]/norbital)
        coo_cidx[i] = key_info_s[i]%norbital

    matlen = key_num_s[0,0]
    for i in range(1,ncell):
        if (matlen<key_num_s[i,0]):
            matlen = key_num_s[i,0]
    dhexpikR = <double complex*>malloc(sizeof(double complex)*matlen)

    dhvexpikR = <double complex*>malloc(sizeof(double complex)*norbnb)
    vdhvexpikR = <double complex*>malloc(sizeof(double complex)*nbands2)
    len_hv = sizeof(double complex)*nm_num*ncell*norbnb_l
    hamilveck = <double complex*>malloc(len_hv)

    knum_p = kproc_num[myid]
    knum_s = kproc[myid]

    for h in range(knum_p):
        start = mpi.MPI_Wtime()
        k1 = h+knum_s
        kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
        kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
        kx = (<double>(kidx_x))/(<double>(nq[0]))
        ky = (<double>(kidx_y))/(<double>(nq[1]))
        kz = (<double>(kidx_z))/(<double>(nq[2]))

        hamilveck_t = <MKL_Complex16*>(&hamilveck[0])
        for i in range(nm_num):
            for j in range(ncell):
                if tag[j] > 0:
                    memset(dhexpikR,0,key_num_s[j,0]*sizeof(double complex))
                    for k in range(ncell):
                        n = key_num[j*ncell+k,1]
                        if n > 0:
                            m = key_num[j*ncell+k,3]
                            RKx = R_list[k,0]*kx
                            RKy = R_list[k,1]*ky
                            RKz = R_list[k,2]*kz
                            RK = RKx + RKy + RKz
                            expikR = cexp(pi2j*RK)
                            if n == norb2:
                                for l in range(norb2):
                                    dhexpikR[l] \
                                    += dhamil[i,spin,m+l]*expikR
                            else:
                                for l in range(n):
                                    dhexpikR[key_info[m+l,1]] \
                                    += dhamil[i,spin,m+l]*expikR
                    if tag[j] == 1:
                        mkl_sparse_z_create_coo(
                            &cooH,SPARSE_INDEX_BASE_ZERO,norbital,norbital,
                            key_num_s[j,0],&coo_ridx[key_num_s[j,1]],
                            &coo_cidx[key_num_s[j,1]],<MKL_Complex16*>dhexpikR
                        )
                        mkl_sparse_optimize(cooH)
                        mkl_sparse_z_mm(
                            SPARSE_OPERATION_NON_TRANSPOSE,c1mkl,cooH,
                            descrH,SPARSE_LAYOUT_ROW_MAJOR,
                            <MKL_Complex16*>(&bandveck[k1,0,0]),
                            nbands,nbands,c0mkl,hamilveck_t,nbands
                        )
                        mkl_sparse_destroy(cooH)
                    else:
                        cblas_zgemm(
                            CblasRowMajor,CblasNoTrans,CblasNoTrans,
                            norbital,nbands,norbital,&c1,dhexpikR,norbital,
                            &bandveck[k1,0,0],nbands,&c0,hamilveck_t,nbands
                        )
                hamilveck_t += norbnb

        for i in range(nqpath):
            if nqpath < knum:
                qidx_x = <int>round(nq[0]*qpath[i,0])
                qidx_y = <int>round(nq[1]*qpath[i,1])
                qidx_z = <int>round(nq[2]*qpath[i,2])
            else:
                qidx_z = i%nq[2]; qidx_xy = i/nq[2]
                qidx_y = qidx_xy%nq[1]; qidx_x = qidx_xy/nq[1]
            qidx = (qidx_x*nq[1]+qidx_y)*nq[2]+qidx_z
            kpidx_x = (kidx_x+qidx_x)%nq[0]
            kpidx_y = (kidx_y+qidx_y)%nq[1]
            kpidx_z = (kidx_z+qidx_z)%nq[2]
            if kpidx_x<0: kpidx_x += nq[0]
            if kpidx_y<0: kpidx_y += nq[1]
            if kpidx_z<0: kpidx_z += nq[2]
            kpidx = (kpidx_x*nq[1]+kpidx_y)*nq[2]+kpidx_z
            kpx = (<double>(kpidx_x))/(<double>(nq[0]))
            kpy = (<double>(kpidx_y))/(<double>(nq[1]))
            kpz = (<double>(kpidx_z))/(<double>(nq[2]))

            epckq(
                kpx,kpy,kpz,hamilveck,bandveck[kpidx],phvecval[qidx],
                dhvexpikR,vdhvexpikR,&epc_t[h*nqpath+i,0],tag,R_list,
                nmodes,nm_num,ncell,nbands,norbital,nbands2,norbnb
            )

        end = mpi.MPI_Wtime()
        if myid == 0:
            printf("epc_q_s time in loop[%8d]:%12.4fs.\n",h,end-start)

    free(dhexpikR)
    free(dhvexpikR)
    free(vdhvexpikR)
    free(hamilveck)
    free(coo_ridx)
    free(coo_cidx)
    free(tag)

    mpi.MPI_Barrier(c_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_q_s time in mode[%4d:%4d]:%12.4fs.\n",nm_min,nm_min+nm_num,endtime-starttime)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void epckq_p(
    double Kpx, double Kpy, double Kpz, double complex* hamilveck,
    double complex* bandveckp, double complex[:,::1] phvecval,
    double complex* dhvexpikR, double complex* epcq, int* tag, 
    int[:,::1] R_list, int nmodes, int nm_num, int ncell, int norbital
):
    cdef int i, j, k
    cdef double RKpx, RKpy, RKpz, RKp
    cdef double complex pi2j = M_PI*2j
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef double complex expikR, vdhvexpikR

    for i in range(nm_num):
        memset(dhvexpikR,0,norbital*sizeof(double complex))
        for j in range(ncell):
            if tag[j] > 0:
                RKpx = R_list[j,0]*Kpx
                RKpy = R_list[j,1]*Kpy
                RKpz = R_list[j,2]*Kpz
                RKp = RKpx + RKpy + RKpz
                expikR = cexp(-1.0*pi2j*RKp)
                for k in range(norbital):
                    dhvexpikR[k] += hamilveck[(i*ncell+j)*norbital+k]*expikR
        cblas_zdotc_sub(
            norbital,bandveckp,1,dhvexpikR,1,&vdhvexpikR
        )
        for j in range(nmodes):
            epcq[j] += vdhvexpikR*phvecval[i,j]


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepc_p(
    MPI.Comm comm,
    int spin, int nmodes, int nm_num, int nm_min, int norbital, int ncell, 
    int knum, int[::1] nq, int[:,::1] R_list,
    int[:,::1] key_num, int[:,::1] key_num_s, int[:,::1] key_info, 
    int[::1] key_info_s, double[:,:,::1] dhamil, double complex[:,::1] bandveck, 
    double complex[:,:,::1] phvecval, int[::1] kproc,
    int[::1] kproc_num, int[:,::1] bassel, double complex[:,::1] epc_t
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int nprocs, myid, ierr, h, i, j, k, l, m, n
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, kidx, \
             qidx_x, qidx_y, qidx_z, qidx_xy, qidx, \
             kpidx_x, kpidx_y, kpidx_z, kpidx_xy, kpidx, \
             knum_p, knum_s, k1, k2, kt, idx1, idx2, matlen
    cdef int norb2 = norbital*norbital
    cdef int knum2 = knum*knum
    cdef int * tag
    cdef double kx, ky, kz, kpx, kpy, kpz, RKx, RKy, RKz, RK, starttime, endtime
    cdef double complex* hamilveck
    cdef MKL_Complex16* hamilveck_t
    cdef double complex* dhexpikR
    cdef double complex* dhvexpikR
    cdef double complex expikR
    cdef long len_hv
    cdef long norbital_l = norbital
    cdef matrix_descr descrH
    cdef sparse_matrix_t cooH
    cdef MKL_INT* coo_ridx = <MKL_INT*>malloc(sizeof(MKL_INT)*key_num_s[ncell,1])
    cdef MKL_INT* coo_cidx = <MKL_INT*>malloc(sizeof(MKL_INT)*key_num_s[ncell,1])
    cdef double complex pi2j = M_PI*2j
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef MKL_Complex16 c0mkl
    cdef MKL_Complex16 c1mkl
    c0mkl.real = 0.0; c0mkl.imag = 0.0
    c1mkl.real = 1.0; c1mkl.imag = 0.0

    starttime = mpi.MPI_Wtime()
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    descrH.type = SPARSE_MATRIX_TYPE_GENERAL

    tag = <int*>calloc(sizeof(int),ncell)
    for i in range(ncell):
        if key_num_s[i,0]>0:
            tag[i] = 1
            if key_num_s[i,0] == norb2: tag[i] = 2
    for i in range(key_num_s[ncell,1]):
        coo_ridx[i] = <int>(key_info_s[i]/norbital)
        coo_cidx[i] = key_info_s[i]%norbital

    matlen = key_num_s[0,0]
    for i in range(1,ncell):
        if (matlen<key_num_s[i,0]):
            matlen = key_num_s[i,0]
    dhexpikR = <double complex*>malloc(sizeof(double complex)*matlen)

    dhvexpikR = <double complex*>malloc(sizeof(double complex)*norbital)
    len_hv = sizeof(double complex)*nm_num*ncell*norbital_l
    hamilveck = <double complex*>malloc(len_hv)

    knum_p = kproc_num[myid]
    knum_s = kproc[myid]

    kt = knum2
    for h in range(knum_p):
        idx1 = <int>((h+knum_s)/knum)
        idx2 = (h+knum_s)%knum
        k1 = bassel[idx1,0]
        k2 = bassel[idx2,0]

        kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
        kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
        kpidx_z = k2%nq[2]; kpidx_xy = k2/nq[2]
        kpidx_y = kpidx_xy%nq[1]; kpidx_x = kpidx_xy/nq[1]
        kpx = (<double>(kpidx_x))/(<double>(nq[0]))
        kpy = (<double>(kpidx_y))/(<double>(nq[1]))
        kpz = (<double>(kpidx_z))/(<double>(nq[2]))

        qidx_x = (kpidx_x-kidx_x)%nq[0]
        qidx_y = (kpidx_y-kidx_y)%nq[1]
        qidx_z = (kpidx_z-kidx_z)%nq[2]
        if qidx_x<0: qidx_x += nq[0]
        if qidx_y<0: qidx_y += nq[1]
        if qidx_z<0: qidx_z += nq[2]
        qidx = (qidx_x*nq[1]+qidx_y)*nq[2]+qidx_z

        if (kt != idx1):
            kx = (<double>(kidx_x))/(<double>(nq[0]))
            ky = (<double>(kidx_y))/(<double>(nq[1]))
            kz = (<double>(kidx_z))/(<double>(nq[2]))
            hamilveck_t = <MKL_Complex16*>(&hamilveck[0])
            for i in range(nm_num):
                for j in range(ncell):
                    if tag[j] > 0:
                        memset(dhexpikR,0,key_num_s[j,0]*sizeof(double complex))
                        for k in range(ncell):
                            n = key_num[j*ncell+k,1]
                            if n > 0:
                                m = key_num[j*ncell+k,3]
                                RKx = R_list[k,0]*kx
                                RKy = R_list[k,1]*ky
                                RKz = R_list[k,2]*kz
                                RK = RKx + RKy + RKz
                                expikR = cexp(pi2j*RK)
                                if n == norb2:
                                    for l in range(norb2):
                                        dhexpikR[l] \
                                        += dhamil[i,spin,m+l]*expikR
                                else:
                                    for l in range(n):
                                        dhexpikR[key_info[m+l,1]] \
                                        += dhamil[i,spin,m+l]*expikR
                        if tag[j] == 1:
                            mkl_sparse_z_create_coo(
                                &cooH,SPARSE_INDEX_BASE_ZERO,norbital,norbital,
                                key_num_s[j,0],&coo_ridx[key_num_s[j,1]],
                                &coo_cidx[key_num_s[j,1]],<MKL_Complex16*>dhexpikR
                            )
                            mkl_sparse_optimize(cooH)
                            mkl_sparse_z_mv(
                                SPARSE_OPERATION_NON_TRANSPOSE,c1mkl,cooH,
                                descrH,<MKL_Complex16*>(&bandveck[idx1,0]),
                                c0mkl,hamilveck_t
                            )
                            mkl_sparse_destroy(cooH)
                        else:
                            cblas_zgemv(
                                CblasRowMajor,CblasNoTrans,norbital,
                                norbital,&c1,dhexpikR,norbital,
                                &bandveck[idx1,0],1,&c0,hamilveck_t,1
                            )
                    hamilveck_t += norbital
            kt = idx1

        epckq_p(
            kpx,kpy,kpz,hamilveck,&bandveck[idx2,0],phvecval[qidx],
            dhvexpikR,&epc_t[h,0],tag,R_list,nmodes,nm_num,ncell,norbital
        )

    free(dhexpikR)
    free(dhvexpikR)
    free(tag)
    free(hamilveck)
    free(coo_ridx)
    free(coo_cidx)

    mpi.MPI_Barrier(c_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_all_p_s time in mode[%4d:%4d]:%12.4fs.\n",nm_min,nm_min+nm_num,endtime-starttime)


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepc_write(
    MPI.Comm comm, int nmnb2,
    int[::1] kproc, int[::1] kproc_num,
    double complex[:,::1] epc_t, char* filename
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Datatype CPLX_N
    cdef int nprocs, myid, ierr
    cdef long nmnb2l = nmnb2
    cdef mpi.MPI_Offset offset
    cdef mpi.MPI_Status status
    cdef mpi.MPI_File fh

    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)

    mpi.MPI_Type_contiguous(
        nmnb2,mpi.MPI_DOUBLE_COMPLEX,&CPLX_N
    )
    mpi.MPI_Type_commit(&CPLX_N)
    mpi.MPI_File_open(
        mpi.MPI_COMM_WORLD,filename,
        mpi.MPI_MODE_CREATE | mpi.MPI_MODE_WRONLY,
        mpi.MPI_INFO_NULL,&fh
    )
    offset = nmnb2l*kproc[myid]*sizeof(double complex)
    mpi.MPI_File_write_at_all(
        fh,offset,&epc_t[0,0],kproc_num[myid],CPLX_N,&status
    )
    mpi.MPI_File_close(&fh)
    mpi.MPI_Type_free(&CPLX_N)

    mpi.MPI_Barrier(c_comm)
