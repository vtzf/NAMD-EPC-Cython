#cython: language_level=3
#cython: cdivision=True

cimport cython
from epc cimport *
from mkl_cblas cimport *

@cython.boundscheck(False)
@cython.wraparound(False)
def phvec_read(
    MPI.Comm comm, int myid, 
    int nmodes, int[::1] qproc, int[::1] qproc_num, 
    double factor1, double[::1] mass, double[:,::1] phval, 
    double complex[:,::1] phvecval_p, char* phvecname
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef FILE * fp
    cdef int i, j, k
    cdef int atomnum = nmodes/3
    cdef long nmodes2 = nmodes*nmodes
    cdef int qnum_s = qproc[myid]
    cdef int qnum_p = qproc_num[myid]
    cdef double m, phvalmass, starttime, endtime
    cdef double* mass2 = <double*>malloc(sizeof(double)*atomnum)

    starttime = mpi.MPI_Wtime()
    for i in range(atomnum):
        mass2[i] = 1.0/sqrt(mass[i])*factor1

    fp = fopen(phvecname,"rb")
    fseek(fp,qnum_s*nmodes2*16,SEEK_SET)
    fread(&phvecval_p[0,0],16,qnum_p*nmodes2,fp)
    fclose(fp)
    for i in range(qnum_p):
        for j in range(nmodes):
            m = mass2[j/3]
            for k in range(nmodes):
                phvalmass = phval[i+qnum_s,k]*m
                phvecval_p[i,j*nmodes+k] *= phvalmass
    free(mass2)
    mpi.MPI_Barrier(c_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("ph_read time:%12.4fs.\n",endtime-starttime)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void phvec_gather(
    mpi.MPI_Comm comm, mpi.MPI_Comm shm_comm, mpi.MPI_Comm node_comm, 
    int myid, int shm_id, int qnum, int nmodes, int nm_num, int nm_min,
    int[::1] qproc, int[::1] qproc_num,
    double complex[:,::1] phvecval_p, double complex* phvecval
):
    cdef int i, j, k
    cdef s_dcplx = sizeof(double complex)
    cdef int atomnum = nmodes/3
    cdef int qnum_s = qproc[myid]
    cdef int qnum_p = qproc_num[myid]
    cdef double starttime, endtime
    cdef double complex* phbuf
    cdef mpi.MPI_Datatype CPLX_N

    starttime = mpi.MPI_Wtime()
    mpi.MPI_Barrier(comm)
    phbuf = <double complex*>malloc(qnum_p*nm_num*nmodes*s_dcplx)
    for i in range(qnum_p):
        for j in range(nmodes):
            for k in range(nm_num):
                phbuf[(i*nmodes+j)*nm_num+k] \
                = phvecval_p[i,j*nmodes+k+nm_min]
    mpi.MPI_Type_contiguous(
        nm_num*nmodes,mpi.MPI_DOUBLE_COMPLEX,&CPLX_N
    )
    mpi.MPI_Type_commit(&CPLX_N)
    mpi.MPI_Gatherv(
        phbuf,qnum_p,CPLX_N,phvecval,
        &qproc_num[0],&qproc[0],CPLX_N,0,comm
    )
    if shm_id == 0:
        mpi.MPI_Bcast(phvecval,qnum,CPLX_N,0,node_comm)

    mpi.MPI_Type_free(&CPLX_N)
    free(phbuf)
    mpi.MPI_Barrier(shm_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("ph_gather time:%12.4fs.\n",endtime-starttime)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void drveck_alltoall(
    mpi.MPI_Comm c_comm, 
    int nprocs, int myid,  int nmodes, int nbands, int knum, int norbital, 
    int nspin, double complex* drveck_a, double complex* drveck_p, 
    int[::1] norb, int[::1] norb_num, int[::1] kproc, int[::1] kproc_num
):
    cdef int i, j, k, N1, N2, offset
    cdef int knum_p = kproc_num[myid]
    cdef int norb_p = norb_num[myid]
    cdef int norbnb = norbital*nbands
    cdef int * scount = <int*>malloc(nprocs*sizeof(int))
    cdef int * sdispl = <int*>malloc(nprocs*sizeof(int))
    cdef int * rcount = <int*>malloc(nprocs*sizeof(int))
    cdef int * rdispl = <int*>malloc(nprocs*sizeof(int))
    cdef double complex* drveck_t = <double complex*>malloc(knum_p*3*nspin*norbnb*sizeof(double complex))

    for i in range(nprocs):
        scount[i] = kproc_num[i]*3*nspin*norb_p*nbands
        rcount[i] = knum_p*3*nspin*norb_num[i]*nbands
        sdispl[i] = kproc[i]*3*nspin*norb_p*nbands
        rdispl[i] = knum_p*3*nspin*norb[i]*nbands
    
    mpi.MPI_Alltoallv(
        drveck_a,scount,sdispl,mpi.MPI_DOUBLE_COMPLEX,
        drveck_t,rcount,rdispl,mpi.MPI_DOUBLE_COMPLEX,c_comm
    )
    for i in range(nprocs):
        N1 = norb[i]*nbands
        N2 = norb_num[i]*nbands
        offset = rdispl[i]
        for j in range(knum_p*3*nspin):
            for k in range(N2):
                drveck_p[j*norbnb+k+N1] = drveck_t[offset+j*N2+k]

    free(drveck_t)
    free(scount)
    free(sdispl)
    free(rcount)
    free(rdispl)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void epckqNL_L(
    double complex* drveck, double complex[:,::1] bandveckp, 
    double complex* phvecval, double complex* vdrvexpikR,
    double complex* fepc, double complex* epcq, int nmodes, int natom, 
    int nm_num, int nbands, int norbital, int nbands2, 
    int norbnb, int nspin, int[::1] norb_u, int[::1] norb_u_num
):
    cdef int i, j, k, l, xyz
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef double complex* bandveckp_t
    cdef double complex* drvexpikR_t
    cdef double complex* vdrvexpikR_t

    for i in range(nmodes*nbands2):
        vdrvexpikR[i] = 0.0
    for xyz in range(3):
        for j in range(nspin):
            for i in range(natom):
                bandveckp_t = &bandveckp[j*norbital+norb_u[i],0]
                drvexpikR_t = &drveck[(xyz*nspin+j)*norbnb+norb_u[i]*nbands]
                vdrvexpikR_t = &vdrvexpikR[(i*3+xyz)*nbands2]
                cblas_zgemm3m(
                    CblasRowMajor,CblasConjTrans,CblasNoTrans,
                    nbands,nbands,norb_u_num[i],&c1,bandveckp_t,
                    nbands,drvexpikR_t,nbands,&c1,vdrvexpikR_t,nbands
                )
    cblas_zgemm3m(
        CblasRowMajor,CblasTrans,CblasNoTrans,
        nm_num,nbands2,nmodes,&c1,phvecval,
        nm_num,vdrvexpikR,nbands2,fepc,epcq,nbands2
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepcNL_L(
    MPI.Comm comm, int nmodes, int natom_loop, long natom_buffer, 
    int norbital, int nbands, int ncell, int knum, int nspin, 
    int[::1] natom_split, int[::1] nq, int[:,::1] R_list, int nkpath,
    double[:,::1] kpath, double complex[:,:,:,::1] drSH,
    double complex[:,:,::1] bandveck, double complex[:,::1] phvecval_p, 
    int[::1] kproc, int[::1] kproc_num, int[::1] norb, int[::1] norb_num, 
    int[::1] norb_u, int[::1] norb_u_num, double complex[:,::1] epc_t
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm shm_comm, remote_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Aint l_drveck
    cdef mpi.MPI_Win win, win1
    cdef double complex* phvecval
    cdef double complex* drSHexpikR
    cdef double complex* drveck_a
    cdef double complex* drveck
    cdef double complex* drveck_t
    cdef double complex* vdrvexpikR
    cdef double complex* expikR
    cdef int natom = nmodes/3
    cdef int norbnb = norbital*nbands
    cdef int nbands2 = nbands*nbands
    cdef int nspin2 = nspin*nspin
    cdef int s_int = sizeof(int)
    cdef int s_dcplx = sizeof(double complex)
    cdef int h, i, j, k, l, ik, xyz, shm_proc_s, shm_proc_e, \
             ierr, myid, nprocs, shm_id, shm_nprocs, nnode
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, kidx, \
             qidx_x, qidx_y, qidx_z, qidx_xy, qidx, \
             kpidx_x, kpidx_y, kpidx_z, kpidx_xy, kpidx, \
             qnum_p, qnum_s, norb_p, norb_s, k1, \
             nm_num, nm_min, nm_max, spin, s_r, s_c
    cdef int * nodelist
    cdef int* dr
    cdef int* dr_num
    cdef double kx, ky, kz, RKx, RKy, RKz, RK, \
                starttime, endtime
    cdef double complex pi2j = M_PI*2j
    cdef double complex c1 = 1.0
    cdef double complex c0 = 0.0

    starttime = mpi.MPI_Wtime()
    # get shm_comm
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&remote_comm)

    dr = <int*>calloc((nnode+1),s_int)
    dr_num = <int*>calloc(nnode,s_int)
    for i in range(nnode):
        shm_proc_s = i*shm_nprocs
        shm_proc_e = (i+1)*shm_nprocs
        for j in range(shm_proc_s,shm_proc_e):
            dr_num[i] += norb_num[j]*nbands
        for j in range(i+1,nnode+1):
            dr[j] += dr_num[i]
    norb_p = norb_num[myid]
    norb_s = norb[myid]
    expikR = <double complex*>malloc(s_dcplx*ncell)
    drSHexpikR = <double complex*>malloc(s_dcplx*norbital*norb_p)
    vdrvexpikR = <double complex*>malloc(s_dcplx*nmodes*nbands2)
    drveck_a = <double complex*>calloc(nkpath*3*nspin*norb_p*nbands,s_dcplx)
    phvecval = <double complex*>malloc(s_dcplx*knum*nmodes*natom_buffer*3)
    if (shm_id == 0):
        l_drveck = s_dcplx*3*nspin*norbnb
    else:
        l_drveck = 0
    mpi.MPI_Win_allocate_shared(
        l_drveck,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&drveck,&win
    )
    if (shm_id != 0):
        mpi.MPI_Win_shared_query(win,0,&l_drveck,&s_dcplx,&drveck)
    mpi.MPI_Barrier(shm_comm)

    qnum_p = kproc_num[myid]
    qnum_s = kproc[myid]

    if norb_p > 0:
        for ik in range(nkpath):
            if nkpath < knum:
                kidx_x = <int>round(nq[0]*kpath[ik,0])
                kidx_y = <int>round(nq[1]*kpath[ik,1])
                kidx_z = <int>round(nq[2]*kpath[ik,2])
            else:
                kidx_z = ik%nq[2]; kidx_xy = ik/nq[2]
                kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
            kidx = (kidx_x*nq[1]+kidx_y)*nq[2]+kidx_z
            kx = (<double>(kidx_x))/(<double>(nq[0]))
            ky = (<double>(kidx_y))/(<double>(nq[1]))
            kz = (<double>(kidx_z))/(<double>(nq[2]))
            for i in range(ncell):
                RKx = R_list[i,0]*kx
                RKy = R_list[i,1]*ky
                RKz = R_list[i,2]*kz
                RK = RKx + RKy + RKz
                expikR[i] = cexp(pi2j*RK)
            for spin in range(nspin2):
                s_r = <int>(spin/nspin)
                s_c = spin%nspin
                for xyz in range(3):
                    drveck_t = &drveck_a[((ik*3+xyz)*nspin+s_r)*norb_p*nbands]
                    # drSH[nR,norb_p*norb]*expikR[nR]
                    cblas_zgemv(
                        CblasRowMajor,CblasTrans,ncell,norb_p*norbital,
                        &c1,&drSH[spin,xyz,0,0],norb_p*norbital,
                        expikR,1,&c0,drSHexpikR,1
                    )
                    cblas_zgemm3m(
                        CblasRowMajor,CblasNoTrans,CblasNoTrans,norb_p,
                        nbands,norbital,&c1,drSHexpikR,norbital,
                        &bandveck[kidx,s_c*norbital,0],nbands,&c1,drveck_t,nbands
                    )
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_NL(Left) part1 time:%12.4fs.\n",endtime-starttime)
    for h in range(natom_loop):
        nm_min = natom_split[h]*3
        nm_max = natom_split[h+1]*3
        nm_num = nm_max-nm_min
        starttime = mpi.MPI_Wtime()
        for ik in range(nkpath):
            start = mpi.MPI_Wtime()
            if nkpath < knum:
                kidx_x = <int>round(nq[0]*kpath[ik,0])
                kidx_y = <int>round(nq[1]*kpath[ik,1])
                kidx_z = <int>round(nq[2]*kpath[ik,2])
            else:
                kidx_z = ik%nq[2]; kidx_xy = ik/nq[2]
                kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
            mpi.MPI_Barrier(shm_comm)
            for xyz in range(3):
                for j in range(nspin):
                    for k in range(norb_p*nbands):
                        drveck[(xyz*nspin+j)*norbnb+norb_s*nbands+k] \
                        = drveck_a[((ik*3+xyz)*nspin+j)*norb_p*nbands+k]
                    if nnode > 1:
                        mpi.MPI_Barrier(shm_comm)
                        if (shm_id == 0):
                            mpi.MPI_Allgatherv(
                                mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,
                                &drveck[(xyz*nspin+j)*norbnb],&dr_num[0],
                                &dr[0],mpi.MPI_DOUBLE_COMPLEX,remote_comm
                            )
            mpi.MPI_Barrier(shm_comm)
            for i in range(qnum_p):
                k1 = i+qnum_s
                for j in range(nmodes):
                    for k in range(nm_num):
                        phvecval[j*nm_num+k] \
                        = phvecval_p[i,j*nmodes+k+nm_min]
                qidx_z = k1%nq[2]; qidx_xy = k1/nq[2]
                qidx_y = qidx_xy%nq[1]; qidx_x = qidx_xy/nq[1]
                kpidx_x = (qidx_x+kidx_x)%nq[0]
                kpidx_y = (qidx_y+kidx_y)%nq[1]
                kpidx_z = (qidx_z+kidx_z)%nq[2]
                if kpidx_x<0: kpidx_x += nq[0]
                if kpidx_y<0: kpidx_y += nq[1]
                if kpidx_z<0: kpidx_z += nq[2]
                kpidx = (kpidx_x*nq[1]+kpidx_y)*nq[2]+kpidx_z
                epckqNL_L(
                    drveck,bandveck[kpidx],phvecval,vdrvexpikR,&c1,
                    &epc_t[ik*qnum_p+i,nm_min*nbands2],nmodes,natom,nm_num,
                    nbands,norbital,nbands2,norbnb,nspin,norb_u,norb_u_num
                )
            end = mpi.MPI_Wtime()
            if myid == 0:
                printf("time in mode loop %d, knum loop %d:%12.4fs.\n",h,ik,end-start)
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("epc_NL(Left) part2 time in mode[%4d,%4d]:%12.4fs.\n",nm_min,nm_min+nm_num,endtime-starttime)

    free(drSHexpikR)
    free(vdrvexpikR)
    free(expikR)
    free(nodelist)
    free(dr)
    free(dr_num)
    free(drveck_a)
    free(phvecval)
    mpi.MPI_Win_free(&win)
    mpi.MPI_Barrier(c_comm)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void epckqNL_R(
    double complex* drveck, double complex[:,::1] bandveck,
    double complex* phvecval, double complex* vdrv, 
    double complex* vdrvexpikR, double complex* fepc, double complex* epcq, 
    int nmodes, int natom, int nm_num, int nbands, int norbital, 
    int nbands2, int norbnb, int nspin, int[::1] norb_u, int[::1] norb_u_num
):
    cdef int i, j, k, l, xyz
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef double complex* bandveck_t
    cdef double complex* drvexpikR_t
    cdef double complex* vdrvexpikR_t

    for i in range(nmodes*nbands2):
        vdrvexpikR[i] = 0.0
    for xyz in range(3):
        for l in range(nspin):
            for i in range(natom):
                bandveck_t = &bandveck[l*norbital+norb_u[i],0]
                drvexpikR_t = &drveck[(xyz*nspin+l)*norbnb+norb_u[i]*nbands]
                vdrvexpikR_t = &vdrvexpikR[(i*3+xyz)*nbands2]
                cblas_zgemm3m(
                    CblasRowMajor,CblasTrans,CblasNoTrans,
                    nbands,nbands,norb_u_num[i],&c1,bandveck_t,
                    nbands,drvexpikR_t,nbands,&c0,vdrv,nbands
                )
                for j in range(nbands):
                    for k in range(nbands):
                        vdrvexpikR_t[k*nbands+j] += vdrv[j*nbands+k]
    cblas_zgemm3m(
        CblasRowMajor,CblasTrans,CblasNoTrans,
        nm_num,nbands2,nmodes,&c1,phvecval,
        nm_num,vdrvexpikR,nbands2,fepc,epcq,nbands2
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepcNL_R(
    MPI.Comm comm, int nmodes, int natom_loop, int natom_buffer, 
    int norbital, int nbands, int ncell, int knum, int nspin, 
    int[::1] natom_split, int[::1] nq, int[:,::1] R_list, int nkpath, 
    double[:,::1] kpath, double complex[:,:,:,::1] drSH,
    double complex[:,:,::1] bandveck, double complex[:,::1] phvecval_p,
    int[::1] kproc, int[::1] kproc_num, int[::1] norb, int[::1] norb_num,
    int[::1] norb_u, int[::1] norb_u_num, double complex[:,::1] epc_t
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm shm_comm, remote_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Win win, win1
    cdef double complex* phvecval
    cdef double complex* bandveckp
    cdef double complex* drSHexpikR
    cdef double complex* drveck
    cdef double complex* drveck_p
    cdef double complex* vdrv
    cdef double complex* vdrvexpikR
    cdef double complex* expikR
    cdef int natom = nmodes/3
    cdef int norbnb = norbital*nbands
    cdef int nbands2 = nbands*nbands
    cdef int s_int = sizeof(int)
    cdef int s_dcplx = sizeof(double complex)
    cdef int h, i, j, k, l, ik, xyz, shm_proc_s, shm_proc_e, \
             ierr, myid, nprocs, shm_id, shm_nprocs, nnode
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, kidx, \
             qidx_x, qidx_y, qidx_z, qidx_xy, qidx, \
             kpidx_x, kpidx_y, kpidx_z, kpidx_xy, kpidx, \
             qnum_p, qnum_s, norb_p, norb_s, k1, \
             nm_num, nm_min, nm_max, spin, s_r, s_c
    cdef int * nodelist
    cdef int* dr
    cdef int* dr_num
    cdef double kpx, kpy, kpz, RKpx, RKpy, RKpz, RKp, \
                starttime, endtime
    cdef double complex pi2j = M_PI*2j
    cdef double complex c1 = 1.0
    cdef double complex c0 = 0.0

    starttime = mpi.MPI_Wtime()
    # get shm_comm
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&remote_comm)

    dr = <int*>calloc((nnode+1),s_int)
    dr_num = <int*>calloc(nnode,s_int)
    for i in range(nnode):
        shm_proc_s = i*shm_nprocs
        shm_proc_e = (i+1)*shm_nprocs
        for j in range(shm_proc_s,shm_proc_e):
            dr_num[i] += norb_num[j]*nbands
        for j in range(i+1,nnode+1):
            dr[j] += dr_num[i]
    norb_p = norb_num[myid]
    norb_s = norb[myid]
    qnum_p = kproc_num[myid]
    qnum_s = kproc[myid]
    bandveckp = <double complex*>malloc(s_dcplx*nspin*norbnb)
    expikR = <double complex*>malloc(s_dcplx*ncell)
    drSHexpikR = <double complex*>malloc(s_dcplx*norbital*norb_p)
    vdrv = <double complex*>malloc(s_dcplx*nbands2)
    vdrvexpikR = <double complex*>malloc(s_dcplx*nmodes*nbands2)
    drveck_p = <double complex*>malloc(s_dcplx*norb_p*nbands)
    phvecval = <double complex*>malloc(s_dcplx*knum*nmodes*natom_buffer*3)

    if (shm_id == 0):
        l_drveck = s_dcplx*knum*3*nspin*norbnb
    else:
        l_drveck = 0
    mpi.MPI_Win_allocate_shared(
        l_drveck,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&drveck,&win
    )
    if (shm_id != 0):
        mpi.MPI_Win_shared_query(win,0,&l_drveck,&s_dcplx,&drveck)
    mpi.MPI_Barrier(shm_comm)

    if norb_p > 0:
        for h in range(knum):
            kpidx_z = h%nq[2]; kpidx_xy = h/nq[2]
            kpidx_y = kpidx_xy%nq[1]; kpidx_x = kpidx_xy/nq[1]
            kpidx = (kpidx_x*nq[1]+kpidx_y)*nq[2]+kpidx_z
            kpx = (<double>(kpidx_x))/(<double>(nq[0]))
            kpy = (<double>(kpidx_y))/(<double>(nq[1]))
            kpz = (<double>(kpidx_z))/(<double>(nq[2]))
            for i in range(ncell):
                RKpx = R_list[i,0]*kpx
                RKpy = R_list[i,1]*kpy
                RKpz = R_list[i,2]*kpz
                RKp = RKpx + RKpy + RKpz
                expikR[i] = cexp(-1.0*pi2j*RKp)
            for i in range(nspin*norbital):
                for j in range(nbands):
                    bandveckp[i*nbands+j] = conj(bandveck[kpidx,i,j])
            for s_r in range(nspin):
                for xyz in range(3):
                    l = (h*3+xyz)*nspin+s_r
                    memset(drveck_p,0,s_dcplx*norb_p*nbands)
                    for s_c in range(nspin):
                        spin = s_r*nspin+s_c
                        # drSH[nR,norb_p*norb]*expikR[nR]
                        cblas_zgemv(
                            CblasRowMajor,CblasTrans,ncell,norb_p*norbital,
                            &c1,&drSH[spin,xyz,0,0],norb_p*norbital,
                            expikR,1,&c0,drSHexpikR,1
                        )
                        cblas_zgemm3m(
                            CblasRowMajor,CblasNoTrans,CblasNoTrans,norb_p,
                            nbands,norbital,&c1,drSHexpikR,norbital,
                            &bandveckp[s_c*norbnb],nbands,&c1,drveck_p,nbands
                        )
                    # drveck_p -> drveck (gather to node comm)
                    for j in range(norb_p*nbands):
                        drveck[l*norbnb+norb_s*nbands+j] = drveck_p[j]
                    if nnode > 1:
                        mpi.MPI_Barrier(shm_comm)
                        if (shm_id == 0):
                            mpi.MPI_Allgatherv(
                                mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,
                                &drveck[l*norbnb],&dr_num[0],&dr[0],
                                mpi.MPI_DOUBLE_COMPLEX,remote_comm
                            )
    mpi.MPI_Barrier(shm_comm)
    free(drveck_p)

    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_NL(Right) part1 time:%12.4fs.\n",endtime-starttime)

    for h in range(natom_loop):
        nm_min = natom_split[h]*3
        nm_max = natom_split[h+1]*3
        nm_num = nm_max-nm_min
        starttime = mpi.MPI_Wtime()
        for ik in range(nkpath):
            start = mpi.MPI_Wtime()
            if nkpath < knum:
                kidx_x = <int>round(nq[0]*kpath[ik,0])
                kidx_y = <int>round(nq[1]*kpath[ik,1])
                kidx_z = <int>round(nq[2]*kpath[ik,2])
            else:
                kidx_z = ik%nq[2]; kidx_xy = ik/nq[2]
                kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
            kidx = (kidx_x*nq[1]+kidx_y)*nq[2]+kidx_z
            for l in range(qnum_p):
                k1 = l+qnum_s
                qidx_z = k1%nq[2]; qidx_xy = k1/nq[2]
                qidx_y = qidx_xy%nq[1]; qidx_x = qidx_xy/nq[1]
                kpidx_x = (kidx_x+qidx_x)%nq[0]
                kpidx_y = (kidx_y+qidx_y)%nq[1]
                kpidx_z = (kidx_z+qidx_z)%nq[2]
                if kpidx_x<0: kpidx_x += nq[0]
                if kpidx_y<0: kpidx_y += nq[1]
                if kpidx_z<0: kpidx_z += nq[2]
                kpidx = (kpidx_x*nq[1]+kpidx_y)*nq[2]+kpidx_z
                for j in range(nmodes):
                    for k in range(nm_num):
                        phvecval[j*nm_num+k] \
                        = phvecval_p[l,j*nmodes+k+nm_min]
                epckqNL_R(
                    &drveck[kpidx*3*nspin*norbnb],bandveck[kidx],phvecval,vdrv,
                    vdrvexpikR,&c1,&epc_t[ik*qnum_p+l,nm_min*nbands2],nmodes,natom,
                    nm_num,nbands,norbital,nbands2,norbnb,nspin,norb_u,norb_u_num
                )
            end = mpi.MPI_Wtime()
            if myid == 0:
                printf("time in mode loop %d, knum loop %d:%12.4fs.\n",h,ik,end-start)
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("epc_NL(Right) part2 time in mode[%4d,%4d]:%12.4fs.\n",nm_min,nm_min+nm_num,endtime-starttime)

    free(bandveckp)
    free(drSHexpikR)
    free(vdrv)
    free(vdrvexpikR)
    free(expikR)
    free(nodelist)
    free(dr)
    free(dr_num)
    free(phvecval)
    mpi.MPI_Win_free(&win)
    mpi.MPI_Barrier(c_comm)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void epckqNL_p_L(
    double complex* drveck, double complex* bandveckp, 
    double complex* phvecval, double complex* vdrvexpikR,
    double complex* fepc, double complex* epcq, int nmodes, int natom, 
    int nm_num, int norbital, int nspin, int[::1] norb_u, int[::1] norb_u_num
):
    cdef int i, j, k, l, xyz
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef double complex vdrvexpikR_t
    cdef double complex* drvexpikR_t

    for i in range(nmodes):
        vdrvexpikR[i] = 0.0
    for xyz in range(3):
        for j in range(nspin):
            for i in range(natom):
                drvexpikR_t = &drveck[(xyz*nspin+j)*norbital+norb_u[i]]
                cblas_zdotc_sub(
                    norb_u_num[i],&bandveckp[j*norbital+norb_u[i]],1,
                    drvexpikR_t,1,&vdrvexpikR_t
                )
                vdrvexpikR[i*3+xyz] += vdrvexpikR_t
    cblas_zgemv(
        CblasRowMajor,CblasTrans,nmodes,
        nm_num,&c1,phvecval,nm_num,
        vdrvexpikR,1,fepc,epcq,1
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepcNL_p_L(
    MPI.Comm comm, int nmodes, int natom_loop, int natom_buffer, 
    int norbital, int ncell, int knum, int qnum, int nspin, 
    int[::1] natom_split, int[::1] nq, int[:,::1] R_list, 
    double complex[:,:,:,::1] drSH, double complex[:,::1] bandveck, 
    double complex[:,::1] phvecval_p, int[::1] kproc, int[::1] kproc_num, 
    int[::1] qproc, int[::1] qproc_num, int[:,::1] bassel, int[::1] norb, 
    int[::1] norb_num, int[::1] norb_u, int[::1] norb_u_num, 
    double complex[:,::1] epc_t
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm shm_comm, remote_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Aint l_drveck
    cdef mpi.MPI_Win win, win1
    cdef double complex* phvecval
    cdef double complex* drSHexpikR
    cdef double complex* drveck
    cdef double complex* drveck_a
    cdef double complex* drveck_t
    cdef double complex* vdrvexpikR
    cdef double complex* expikR
    cdef int natom = nmodes/3
    cdef int nspin2 = nspin*nspin
    cdef int s_int = sizeof(int)
    cdef int s_dcplx = sizeof(double complex)
    cdef int h, i, j, k, l, m, xyz, shm_proc_s, shm_proc_e, \
             ierr, myid, nprocs, shm_id, shm_nprocs, nnode
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, kidx, \
             qidx_x, qidx_y, qidx_z, qidx_xy, qidx, \
             kpidx_x, kpidx_y, kpidx_z, kpidx_xy, kpidx, \
             knum_p, knum_s, norb_p, norb_s, k1, k2, \
             nm_min, nm_max, nm_num, spin, s_r, s_c
    cdef int * nodelist
    cdef int* dr
    cdef int* dr_num
    cdef double kx, ky, kz, RKx, RKy, RKz, RK, \
                starttime, endtime
    cdef double complex pi2j = M_PI*2j
    cdef double complex c1 = 1.0
    cdef double complex c0 = 0.0

    starttime = mpi.MPI_Wtime()
    # get shm_comm
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&remote_comm)

    dr = <int*>calloc((nnode+1),s_int)
    dr_num = <int*>calloc(nnode,s_int)
    for i in range(nnode):
        shm_proc_s = i*shm_nprocs
        shm_proc_e = (i+1)*shm_nprocs
        for j in range(shm_proc_s,shm_proc_e):
            dr_num[i] += norb_num[j]
        for j in range(i+1,nnode+1):
            dr[j] += dr_num[i]

    norb_p = norb_num[myid]
    norb_s = norb[myid]
    expikR = <double complex*>malloc(s_dcplx*ncell)
    drSHexpikR = <double complex*>malloc(s_dcplx*norbital*norb_p)
    vdrvexpikR = <double complex*>malloc(s_dcplx*nmodes)
    drveck_a = <double complex*>calloc(knum*3*nspin*norb_p,s_dcplx)
    if (shm_id == 0):
        l_phvecval = s_dcplx*qnum*natom_buffer*3*nmodes
        l_drveck = s_dcplx*3*nspin*norbital
    else:
        l_phvecval = 0
        l_drveck = 0
    mpi.MPI_Win_allocate_shared(
        l_drveck,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&drveck,&win
    )
    mpi.MPI_Win_allocate_shared(
        l_phvecval,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&phvecval,&win1
    )
    if (shm_id != 0):
        mpi.MPI_Win_shared_query(win,0,&l_drveck,&s_dcplx,&drveck)
        mpi.MPI_Win_shared_query(win1,0,&l_phvecval,&s_dcplx,&phvecval)
    mpi.MPI_Barrier(shm_comm)

    knum_p = kproc_num[myid]
    knum_s = kproc[myid]

    if norb_p > 0:
        for h in range(knum):
            k1 = bassel[h,0]
            kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
            kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
            kx = (<double>(kidx_x))/(<double>(nq[0]))
            ky = (<double>(kidx_y))/(<double>(nq[1]))
            kz = (<double>(kidx_z))/(<double>(nq[2]))
            for i in range(ncell):
                RKx = R_list[i,0]*kx
                RKy = R_list[i,1]*ky
                RKz = R_list[i,2]*kz
                RK = RKx + RKy + RKz
                expikR[i] = cexp(pi2j*RK)
            for spin in range(nspin2):
                s_r = <int>(spin/nspin)
                s_c = spin%nspin
                for xyz in range(3):
                    drveck_t = &drveck_a[((h*3+xyz)*nspin+s_r)*norb_p]
                    # drSH[nR,norb_p*norb]*expikR[nR]
                    cblas_zgemv(
                        CblasRowMajor,CblasTrans,ncell,norb_p*norbital,
                        &c1,&drSH[spin,xyz,0,0],norb_p*norbital,
                        expikR,1,&c0,drSHexpikR,1
                    )
                    cblas_zgemv(
                        CblasRowMajor,CblasNoTrans,norb_p,
                        norbital,&c1,drSHexpikR,norbital,
                        &bandveck[h,s_c*norbital],1,&c1,drveck_t,1
                    )
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_p_NL(Left) part1 time:%12.4fs.\n",endtime-starttime)

    for h in range(natom_loop):
        nm_min = natom_split[h]*3
        nm_max = natom_split[h+1]*3
        nm_num = nm_max-nm_min
        phvec_gather(
            c_comm,shm_comm,remote_comm,myid,shm_id,
            qnum,nmodes,nm_num,nm_min,
            qproc,qproc_num,phvecval_p,phvecval
        )
        starttime = mpi.MPI_Wtime()
        for l in range(knum):
            start = mpi.MPI_Wtime()
            k1 = bassel[l,0]
            kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
            kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
            mpi.MPI_Barrier(shm_comm)
            for xyz in range(3):
                for j in range(nspin):
                    for k in range(norb_p):
                        drveck[(xyz*nspin+j)*norbital+norb_s+k] \
                        = drveck_a[((l*3+xyz)*nspin+j)*norb_p+k]
                    if nnode > 1:
                        mpi.MPI_Barrier(shm_comm)
                        if (shm_id == 0):
                            mpi.MPI_Allgatherv(
                                mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,
                                &drveck[(xyz*nspin+j)*norbital],&dr_num[0],
                                &dr[0],mpi.MPI_DOUBLE_COMPLEX,remote_comm
                            )
            mpi.MPI_Barrier(shm_comm)
            for i in range(knum_p):
                k2 = bassel[i+knum_s,0]
                kpidx_z = k2%nq[2]; kpidx_xy = k2/nq[2]
                kpidx_y = kpidx_xy%nq[1]; kpidx_x = kpidx_xy/nq[1]
                qidx_x = (kpidx_x-kidx_x)%nq[0]
                qidx_y = (kpidx_y-kidx_y)%nq[1]
                qidx_z = (kpidx_z-kidx_z)%nq[2]
                if qidx_x<0: qidx_x += nq[0]
                if qidx_y<0: qidx_y += nq[1]
                if qidx_z<0: qidx_z += nq[2]
                qidx = (qidx_x*nq[1]+qidx_y)*nq[2]+qidx_z
                epckqNL_p_L(
                    drveck,&bandveck[i+knum_s,0],&phvecval[qidx*nm_num*nmodes],
                    vdrvexpikR,&c1,&epc_t[l*knum_p+i,nm_min],nmodes,
                    natom,nm_num,norbital,nspin,norb_u,norb_u_num
                )
            end = mpi.MPI_Wtime()
            if myid == 0:
                printf("time in mode loop %d, knum loop %d:%12.4fs.\n",h,l,end-start)
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("epc_p_NL(Left) part2 time in mode[%4d,%4d]:%12.4fs.\n",nm_min,nm_min+nm_num,endtime-starttime)

    free(drSHexpikR)
    free(vdrvexpikR)
    free(expikR)
    free(nodelist)
    free(dr)
    free(dr_num)
    free(drveck_a)
    mpi.MPI_Win_free(&win)
    mpi.MPI_Win_free(&win1)
    mpi.MPI_Barrier(c_comm)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void epckqNL_p_R(
    double complex* drveck, double complex* bandveck,
    double complex* phvecval, double complex* vdrvexpikR,
    double complex* fepc, double complex* epcq, int nmodes, int natom, 
    int nm_num, int norbital, int nspin, int[::1] norb_u, int[::1] norb_u_num
):
    cdef int i, j, k, l, xyz
    cdef double complex c0 = 0.0
    cdef double complex c1 = 1.0
    cdef double complex vdrvexpikR_t
    cdef double complex* drvexpikR_t

    for i in range(nmodes):
        vdrvexpikR[i] = 0.0
    for xyz in range(3):
        for j in range(nspin):
            for i in range(natom):
                drvexpikR_t = &drveck[(xyz*nspin+j)*norbital+norb_u[i]]
                cblas_zdotu_sub(
                    norb_u_num[i],&bandveck[j*norbital+norb_u[i]],1,
                    drvexpikR_t,1,&vdrvexpikR_t
                )
                vdrvexpikR[i*3+xyz] += vdrvexpikR_t
    cblas_zgemv(
        CblasRowMajor,CblasTrans,nmodes,
        nm_num,&c1,phvecval,nm_num,
        vdrvexpikR,1,fepc,epcq,1
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepcNL_p_R(
    MPI.Comm comm, int nmodes, int natom_loop, int natom_buffer, 
    int norbital, int ncell, int knum, int qnum, int nspin, 
    int[::1] natom_split, int[::1] nq, int[:,::1] R_list, 
    double complex[:,:,:,::1] drSH, double complex[:,::1] bandveck,
    double complex[:,::1] phvecval_p, int[::1] kproc, int[::1] kproc_num, 
    int[::1] qproc, int[::1] qproc_num, int[:,::1] bassel, int[::1] norb, 
    int[::1] norb_num, int[::1] norb_u, int[::1] norb_u_num, 
    double complex[:,::1] epc_t
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm shm_comm, remote_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Win win, win1
    cdef double complex* phvecval
    cdef double complex* bandveckp
    cdef double complex* drSHexpikR
    cdef double complex* drveck_p
    cdef double complex* drveck_a
    cdef double complex* drveck_t
    cdef double complex* vdrvexpikR
    cdef double complex* expikR
    cdef int natom = nmodes/3
    cdef int nspin2 = nspin*nspin
    cdef int s_int = sizeof(int)
    cdef int s_dcplx = sizeof(double complex)
    cdef int h, i, j, k, l, m, xyz, shm_proc_s, shm_proc_e, \
             ierr, myid, nprocs, shm_id, shm_nprocs, nnode
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, kidx, \
             qidx_x, qidx_y, qidx_z, qidx_xy, qidx, \
             kpidx_x, kpidx_y, kpidx_z, kpidx_xy, kpidx, \
             knum_p, knum_s, norb_p, norb_s, k1, k2, \
             nm_min, nm_max, nm_num, spin, s_r, s_c
    cdef int * nodelist
    cdef int* dr
    cdef int* dr_num
    cdef double kpx, kpy, kpz, RKpx, RKpy, RKpz, RKp, \
                starttime, endtime
    cdef double complex pi2j = M_PI*2j
    cdef double complex c1 = 1.0
    cdef double complex c0 = 0.0

    starttime = mpi.MPI_Wtime()
    # get shm_comm
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&remote_comm)

    dr = <int*>calloc((nnode+1),s_int)
    dr_num = <int*>calloc(nnode,s_int)
    for i in range(nnode):
        shm_proc_s = i*shm_nprocs
        shm_proc_e = (i+1)*shm_nprocs
        for j in range(shm_proc_s,shm_proc_e):
            dr_num[i] += norb_num[j]
        for j in range(i+1,nnode+1):
            dr[j] += dr_num[i]

    norb_p = norb_num[myid]
    norb_s = norb[myid]
    bandveckp = <double complex*>malloc(s_dcplx*nspin*norbital)
    expikR = <double complex*>malloc(s_dcplx*ncell)
    drSHexpikR = <double complex*>malloc(s_dcplx*norbital*norb_p)
    vdrvexpikR = <double complex*>malloc(s_dcplx*nmodes)
    drveck_a = <double complex*>calloc(knum*3*nspin*norb_p,s_dcplx)
    if (shm_id == 0):
        l_phvecval = s_dcplx*qnum*natom_buffer*3*nmodes
    else:
        l_phvecval = 0
    mpi.MPI_Win_allocate_shared(
        l_phvecval,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&phvecval,&win1
    )
    if (shm_id != 0):
        mpi.MPI_Win_shared_query(win1,0,&l_phvecval,&s_dcplx,&phvecval)
    mpi.MPI_Barrier(shm_comm)

    knum_p = kproc_num[myid]
    knum_s = kproc[myid]

    if norb_p > 0:
        for h in range(knum):
            k1 = bassel[h,0]
            kpidx_z = k1%nq[2]; kpidx_xy = k1/nq[2]
            kpidx_y = kpidx_xy%nq[1]; kpidx_x = kpidx_xy/nq[1]
            kpx = (<double>(kpidx_x))/(<double>(nq[0]))
            kpy = (<double>(kpidx_y))/(<double>(nq[1]))
            kpz = (<double>(kpidx_z))/(<double>(nq[2]))
            for i in range(ncell):
                RKpx = R_list[i,0]*kpx
                RKpy = R_list[i,1]*kpy
                RKpz = R_list[i,2]*kpz
                RKp = RKpx + RKpy + RKpz
                expikR[i] = cexp(-1.0*pi2j*RKp)
            for i in range(nspin*norbital):
                bandveckp[i] = conj(bandveck[h,i])
            for spin in range(nspin2):
                s_r = <int>(spin/nspin)
                s_c = spin%nspin
                for xyz in range(3):
                    drveck_t = &drveck_a[((h*3+xyz)*nspin+s_r)*norb_p]
                    # drSH[nR,norb_p*norb]*expikR[nR]
                    cblas_zgemv(
                        CblasRowMajor,CblasTrans,ncell,norb_p*norbital,
                        &c1,&drSH[spin,xyz,0,0],norb_p*norbital,
                        expikR,1,&c0,drSHexpikR,1
                    )
                    cblas_zgemv(
                        CblasRowMajor,CblasNoTrans,norb_p,
                        norbital,&c1,drSHexpikR,norbital,
                        &bandveckp[s_c*norbital],1,&c1,drveck_t,1
                    )
    # drveck_a -> drveck_p (alltoallv)
    drveck_p = <double complex*>malloc(s_dcplx*knum_p*3*nspin*norbital)
    drveck_alltoall(
        c_comm,nprocs,myid,nmodes,1,knum,norbital,nspin,
        drveck_a,drveck_p,norb,norb_num,kproc,kproc_num
    )
    free(drveck_a)

    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_p_NL(Right) part1 time:%12.4fs.\n",endtime-starttime)
    for h in range(natom_loop):
        nm_min = natom_split[h]*3
        nm_max = natom_split[h+1]*3
        nm_num = nm_max-nm_min
        phvec_gather(
            c_comm,shm_comm,remote_comm,myid,shm_id,
            qnum,nmodes,nm_num,nm_min,
            qproc,qproc_num,phvecval_p,phvecval
        )
        starttime = mpi.MPI_Wtime()
        for l in range(knum):
            start = mpi.MPI_Wtime()
            k1 = bassel[l,0]
            kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
            kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
            for i in range(knum_p):
                k2 = bassel[i+knum_s,0]
                kpidx_z = k2%nq[2]; kpidx_xy = k2/nq[2]
                kpidx_y = kpidx_xy%nq[1]; kpidx_x = kpidx_xy/nq[1]
                kpidx = (kpidx_x*nq[1]+kpidx_y)*nq[2]+kpidx_z
                qidx_x = (kpidx_x-kidx_x)%nq[0]
                qidx_y = (kpidx_y-kidx_y)%nq[1]
                qidx_z = (kpidx_z-kidx_z)%nq[2]
                if qidx_x<0: qidx_x += nq[0]
                if qidx_y<0: qidx_y += nq[1]
                if qidx_z<0: qidx_z += nq[2]
                qidx = (qidx_x*nq[1]+qidx_y)*nq[2]+qidx_z
                epckqNL_p_R(
                    &drveck_p[i*3*nspin*norbital],&bandveck[l,0],
                    &phvecval[qidx*nm_num*nmodes],vdrvexpikR,&c1,
                    &epc_t[l*knum_p+i,nm_min],nmodes,natom,nm_num,
                    norbital,nspin,norb_u,norb_u_num
                )
            end = mpi.MPI_Wtime()
            if myid == 0:
                printf("time in mode loop %d, knum loop %d:%12.4fs.\n",h,l,end-start)
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("epc_p_NL(Right) part2 time in mode[%4d,%4d]:%12.4fs.\n",nm_min,nm_min+nm_num,endtime-starttime)

    free(bandveckp)
    free(drSHexpikR)
    free(vdrvexpikR)
    free(expikR)
    free(nodelist)
    free(dr)
    free(dr_num)
    free(drveck_p)
    mpi.MPI_Win_free(&win1)
    mpi.MPI_Barrier(c_comm)


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepcNL_L_q(
    MPI.Comm comm, int nmodes, int natom_loop, long natom_buffer,
    int norbital, int nbands, int ncell, int knum, int nspin, 
    int[::1] natom_split, int[::1] nq, int[:,::1] R_list, int nqpath,
    double[:,::1] qpath, double complex[:,:,:,::1] drSH,
    double complex[:,:,::1] bandveck, double complex[:,::1] phvecval_p,
    int[::1] kproc, int[::1] kproc_num, int[::1] norb, int[::1] norb_num,
    int[::1] norb_u, int[::1] norb_u_num, double complex[:,::1] epc_t
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm shm_comm, remote_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Win win, win1
    cdef double complex* phvecval
    cdef double complex* drSHexpikR
    cdef double complex* drveck_a
    cdef double complex* drveck_p
    cdef double complex* drveck_t
    cdef double complex* vdrvexpikR
    cdef double complex* expikR
    cdef int natom = nmodes/3
    cdef int norbnb = norbital*nbands
    cdef int nbands2 = nbands*nbands
    cdef int nspin2 = nspin*nspin
    cdef int s_int = sizeof(int)
    cdef int s_dcplx = sizeof(double complex)
    cdef int h, i, j, k, l, iq, xyz, shm_proc_s, shm_proc_e, \
             ierr, myid, nprocs, shm_id, shm_nprocs, nnode
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, kidx, \
             qidx_x, qidx_y, qidx_z, qidx_xy, qidx, \
             kpidx_x, kpidx_y, kpidx_z, kpidx_xy, kpidx, \
             knum_p, knum_s, norb_p, norb_s, k1, \
             nm_num, nm_min, nm_max, spin, s_r, s_c
    cdef int * nodelist
    cdef int* dr
    cdef int* dr_num
    cdef double kx, ky, kz, RKx, RKy, RKz, RK, \
                starttime, endtime
    cdef double complex pi2j = M_PI*2j
    cdef double complex c1 = 1.0
    cdef double complex c0 = 0.0

    starttime = mpi.MPI_Wtime()
    # get shm_comm
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&remote_comm)

    dr = <int*>calloc((nnode+1),s_int)
    dr_num = <int*>calloc(nnode,s_int)
    for i in range(nnode):
        shm_proc_s = i*shm_nprocs
        shm_proc_e = (i+1)*shm_nprocs
        for j in range(shm_proc_s,shm_proc_e):
            dr_num[i] += norb_num[j]*nbands
        for j in range(i+1,nnode+1):
            dr[j] += dr_num[i]
    norb_p = norb_num[myid]
    norb_s = norb[myid]
    expikR = <double complex*>malloc(s_dcplx*ncell)
    drSHexpikR = <double complex*>malloc(s_dcplx*norbital*norb_p)
    vdrvexpikR = <double complex*>malloc(s_dcplx*nmodes*nbands2)
    drveck_a = <double complex*>calloc(knum*3*nspin*norb_p*nbands,s_dcplx)
    if (shm_id == 0):
        l_phvecval = s_dcplx*knum*natom_buffer*3*nmodes
    else:
        l_phvecval = 0
    mpi.MPI_Win_allocate_shared(
        l_phvecval,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&phvecval,&win1
    )
    if (shm_id != 0):
        mpi.MPI_Win_shared_query(win1,0,&l_phvecval,&s_dcplx,&phvecval)
    mpi.MPI_Barrier(shm_comm)

    knum_p = kproc_num[myid]
    knum_s = kproc[myid]

    if norb_p > 0:
        for h in range(knum):
            kidx_z = h%nq[2]; kidx_xy = h/nq[2]
            kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
            kx = (<double>(kidx_x))/(<double>(nq[0]))
            ky = (<double>(kidx_y))/(<double>(nq[1]))
            kz = (<double>(kidx_z))/(<double>(nq[2]))
            for i in range(ncell):
                RKx = R_list[i,0]*kx
                RKy = R_list[i,1]*ky
                RKz = R_list[i,2]*kz
                RK = RKx + RKy + RKz
                expikR[i] = cexp(pi2j*RK)
            for spin in range(nspin2):
                s_r = <int>(spin/nspin)
                s_c = spin%nspin
                for xyz in range(3):
                    drveck_t = &drveck_a[((h*3+xyz)*nspin+s_r)*norb_p*nbands]
                    # drSH[nR,norb_p*norb]*expikR[nR]
                    cblas_zgemv(
                        CblasRowMajor,CblasTrans,ncell,norb_p*norbital,
                        &c1,&drSH[spin,xyz,0,0],norb_p*norbital,
                        expikR,1,&c0,drSHexpikR,1
                    )
                    cblas_zgemm3m(
                        CblasRowMajor,CblasNoTrans,CblasNoTrans,norb_p,
                        nbands,norbital,&c1,drSHexpikR,norbital,
                        &bandveck[h,s_c*norbital,0],nbands,&c1,drveck_t,nbands
                    )
    # drveck_a -> drveck_p (alltoallv)
    drveck_p = <double complex*>malloc(s_dcplx*knum_p*3*nspin*norbnb)
    drveck_alltoall(
        c_comm,nprocs,myid,nmodes,nbands,knum,norbital,nspin,
        drveck_a,drveck_p,norb,norb_num,kproc,kproc_num
    )
    free(drveck_a)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_NL(Left)_q part1 time:%12.4fs.\n",endtime-starttime)
    for h in range(natom_loop):
        nm_min = natom_split[h]*3
        nm_max = natom_split[h+1]*3
        nm_num = nm_max-nm_min
        phvec_gather(
            c_comm,shm_comm,remote_comm,myid,shm_id,
            knum,nmodes,nm_num,nm_min,
            kproc,kproc_num,phvecval_p,phvecval
        )
        starttime = mpi.MPI_Wtime()
        for iq in range(nqpath):
            start = mpi.MPI_Wtime()
            if nqpath < knum:
                qidx_x = <int>round(nq[0]*qpath[iq,0])
                qidx_y = <int>round(nq[1]*qpath[iq,1])
                qidx_z = <int>round(nq[2]*qpath[iq,2])
            else:
                qidx_z = iq%nq[2]; qidx_xy = iq/nq[2]
                qidx_y = qidx_xy%nq[1]; qidx_x = qidx_xy/nq[1]
            qidx = (qidx_x*nq[1]+qidx_y)*nq[2]+qidx_z
            for l in range(knum_p):
                k1 = l+knum_s
                kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
                kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
                kpidx_x = (qidx_x+kidx_x)%nq[0]
                kpidx_y = (qidx_y+kidx_y)%nq[1]
                kpidx_z = (qidx_z+kidx_z)%nq[2]
                if kpidx_x<0: kpidx_x += nq[0]
                if kpidx_y<0: kpidx_y += nq[1]
                if kpidx_z<0: kpidx_z += nq[2]
                kpidx = (kpidx_x*nq[1]+kpidx_y)*nq[2]+kpidx_z
                epckqNL_L(
                    &drveck_p[l*3*nspin*norbnb],bandveck[kpidx],
                    &phvecval[qidx*nm_num*nmodes],vdrvexpikR,&c1,
                    &epc_t[l*nqpath+iq,nm_min*nbands2],nmodes,natom,nm_num,
                    nbands,norbital,nbands2,norbnb,nspin,norb_u,norb_u_num
                )
            end = mpi.MPI_Wtime()
            if myid == 0:
                printf("time in mode loop %d, knum loop %d:%12.4fs.\n",h,iq,end-start)
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("epc_NL(Left)_q part2 time in mode[%4d,%4d]:%12.4fs.\n",nm_min,nm_min+nm_num,endtime-starttime)

    free(drSHexpikR)
    free(vdrvexpikR)
    free(expikR)
    free(nodelist)
    free(dr)
    free(dr_num)
    free(drveck_p)
    mpi.MPI_Win_free(&win1)
    mpi.MPI_Barrier(c_comm)


@cython.boundscheck(False)
@cython.wraparound(False)
def MPIepcNL_R_q(
    MPI.Comm comm, int nmodes, int natom_loop, int natom_buffer,
    int norbital, int nbands, int ncell, int knum, nspin, 
    int[::1] natom_split, int[::1] nq, int[:,::1] R_list, int nqpath, 
    double[:,::1] qpath, double complex[:,:,:,::1] drSH,
    double complex[:,:,::1] bandveck, double complex[:,::1] phvecval_p,
    int[::1] kproc, int[::1] kproc_num, int[::1] norb, int[::1] norb_num,
    int[::1] norb_u, int[::1] norb_u_num, double complex[:,::1] epc_t
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm shm_comm, remote_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Aint l_drveck
    cdef mpi.MPI_Win win, win1
    cdef double complex* phvecval
    cdef double complex* bandveckp
    cdef double complex* drSHexpikR
    cdef double complex* drveck
    cdef double complex* drveck_p
    cdef double complex* vdrv
    cdef double complex* vdrvexpikR
    cdef double complex* expikR
    cdef int natom = nmodes/3
    cdef int norbnb = norbital*nbands
    cdef int nbands2 = nbands*nbands
    cdef int s_int = sizeof(int)
    cdef int s_dcplx = sizeof(double complex)
    cdef int h, i, j, k, l, xyz, iq, shm_proc_s, shm_proc_e, \
             ierr, myid, nprocs, shm_id, shm_nprocs, nnode
    cdef int kidx_x, kidx_y, kidx_z, kidx_xy, kidx, \
             qidx_x, qidx_y, qidx_z, qidx_xy, qidx, \
             kpidx_x, kpidx_y, kpidx_z, kpidx_xy, kpidx, \
             knum_p, knum_s, norb_p, norb_s, k1, \
             nm_num, nm_min, nm_max, spin, s_r, s_c
    cdef int * nodelist
    cdef int* dr
    cdef int* dr_num
    cdef double kpx, kpy, kpz, RKpx, RKpy, RKpz, RKp, \
                starttime, endtime
    cdef double complex pi2j = M_PI*2j
    cdef double complex c1 = 1.0
    cdef double complex c0 = 0.0

    starttime = mpi.MPI_Wtime()
    # get shm_comm
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&remote_comm)

    dr = <int*>calloc((nnode+1),s_int)
    dr_num = <int*>calloc(nnode,s_int)
    for i in range(nnode):
        shm_proc_s = i*shm_nprocs
        shm_proc_e = (i+1)*shm_nprocs
        for j in range(shm_proc_s,shm_proc_e):
            dr_num[i] += norb_num[j]*nbands
        for j in range(i+1,nnode+1):
            dr[j] += dr_num[i]
    norb_p = norb_num[myid]
    norb_s = norb[myid]
    bandveckp = <double complex*>malloc(s_dcplx*nspin*norbnb)
    expikR = <double complex*>malloc(s_dcplx*ncell)
    drSHexpikR = <double complex*>malloc(s_dcplx*norbital*norb_p)
    vdrv = <double complex*>malloc(s_dcplx*nbands2)
    vdrvexpikR = <double complex*>malloc(s_dcplx*nmodes*nbands2)
    drveck_p = <double complex*>malloc(s_dcplx*norb_p*nbands)
    if (shm_id == 0):
        l_phvecval = s_dcplx*knum*natom_buffer*3*nmodes
        l_drveck = s_dcplx*knum*3*nspin*norbnb
    else:
        l_phvecval = 0
        l_drveck = 0
    mpi.MPI_Win_allocate_shared(
        l_drveck,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&drveck,&win
    )
    mpi.MPI_Win_allocate_shared(
        l_phvecval,s_dcplx,mpi.MPI_INFO_NULL,shm_comm,&phvecval,&win1
    )
    if (shm_id != 0):
        mpi.MPI_Win_shared_query(win,0,&l_drveck,&s_dcplx,&drveck)
        mpi.MPI_Win_shared_query(win1,0,&l_phvecval,&s_dcplx,&phvecval)
    mpi.MPI_Barrier(shm_comm)

    knum_p = kproc_num[myid]
    knum_s = kproc[myid]

    if norb_p > 0:
        for h in range(knum):
            kpidx_z = h%nq[2]; kpidx_xy = h/nq[2]
            kpidx_y = kpidx_xy%nq[1]; kpidx_x = kpidx_xy/nq[1]
            kpidx = (kpidx_x*nq[1]+kpidx_y)*nq[2]+kpidx_z
            kpx = (<double>(kpidx_x))/(<double>(nq[0]))
            kpy = (<double>(kpidx_y))/(<double>(nq[1]))
            kpz = (<double>(kpidx_z))/(<double>(nq[2]))
            for i in range(ncell):
                RKpx = R_list[i,0]*kpx
                RKpy = R_list[i,1]*kpy
                RKpz = R_list[i,2]*kpz
                RKp = RKpx + RKpy + RKpz
                expikR[i] = cexp(-1.0*pi2j*RKp)
            for i in range(nspin*norbital):
                for j in range(nbands):
                    bandveckp[i*nbands+j] = conj(bandveck[kpidx,i,j])
            for s_r in range(nspin):
                for xyz in range(3):
                    l = (h*3+xyz)*nspin+s_r
                    memset(drveck_p,0,s_dcplx*norb_p*nbands)
                    for s_c in range(nspin):
                        spin = s_r*nspin+s_c
                        # drSH[nR,norb_p*norb]*expikR[nR]
                        cblas_zgemv(
                            CblasRowMajor,CblasTrans,ncell,norb_p*norbital,
                            &c1,&drSH[spin,xyz,0,0],norb_p*norbital,
                            expikR,1,&c0,drSHexpikR,1
                        )
                        cblas_zgemm3m(
                            CblasRowMajor,CblasNoTrans,CblasNoTrans,norb_p,
                            nbands,norbital,&c1,drSHexpikR,norbital,
                            &bandveckp[s_c*norbnb],nbands,&c1,drveck_p,nbands
                        )
                    # drveck_p -> drveck (gather to node comm)
                    for j in range(norb_p*nbands):
                        drveck[l*norbnb+norb_s*nbands+j] = drveck_p[j]
                    if nnode > 1:
                        mpi.MPI_Barrier(shm_comm)
                        if (shm_id == 0):
                            mpi.MPI_Allgatherv(
                                mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,
                                &drveck[l*norbnb],&dr_num[0],&dr[0],
                                mpi.MPI_DOUBLE_COMPLEX,remote_comm
                            )
    mpi.MPI_Barrier(shm_comm)
    free(drveck_p)
    
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("epc_NL(Right)_q part1 time:%12.4fs.\n",endtime-starttime)
    for h in range(natom_loop):
        nm_min = natom_split[h]*3
        nm_max = natom_split[h+1]*3
        nm_num = nm_max-nm_min
        phvec_gather(
            c_comm,shm_comm,remote_comm,myid,shm_id,
            knum,nmodes,nm_num,nm_min,
            kproc,kproc_num,phvecval_p,phvecval
        )
        starttime = mpi.MPI_Wtime()
        for iq in range(nqpath):
            start = mpi.MPI_Wtime()
            if nqpath < knum:
                qidx_x = <int>round(nq[0]*qpath[iq,0])
                qidx_y = <int>round(nq[1]*qpath[iq,1])
                qidx_z = <int>round(nq[2]*qpath[iq,2])
            else:
                qidx_z = iq%nq[2]; qidx_xy = iq/nq[2]
                qidx_y = qidx_xy%nq[1]; qidx_x = qidx_xy/nq[1]
            qidx = (qidx_x*nq[1]+qidx_y)*nq[2]+qidx_z
            for l in range(knum_p):
                k1 = l+knum_s
                kidx_z = k1%nq[2]; kidx_xy = k1/nq[2]
                kidx_y = kidx_xy%nq[1]; kidx_x = kidx_xy/nq[1]
                kpidx_x = (qidx_x+kidx_x)%nq[0]
                kpidx_y = (qidx_y+kidx_y)%nq[1]
                kpidx_z = (qidx_z+kidx_z)%nq[2]
                if kpidx_x<0: kpidx_x += nq[0]
                if kpidx_y<0: kpidx_y += nq[1]
                if kpidx_z<0: kpidx_z += nq[2]
                kpidx = (kpidx_x*nq[1]+kpidx_y)*nq[2]+kpidx_z
                epckqNL_R(
                    &drveck[kpidx*3*nspin*norbnb],bandveck[k1],&phvecval[qidx*nm_num*nmodes],vdrv,
                    vdrvexpikR,&c1,&epc_t[l*nqpath+iq,nm_min*nbands2],nmodes,natom,
                    nm_num,nbands,norbital,nbands2,norbnb,nspin,norb_u,norb_u_num
                )
            end = mpi.MPI_Wtime()
            if myid == 0:
                printf("time in mode loop %d, knum loop %d:%12.4fs.\n",h,iq,end-start)
        endtime = mpi.MPI_Wtime()
        if myid == 0:
            printf("epc_NL(Right)_q part2 time in mode[%4d,%4d]:%12.4fs.\n",nm_min,nm_min+nm_num,endtime-starttime)

    free(bandveckp)
    free(drSHexpikR)
    free(vdrv)
    free(vdrvexpikR)
    free(expikR)
    free(nodelist)
    free(dr)
    free(dr_num)
    mpi.MPI_Win_free(&win)
    mpi.MPI_Win_free(&win1)
    mpi.MPI_Barrier(c_comm)

