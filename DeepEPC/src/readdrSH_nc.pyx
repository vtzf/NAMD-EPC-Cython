#cython: language_level=3
#cython: cdivision=True

cimport cython
from epc cimport *
from hdf5 cimport *
from mkl_spblas cimport *
from mkl_blacs cimport *

cdef long norb
cdef int* atom_idx
cdef int* atom_idx_sum
cdef int key_num_p
cdef double Hartree2eV = 27.211396641308
#cdef double Hartree2eV = 27.2113845
cdef double Bohr2Ang = 0.529177249


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readh5_h(
    mpi.MPI_Comm shm_comm, int shm_nprocs, int shm_id, char* h5_name,
    int nfile, int norb_m, int[:,::1] key_num, int[:,::1] pub_key,
    long[:,::1] key_info, double[:,:,::1] data, double factor
):
    cdef int h, i, j, k, l, m, key_min, key_max, offset, TNO1, TNO2
    cdef hid_t f, memtype
    cdef herr_t status
    cdef char h5name[500]
    cdef char key_t[100]
    cdef double complex* data_buf = <double complex*>malloc(norb_m*norb_m*4*sizeof(double complex))
    cdef double* databuf = <double*>data_buf

    memtype = H5Tcreate(H5T_COMPOUND,16)
    H5Tinsert(memtype,"r",0,H5T_NATIVE_DOUBLE)
    H5Tinsert(memtype,"i",8,H5T_NATIVE_DOUBLE)
    if nfile>1:
        for i in range(nfile):
            mpi.MPI_Barrier(shm_comm)
            key_min = key_num[i,2]+(key_num[i,0]*shm_id)/shm_nprocs
            key_max = key_num[i,2]+(key_num[i,0]*(shm_id+1))/shm_nprocs
            sprintf(h5name,"%s_%d.h5",h5_name,i)
            if (key_min<key_max):
                f = H5Fopen(h5name,H5F_ACC_RDONLY,H5P_DEFAULT)
                for j in range(key_min,key_max):
                    sprintf(key_t,"[0, 0, 0, %d, %d]",pub_key[j,0]+1,pub_key[j,1]+1)
                    data_id = H5Dopen(f,key_t,H5P_DEFAULT)
                    status = H5Dread(
                        data_id,memtype,H5S_ALL,
                        H5S_ALL,H5P_DEFAULT,data_buf
                    )
                    TNO1 = pub_key[j,2]
                    TNO2 = pub_key[j,3]
                    offset = pub_key[j,5]
                    for k in range(TNO1):
                        for l in range(TNO2):
                            m = key_info[k*TNO2+l+offset,1]
                            data[0,0,m] = databuf[(k*TNO2*2+l)*2]*factor
                            data[1,0,m] = databuf[(k*TNO2*2+l)*2+1]*factor
                            data[0,1,m] = databuf[(k*TNO2*2+l+TNO2)*2]*factor
                            data[1,1,m] = databuf[(k*TNO2*2+l+TNO2)*2+1]*factor
                            data[0,2,m] = databuf[((k+TNO1)*TNO2*2+l)*2]*factor
                            data[1,2,m] = databuf[((k+TNO1)*TNO2*2+l)*2+1]*factor
                            data[0,3,m] = databuf[((k+TNO1)*TNO2*2+l+TNO2)*2]*factor
                            data[1,3,m] = databuf[((k+TNO1)*TNO2*2+l+TNO2)*2+1]*factor
                    status = H5Dclose(data_id)
                status = H5Fclose(f)
    else:
        mpi.MPI_Barrier(shm_comm)
        key_min = (key_num[0,0]*shm_id)/shm_nprocs
        key_max = (key_num[0,0]*(shm_id+1))/shm_nprocs
        if (key_min<key_max):
            sprintf(h5name,"%s.h5",h5_name)
            f = H5Fopen(h5name,H5F_ACC_RDONLY,H5P_DEFAULT)
            for j in range(key_min,key_max):
                sprintf(key_t,"[0, 0, 0, %d, %d]",pub_key[j,0]+1,pub_key[j,1]+1)
                data_id = H5Dopen(f,key_t,H5P_DEFAULT)
                status = H5Dread(
                    data_id,memtype,H5S_ALL,
                    H5S_ALL,H5P_DEFAULT,data_buf
                )
                TNO1 = pub_key[j,2]
                TNO2 = pub_key[j,3]
                offset = pub_key[j,5]
                for k in range(TNO1):
                    for l in range(TNO2):
                        m = key_info[k*TNO2+l+offset,1]
                        data[0,0,m] = databuf[(k*TNO2*2+l)*2]*factor
                        data[1,0,m] = databuf[(k*TNO2*2+l)*2+1]*factor
                        data[0,1,m] = databuf[(k*TNO2*2+l+TNO2)*2]*factor
                        data[1,1,m] = databuf[(k*TNO2*2+l+TNO2)*2+1]*factor
                        data[0,2,m] = databuf[((k+TNO1)*TNO2*2+l)*2]*factor
                        data[1,2,m] = databuf[((k+TNO1)*TNO2*2+l)*2+1]*factor
                        data[0,3,m] = databuf[((k+TNO1)*TNO2*2+l+TNO2)*2]*factor
                        data[1,3,m] = databuf[((k+TNO1)*TNO2*2+l+TNO2)*2+1]*factor
                status = H5Dclose(data_id)
            status = H5Fclose(f)

    status = H5Tclose(memtype)
    free(data_buf)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readh5(
    mpi.MPI_Comm shm_comm, int shm_nprocs, int shm_id, char* h5_name,
    int nfile, int norb_m, int[:,::1] key_num, int[:,::1] pub_key,
    long[:,::1] key_info, double[::1] data, double factor
):
    cdef int h, i, j, k, key_min, key_max, offset
    cdef hid_t f
    cdef herr_t status
    cdef char h5name[500]
    cdef char key_t[100]
    cdef double* data_buf = <double*>malloc(norb_m*norb_m*sizeof(double))

    if nfile>1:
        for i in range(nfile):
            mpi.MPI_Barrier(shm_comm)
            key_min = key_num[i,2]+(key_num[i,0]*shm_id)/shm_nprocs
            key_max = key_num[i,2]+(key_num[i,0]*(shm_id+1))/shm_nprocs
            if (key_min<key_max):
                sprintf(h5name,"%s_%d.h5",h5_name,i)
                f = H5Fopen(h5name,H5F_ACC_RDONLY,H5P_DEFAULT)
                for j in range(key_min,key_max):
                    sprintf(key_t,"[0, 0, 0, %d, %d]",pub_key[j,0]+1,pub_key[j,1]+1)
                    data_id = H5Dopen(f,key_t,H5P_DEFAULT)
                    status = H5Dread(
                        data_id,H5T_NATIVE_DOUBLE,H5S_ALL,
                        H5S_ALL,H5P_DEFAULT,data_buf
                    )
                    offset = pub_key[j,5]
                    for k in range(pub_key[j,4]):
                        data[key_info[k+offset,1]] = data_buf[k]*factor
                    status = H5Dclose(data_id)
                status = H5Fclose(f)
    else:
        mpi.MPI_Barrier(shm_comm)
        key_min = (key_num[0,0]*shm_id)/shm_nprocs
        key_max = (key_num[0,0]*(shm_id+1))/shm_nprocs
        if (key_min<key_max):
            sprintf(h5name,"%s.h5",h5_name)
            f = H5Fopen(h5name,H5F_ACC_RDONLY,H5P_DEFAULT)
            for j in range(key_min,key_max):
                sprintf(key_t,"[0, 0, 0, %d, %d]",pub_key[j,0]+1,pub_key[j,1]+1)
                data_id = H5Dopen(f,key_t,H5P_DEFAULT)
                status = H5Dread(
                    data_id,H5T_NATIVE_DOUBLE,H5S_ALL,
                    H5S_ALL,H5P_DEFAULT,data_buf
                )
                offset = pub_key[j,5]
                for k in range(pub_key[j,4]):
                    data[key_info[k+offset,1]] = data_buf[k]*factor
                status = H5Dclose(data_id)
            status = H5Fclose(f)

    free(data_buf)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readscfout(
    char* name, int norb_m,  
    int[:,::1] key_num, int[:,::1] pub_key, long[:,::1] key_info,
    double[:,:,::1] data_h, double[::1] data_o, double[:,::1] data_dr,
    double factor_h, double factor_o, double factor_dr
):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j ,k, spin, xyz, ct_AN, h_AN, Gh_AN, \
             atomnum, factor, offset, offset_i, TNO1, TNO2
    cdef int* atv_ijk
    cdef int* FNAN
    cdef int** natn
    cdef int** ncn

    cdef double* data_buf = <double*>malloc(norb_m*norb_m*sizeof(double))

    fp = fopen(name,'rb')
    fseek(fp,0,SEEK_SET)
    fread(i_vec,sizeof(int),6,fp)
    atomnum = i_vec[0]
    TCpyCell = i_vec[5]
    fseek(fp,4+(TCpyCell+1)*4*(8+4),SEEK_CUR)

    Total_NumOrbs = <int*>malloc(sizeof(int)*(atomnum+1))
    Total_NumOrbs[0] = 1
    fread(&(Total_NumOrbs[1]),sizeof(int),atomnum,fp)
    FNAN = <int*>malloc(sizeof(int)*(atomnum+1))
    FNAN[0] = 0
    fread(&(FNAN[1]),sizeof(int),atomnum,fp)

    natn = <int**>malloc(sizeof(int*)*(atomnum+1))
    for ct_AN in range(1,atomnum+1):
        natn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(natn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    ncn = <int**>malloc(sizeof(int*)*(atomnum+1))
    for ct_AN in range(1,atomnum+1):
        ncn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(ncn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    fseek(fp,(3+3+atomnum)*4*8,SEEK_CUR)

    for spin in range(4):
        offset = 0
        for ct_AN in range(1,atomnum+1):
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN]+1):
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                if (ncn[ct_AN][h_AN]==0):
                    fread(data_buf,sizeof(double),TNO1*TNO2,fp)
                    if spin == 0:
                        for i in range(TNO1*TNO2):
                            data_h[0,0,key_info[i+offset,1]] \
                            = data_buf[i]*factor_h
                    elif spin == 1:
                        for i in range(TNO1*TNO2):
                            data_h[0,3,key_info[i+offset,1]] \
                            = data_buf[i]*factor_h
                    elif spin == 2:
                        for i in range(TNO1*TNO2):
                            data_h[0,1,key_info[i+offset,1]] \
                            = data_buf[i]*factor_h
                    else:
                        for i in range(TNO1*TNO2):
                            data_h[1,1,key_info[i+offset,1]] \
                            = data_buf[i]*factor_h
                    offset += TNO1*TNO2
                else:
                    fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    for spin in range(3):
        offset = 0
        for ct_AN in range(1,atomnum+1):
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN]+1):
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                if (ncn[ct_AN][h_AN]==0):
                    fread(data_buf,sizeof(double),TNO1*TNO2,fp)
                    if spin == 0:
                        for i in range(TNO1*TNO2):
                            data_h[1,0,key_info[i+offset,1]] \
                            = data_buf[i]*factor_h
                    elif spin == 1:
                        for i in range(TNO1*TNO2):
                            data_h[1,3,key_info[i+offset,1]] \
                            = data_buf[i]*factor_h
                    else:
                        for i in range(TNO1*TNO2):
                            data_h[1,1,key_info[i+offset,1]] \
                            += data_buf[i]*factor_h
                    offset += TNO1*TNO2
                else:
                    fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    offset = 0
    for ct_AN in range(1,atomnum+1):
        TNO1 = Total_NumOrbs[ct_AN]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            TNO2 = Total_NumOrbs[Gh_AN]
            if (ncn[ct_AN][h_AN]==0):
                # get H[0][i(dn),j(up)] offset_i
                offset_i = -1
                for i in range(key_num[1,2]):
                    if (pub_key[i,0]==Gh_AN-1 and pub_key[i,1]==ct_AN-1):
                        offset_i = pub_key[i,5]
                        break
                # H[0][i(dn),j(up)] = H[0][j(up),i(dn)].conj().T
                for i in range(TNO1):
                    for j in range(TNO2):
                        data_h[0,2,key_info[j*TNO1+i+offset_i,1]] \
                        = data_h[0,1,key_info[i*TNO2+j+offset,1]]
                        data_h[1,2,key_info[j*TNO1+i+offset_i,1]] \
                        = -1.0*data_h[1,1,key_info[i*TNO2+j+offset,1]]

                offset += TNO1*TNO2

    offset = 0
    for ct_AN in range(1,atomnum+1):
        TNO1 = Total_NumOrbs[ct_AN]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            TNO2 = Total_NumOrbs[Gh_AN]
            if (ncn[ct_AN][h_AN]==0):
                fread(data_buf,sizeof(double),TNO1*TNO2,fp)
                for i in range(TNO1*TNO2):
                    data_o[key_info[i+offset,1]] \
                    = data_buf[i]*factor_o
                offset += TNO1*TNO2
            else:
                fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    # pass olpr
    for xyz in range(3):
        for ct_AN in range(1,atomnum+1):
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN]+1):
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    for xyz in range(3):
        offset = 0
        for ct_AN in range(1,atomnum+1):
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN]+1):
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                if (ncn[ct_AN][h_AN]==0):
                    fread(data_buf,sizeof(double),TNO1*TNO2,fp)
                    for i in range(TNO1*TNO2):
                        data_dr[xyz,key_info[i+offset,1]] \
                        = data_buf[i]*factor_dr
                    offset += TNO1*TNO2
                else:
                    fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    fclose(fp)
    for ct_AN in range(1,atomnum+1):
        free(natn[ct_AN])
        free(ncn[ct_AN])
    free(natn)
    free(ncn)
    free(FNAN)
    free(Total_NumOrbs)
    free(data_buf)


def GetSparseData(
    MPI.Comm shm_comm_py,
    char* inDir, char* H5HamName, char* H5OlpName, char* H5DrName,
    int nfileham, int nfileolp, int nfiledr, int atomnum, int norb_m,
    int[:,::1] key_num_h, int[:,::1] key_num_o, int[:,::1] key_num_dr,
    int[:,::1] pub_key_h, int[:,::1] pub_key_o, int[:,::1] pub_key_dr,
    long[:,::1] keyinfo_h, long[:,::1] keyinfo_o, long[:,::1] keyinfo_dr,
    double[:,:,::1] data_h, double[::1] data_o, double[:,::1] data_dr, bint IsH5
):
    cdef int shm_nprocs, shm_id, ierr
    cdef char data_name[500]
    cdef double f_dr = 1.0/Bohr2Ang

    cdef mpi.MPI_Comm shm_comm = shm_comm_py.ob_mpi
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)

    if IsH5:
        readh5_h(
            shm_comm,shm_nprocs,shm_id,H5HamName,nfileham,
            norb_m,key_num_h,pub_key_h,keyinfo_h,data_h,1.0
        )
        readh5(
            shm_comm,shm_nprocs,shm_id,H5OlpName,nfileolp,
            norb_m,key_num_o,pub_key_o,keyinfo_o,data_o,1.0
        )
        sprintf(data_name,"%sx",H5DrName)
        readh5(
            shm_comm,shm_nprocs,shm_id,data_name,nfiledr,
            norb_m,key_num_dr,pub_key_dr,keyinfo_dr,data_dr[0],1.0
        )
        sprintf(data_name,"%sy",H5DrName)
        readh5(
            shm_comm,shm_nprocs,shm_id,data_name,nfiledr,
            norb_m,key_num_dr,pub_key_dr,keyinfo_dr,data_dr[1],1.0
        )
        sprintf(data_name,"%sz",H5DrName)
        readh5(
            shm_comm,shm_nprocs,shm_id,data_name,nfiledr,
            norb_m,key_num_dr,pub_key_dr,keyinfo_dr,data_dr[2],1.0
        )
    else:
        sprintf(data_name,"%s/openmx.scfout",inDir)
        if (shm_id==0):
            readscfout(
                data_name,norb_m,key_num_h,pub_key_h,keyinfo_h,
                data_h,data_o,data_dr,Hartree2eV,1.0,f_dr
            )
    mpi.MPI_Barrier(shm_comm)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sparse2dense_coo(
    double[::1] olp, int[:,::1] keyinfo, int N, int Nsparse,
    int Nproc_min, int Nproc_max, int Nproc_len, double factor,
    double* olp_f
):
    cdef int i
    cdef int irow, icol

    memset(olp_f,0,sizeof(double)*N*Nproc_len)
    for i in range(Nsparse):
        icol = keyinfo[1,i]
        if icol<Nproc_min or icol>=Nproc_max:
            continue
        else:
            irow = keyinfo[0,i]
            olp_f[irow*Nproc_len+icol-Nproc_min] = olp[i]*factor


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mat_tran(
    mpi.MPI_Comm comm, int nprocs, int N, int Nproc_len, int* Nproc,
    int* Nproc_num, double* matbuf, double* recvbuf
):
    cdef int i, j, k, N1, N2, offset
    cdef int * count = <int*>malloc(nprocs*sizeof(int))
    cdef int * displ = <int*>malloc(nprocs*sizeof(int))

    for i in range(nprocs):
        count[i] = Nproc_len*Nproc_num[i]
        displ[i] = Nproc_len*Nproc[i]
    mpi.MPI_Alltoallv(
        matbuf,count,displ,mpi.MPI_DOUBLE,
        recvbuf,count,displ,mpi.MPI_DOUBLE,comm
    )
    for i in range(nprocs):
        N1 = Nproc[i]
        N2 = Nproc_num[i]
        offset = displ[i]
        for j in range(Nproc_len):
            for k in range(N2):
                matbuf[(k+N1)*Nproc_len+j] = recvbuf[offset+j*N2+k]
    free(count)
    free(displ)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mat_reshape(
    mpi.MPI_Comm comm, double* olp, int nprocs, int myid, 
    int Limag, int N, int Nproc_len, int Nsplit,
    int* Nproc, int* Nproc_num, double complex[:,::1] olp_r
):
    cdef int i, j, k, l, m, n, offset, N1, N2
    cdef int s_int = sizeof(int)
    cdef int s_d = sizeof(double)
    cdef int Np = N/Nsplit

    cdef int * Npproc = <int*>calloc(s_int,nprocs+1)
    cdef int * Npproc_num = <int*>calloc(s_int,nprocs)
    cdef int * scount = <int*>malloc(nprocs*sizeof(int))
    cdef int * sdispl = <int*>malloc(nprocs*sizeof(int))
    cdef int * rcount = <int*>malloc(nprocs*sizeof(int))
    cdef int * rdispl = <int*>malloc(nprocs*sizeof(int))
    cdef double * olp_rbuf

    for i in range(nprocs):
        Np_min = (Np*i)/nprocs
        Np_max = (Np*(i+1))/nprocs
        Npproc_num[i] = Np_max-Np_min
        Npproc[i+1] = Np_max
    Np_num = Npproc_num[myid]

    for i in range(nprocs):
        scount[i] = Nproc_len*Npproc_num[i]
        rcount[i] = Np_num*Nproc_num[i]
        sdispl[i] = Nproc_len*Npproc[i]
        rdispl[i] = Np_num*Nproc[i]

    olp_rbuf = <double*>malloc(s_d*Np_num*N)
    mpi.MPI_Alltoallv(
        olp,scount,sdispl,mpi.MPI_DOUBLE,
        olp_rbuf,rcount,rdispl,mpi.MPI_DOUBLE,comm
    )
    offset = 0
    if Limag:
        for j in range(nprocs):
            N1 = Nproc[j]
            N2 = Nproc_num[j]
            offset = rdispl[j]
            for k in range(Np_num):
                for l in range(N2):
                    m = (l+N1)/Np
                    n = (l+N1)%Np
                    olp_r[m,k*Np+n] += 1j*olp_rbuf[offset+k*N2+l]
    else:
        for j in range(nprocs):
            N1 = Nproc[j]
            N2 = Nproc_num[j]
            offset = rdispl[j]
            for k in range(Np_num):
                for l in range(N2):
                    m = (l+N1)/Np
                    n = (l+N1)%Np
                    olp_r[m,k*Np+n] = olp_rbuf[offset+k*N2+l]

    free(Npproc)
    free(Npproc_num)
    free(scount)
    free(sdispl)
    free(rcount)
    free(rdispl)
    free(olp_rbuf)


@cython.boundscheck(False)
@cython.wraparound(False)
def olp_inv_old(
    MPI.Comm comm, int nprocs, int myid, double[::1] olp,
    int[:,::1] olp_keyinfo, int Nsparse_o, double[:,:,::1] ham, 
    int[:,::1] ham_csr_ridx, int[::1] ham_csr_cidx, int Nsparse_h,
    double[:,::1] dr, int[:,::1] dr_csr_ridx, int[::1] dr_csr_cidx,
    int Nsparse_dr, int drp_min, int N, int Nsplit,
    MKL_INT Mb, MKL_INT Nb, double complex[:,:,:,::1] drSH #drSH[4,3,NR,(Np)_p*Np]
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int i, j, k, Nproc_min, Nproc_max, Nproc_len
    cdef MKL_INT i1 = 1
    cdef MKL_INT i0 = 0
    cdef MKL_INT nprow, npcol, myrow, mycol, mb, nb, \
                 nprow1, npcol1, myrow1, mycol1, mb1, \
                 ictxt, ictxt1, N_mkl, Nsparse_mkl, \
                 Nproc_mkl, Np_mkl, lldSp, lldS, info
    cdef MKL_INT descSp[9]
    cdef MKL_INT descS[9]
    cdef int s_int = sizeof(int)
    cdef int s_d = sizeof(double)
    cdef int * Nproc
    cdef int * Nproc_num
    cdef double* olpinvham
    cdef double* matbuf
    cdef double* olpbuf_pb
    cdef double* hambuf_pb
    cdef double f0 = 0.0
    cdef double f1 = 1.0
    cdef double lv, starttime, endtime, tt
    cdef double t_s2d=0.0, t_redist=0.0, t_solve=0.0, t_opt=0.0
    cdef double t_hmm=0.0, t_tran=0.0, t_drmm=0.0, t_reshape=0.0
    cdef matrix_descr descrH
    cdef matrix_descr descrdr
    cdef sparse_matrix_t csrH
    cdef sparse_matrix_t csrdr

    starttime = mpi.MPI_Wtime()
    # split N
    Nproc_num = <int*>calloc(s_int,nprocs)
    Nproc = <int*>calloc(s_int,(nprocs+1))
    if (N%nprocs==0):
        j = N/nprocs
        for i in range(nprocs):
            Nproc_num[i] = j
    else:
        j = N/nprocs+1
        k = N/j
        for i in range(k):
            Nproc_num[i] = j
        Nproc_num[k] = N-j*k
    for i in range(nprocs):
        for j in range(i+1,nprocs+1):
            Nproc[j] += Nproc_num[i]
    Nproc_min = Nproc[myid]
    Nproc_max = Nproc[myid+1]
    Nproc_len = Nproc_num[myid]
    N_mkl = N
    Nsparse_mkl = Nsparse_dr
    Nproc_mkl = Nproc_num[0]
    Np_mkl = N/Nsplit

    matbuf = <double*>malloc(s_d*N*Nproc_len)
    # get scalapack info
    blacs_get(&i0,&i0,&ictxt)
    blacs_get(&i0,&i0,&ictxt1)
    nprow = 1
    npcol = nprocs
    for i in range(1,<int>(sqrt(nprocs))+1):
        if (nprocs%i==0):
            nprow = i
            npcol = nprocs/i
    nprow1 = nprocs
    npcol1 = 1

    blacs_gridinit(&ictxt,"Row",&nprow,&npcol)
    blacs_gridinit(&ictxt1,"Row",&nprow1,&npcol1)
    blacs_gridinfo(&ictxt,&nprow,&npcol,&myrow,&mycol)
    blacs_gridinfo(&ictxt1,&nprow1,&npcol1,&myrow1,&mycol1)
    mb = numroc(&N_mkl,&Mb,&myrow,&i0,&nprow)
    nb = numroc(&N_mkl,&Nb,&mycol,&i0,&npcol)

    lldSp = Nproc_len
    lldS = <MKL_INT>fmax(mb,1)
    descinit(descS,&N_mkl,&N_mkl,&Mb,&Nb,&i0,&i0,&ictxt,&lldS,&info)
    #descinit(descSp,&N_mkl,&N_mkl,&Nproc_mkl,&N_mkl,&i0,&i0,&ictxt1,&lldSp,&info)
    # In some spacial case,
    # lldSp = numroc(&N_mkl,&Nproc_mkl,&myrow1,&i0,&nprow1) = 0,
    # which causes illegal checking results in descinit. Actually,
    # lldSp = 0 has no influence on data redistribution of pdgemr2d.
    # Here we manually assign descSp to avoid using descinit.
    descSp[0] = i1;    descSp[1] = ictxt1;    descSp[2] = N_mkl
    descSp[3] = N_mkl; descSp[4] = Nproc_mkl; descSp[5] = N_mkl
    descSp[6] = i0;    descSp[7] = i0;        descSp[8] = lldSp

    olpbuf_pb = <double*>malloc(s_d*mb*nb)
    olpinv_pb = <double*>malloc(s_d*mb*nb)

    # init olp_pb
    tt = mpi.MPI_Wtime()
    sparse2dense_coo(
        olp,olp_keyinfo,N,Nsparse_o,Nproc_min,
        Nproc_max,Nproc_len,1.0,matbuf
    )
    t_s2d += mpi.MPI_Wtime()-tt
    tt = mpi.MPI_Wtime()
    pdgemr2d(
        &N_mkl,&N_mkl,matbuf,&i1,&i1,descSp,
        olpbuf_pb,&i1,&i1,descS,&ictxt
    )
    # init unit matrix I_pb in olpinv_pb
    memset(matbuf,0,sizeof(double)*N*Nproc_len)
    for i in range(Nproc_len):
        matbuf[(i+Nproc_min)*Nproc_len+i] = 1.0
    pdgemr2d(
        &N_mkl,&N_mkl,matbuf,&i1,&i1,descSp,
        olpinv_pb,&i1,&i1,descS,&ictxt
    )
    t_redist += mpi.MPI_Wtime()-tt
    # calculate S^-1 and overwrite olpinv_pb
    tt = mpi.MPI_Wtime()
    pdposv(
        'U',&N_mkl,&N_mkl,olpbuf_pb,&i1,&i1,
        descS,olpinv_pb,&i1,&i1,descS,&info
    )
    t_solve += mpi.MPI_Wtime()-tt
    free(olpbuf_pb)
    # transfer olpinv_pb[mb,nb] to olpinv[N,Nproc_len]
    olpinv = <double*>malloc(s_d*N*Nproc_len)
    tt = mpi.MPI_Wtime()
    pdgemr2d(
        &N_mkl,&N_mkl,olpinv_pb,&i1,&i1,descS,
        olpinv,&i1,&i1,descSp,&ictxt1
    )
    t_redist += mpi.MPI_Wtime()-tt
    free(olpinv_pb)
    olpinvham = <double*>malloc(s_d*N*Nproc_len)

    descrH.type = SPARSE_MATRIX_TYPE_GENERAL
    descrdr.type = SPARSE_MATRIX_TYPE_GENERAL
    for m in range(2):
        for n in range(4):
            tt = mpi.MPI_Wtime()
            mkl_sparse_d_create_csr(
                &csrH,SPARSE_INDEX_BASE_ZERO,N_mkl,N_mkl,
                &ham_csr_ridx[0,0],&ham_csr_ridx[1,0],
                &ham_csr_cidx[0],&ham[m,n,0]
            )
            mkl_sparse_optimize(csrH)
            t_opt += mpi.MPI_Wtime()-tt
            # dot(H[m,n],S^-1) [N,N]*[N,Nproc_len]
            tt = mpi.MPI_Wtime()
            mkl_sparse_d_mm(
                SPARSE_OPERATION_NON_TRANSPOSE,1.0,csrH,
                descrH,SPARSE_LAYOUT_ROW_MAJOR,olpinv,
                lldSp,lldSp,0.0,olpinvham,lldSp
            )
            t_hmm += mpi.MPI_Wtime()-tt
#            mkl_sparse_destroy(csrH)
            # H*S^-1 has C order, needs to transpose because
            # in C order, actually matrix is (H*S^-1)^T = S^-1*H
            tt = mpi.MPI_Wtime()
            mat_tran(
                c_comm,nprocs,N,Nproc_len,Nproc,
                Nproc_num,olpinvham,matbuf
            )
            t_tran += mpi.MPI_Wtime()-tt
            for i in range(3):
                tt = mpi.MPI_Wtime()
                mkl_sparse_d_create_csr(
                    &csrdr,SPARSE_INDEX_BASE_ZERO,Np_mkl,N_mkl,
                    &dr_csr_ridx[0,0],&dr_csr_ridx[1,0],
                    &dr_csr_cidx[drp_min],&dr[i,drp_min]
                )
                mkl_sparse_optimize(csrdr)
                t_opt += mpi.MPI_Wtime()-tt
                # dot(dr,S^-1*H) [Np,N]*[N,Nproc_len]
                tt = mpi.MPI_Wtime()
                mkl_sparse_d_mm(
                    SPARSE_OPERATION_NON_TRANSPOSE,1.0,csrdr,
                    descrdr,SPARSE_LAYOUT_ROW_MAJOR,olpinvham,
                    lldSp,lldSp,0.0,matbuf,lldSp
                )
                t_drmm += mpi.MPI_Wtime()-tt
                # reshape dr*S^-1*H [Np,N]->[nR,(Np)_p,Np]
                tt = mpi.MPI_Wtime()
                mat_reshape(
                    c_comm,matbuf,nprocs,myid,m,N,Nproc_len,
                    Nsplit,Nproc,Nproc_num,drSH[n,i]
                )
                t_reshape += mpi.MPI_Wtime()-tt
#                mkl_sparse_destroy(csrdr)

    free(Nproc)
    free(Nproc_num)
    free(olpinv)
    free(olpinvham)
    free(matbuf)

    blacs_gridexit(&ictxt)
    blacs_gridexit(&ictxt1)

    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("olpinv time: %.5fs.\n",endtime-starttime)
        printf("  [breakdown] s2d=%.1f redist=%.1f SOLVE(pdposv)=%.1f | opt=%.1f Hmm=%.1f tran=%.1f drmm=%.1f reshape=%.1f\n",
               t_s2d,t_redist,t_solve,t_opt,t_hmm,t_tran,t_drmm,t_reshape)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sparse2dense_transpose(
    double[:,::1] dr, int xyz, int[:,::1] ridx, int[::1] cidx,
    int drp_min, int N, int Np_min, int Np_max, int Np_num,
    double factor, double* drT
):
    cdef int h, i, j, icol

    memset(drT,0,sizeof(double)*N*Np_num)
    for i in range(Np_min,Np_max):
        j = i-Np_min
        for h in range(ridx[0,i],ridx[1,i]):
            icol = cidx[drp_min+h]
            drT[j*N+icol] = dr[xyz,drp_min+h]*factor


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void mat_reshape_new(
    double* C, int Limag, int N, int Np, int Np_num,
    double complex[:,::1] olp_r
):
    cdef int i, j, k, l

    if Limag:
        for i in range(Np_num):
            for j in range(N):
                k = j/Np
                l = j%Np
                olp_r[k,i*Np+l] += 1j*C[i*N+j]
    else:
        for i in range(Np_num):
            for j in range(N):
                k = j/Np
                l = j%Np
                olp_r[k,i*Np+l] = C[i*N+j]


@cython.boundscheck(False)
@cython.wraparound(False)
def olp_inv(
    MPI.Comm comm, int nprocs, int myid, double[::1] olp,
    int[:,::1] olp_keyinfo, int Nsparse_o, double[:,:,::1] ham,
    int[:,::1] ham_csr_ridx, int[::1] ham_csr_cidx, int Nsparse_h,
    double[:,::1] dr, int[:,::1] dr_csr_ridx, int[::1] dr_csr_cidx,
    int Nsparse_dr, int drp_min, int N, int Nsplit, MKL_INT Mb, MKL_INT Nb,
    int[::1] Npproc, int[::1] Npproc_num, double complex[:,:,:,::1] drSH
):
    # drSH[n,i] = dr[i]*S^-1*H[n], n=0..3 spin blocks, i=0..2 xyz.
    # Instead of forming the full S^-1 (N columns), Cholesky-factor S once
    # then solve only the Np(=N/ncell) right-hand-sides of dr's home block.
    # Mb blocks the N dimension of both S and dr (S needs square Mb*Mb for
    # pdpotrf); Nb blocks dr's Np dimension. Npproc/Npproc_num give the ceil-
    # block split of Np across procs (same rule as descSp splits N), passed in
    # so it stays identical to drSH's row split downstream.
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int h, i, j, k, m, n
    cdef int Nproc_min, Nproc_max, Nproc_len, Np, Np_min, Np_max, Np_num
    cdef MKL_INT i1 = 1
    cdef MKL_INT i0 = 0
    cdef MKL_INT nprow, npcol, myrow, mycol, mb, nb, nb_rhs, \
                 nprow1, npcol1, myrow1, mycol1, \
                 nprow2, npcol2, myrow2, mycol2, \
                 ictxt, ictxt1, ictxt2, N_mkl, Np_mkl, \
                 Nproc_mkl, NpB_mkl, lldSp, lldS, lldRHS, info
    cdef MKL_INT descSp[9]
    cdef MKL_INT descS[9]
    cdef MKL_INT descRHS[9]
    cdef MKL_INT descRHSp[9]
    cdef int s_int = sizeof(int)
    cdef int s_d = sizeof(double)
    cdef int * Nproc
    cdef int * Nproc_num
    cdef double* matbuf
    cdef double* Sbuf
    cdef double* Zbuf
    cdef double* drT
    cdef double* Cbuf
    cdef double starttime, endtime
    cdef matrix_descr descrH
    cdef sparse_matrix_t csrH[8]

    starttime = mpi.MPI_Wtime()
    Np = N/Nsplit
    # split N over procs (ceil-block, one block/proc; matches descSp/descS)
    Nproc_num = <int*>calloc(s_int,nprocs)
    Nproc = <int*>calloc(s_int,(nprocs+1))
    if (N%nprocs==0):
        j = N/nprocs
        for i in range(nprocs):
            Nproc_num[i] = j
    else:
        j = N/nprocs+1
        k = N/j
        for i in range(k):
            Nproc_num[i] = j
        Nproc_num[k] = N-j*k
    for i in range(nprocs):
        for j in range(i+1,nprocs+1):
            Nproc[j] += Nproc_num[i]
    Nproc_min = Nproc[myid]
    Nproc_max = Nproc[myid+1]
    Nproc_len = Nproc_num[myid]
    # Np split is passed in (ceil-block, must match drSH row split & descRHSp)
    Np_min = Npproc[myid]
    Np_max = Npproc[myid+1]
    Np_num = Npproc_num[myid]
    N_mkl = N
    Np_mkl = Np
    Nproc_mkl = Nproc_num[0]
    NpB_mkl = Npproc_num[0]

    # 3 process grids: ictxt 2D (ScaLAPACK), ictxt1 nprocs*1 (S densify, split
    # N rows), ictxt2 1*nprocs (dr^T build / Z read, split Np columns)
    blacs_get(&i0,&i0,&ictxt)
    blacs_get(&i0,&i0,&ictxt1)
    blacs_get(&i0,&i0,&ictxt2)
    nprow = 1
    npcol = nprocs
    for i in range(1,<int>(sqrt(nprocs))+1):
        if (nprocs%i==0):
            nprow = i
            npcol = nprocs/i
    nprow1 = nprocs
    npcol1 = 1
    nprow2 = 1
    npcol2 = nprocs
    blacs_gridinit(&ictxt,"Row",&nprow,&npcol)
    blacs_gridinit(&ictxt1,"Row",&nprow1,&npcol1)
    blacs_gridinit(&ictxt2,"Row",&nprow2,&npcol2)
    blacs_gridinfo(&ictxt,&nprow,&npcol,&myrow,&mycol)
    blacs_gridinfo(&ictxt1,&nprow1,&npcol1,&myrow1,&mycol1)
    blacs_gridinfo(&ictxt2,&nprow2,&npcol2,&myrow2,&mycol2)
    # local block dims: S/dr rows on Mb, S cols on Mb (square), dr cols on Nb
    mb = numroc(&N_mkl,&Mb,&myrow,&i0,&nprow)
    nb = numroc(&N_mkl,&Mb,&mycol,&i0,&npcol)
    nb_rhs = numroc(&Np_mkl,&Nb,&mycol,&i0,&npcol)

    lldSp = Nproc_len
    lldS = <MKL_INT>fmax(mb,1)
    lldRHS = <MKL_INT>fmax(mb,1)
    # descS [N,N] 2D square Mb*Mb (pdpotrf/pdpotrs), descRHS [N,Np] 2D Mb*Nb
    descinit(descS,&N_mkl,&N_mkl,&Mb,&Mb,&i0,&i0,&ictxt,&lldS,&info)
    descinit(descRHS,&N_mkl,&Np_mkl,&Mb,&Nb,&i0,&i0,&ictxt,&lldRHS,&info)
    # descSp [N,N] 1D (S densify), descRHSp [N,Np] 1D (Np-column split);
    # manually assigned to skip descinit's lld=0 check on empty procs
    descSp[0] = i1;    descSp[1] = ictxt1;    descSp[2] = N_mkl
    descSp[3] = N_mkl; descSp[4] = Nproc_mkl; descSp[5] = N_mkl
    descSp[6] = i0;    descSp[7] = i0;        descSp[8] = lldSp
    descRHSp[0] = i1;    descRHSp[1] = ictxt2;  descRHSp[2] = N_mkl
    descRHSp[3] = Np_mkl; descRHSp[4] = N_mkl;   descRHSp[5] = NpB_mkl
    descRHSp[6] = i0;    descRHSp[7] = i0;      descRHSp[8] = N_mkl

    # Phase1: densify sparse S [N,N] -> 2D block-cyclic, Cholesky once.
    matbuf = <double*>malloc(s_d*N*Nproc_len)
    Sbuf = <double*>malloc(s_d*mb*nb)
    sparse2dense_coo(
        olp,olp_keyinfo,N,Nsparse_o,Nproc_min,
        Nproc_max,Nproc_len,1.0,matbuf
    )
    pdgemr2d(
        &N_mkl,&N_mkl,matbuf,&i1,&i1,descSp,
        Sbuf,&i1,&i1,descS,&ictxt
    )
    free(matbuf)   # only needed to feed S into descS, drop before factorizing
    pdpotrf('U',&N_mkl,Sbuf,&i1,&i1,descS,&info)   # Sbuf holds L, reused by pdpotrs

    # 4 spin blocks of H = ham[m,n] (m=0/1 real/imag, n=0..3), full [N,N] sparse
    descrH.type = SPARSE_MATRIX_TYPE_GENERAL
    for m in range(2):
        for n in range(4):
            mkl_sparse_d_create_csr(
                &csrH[m*4+n],SPARSE_INDEX_BASE_ZERO,N_mkl,N_mkl,
                &ham_csr_ridx[0,0],&ham_csr_ridx[1,0],
                &ham_csr_cidx[0],&ham[m,n,0]
            )
            mkl_sparse_optimize(csrH[m*4+n])

    Zbuf = <double*>malloc(s_d*mb*nb_rhs)    # RHS/solution [N,Np] 2D
    drT = <double*>malloc(s_d*N*Np_num)      # dr^T[i]/Z^T [N,Np_num] column-major
    Cbuf = <double*>malloc(s_d*N*Np_num)     # H*Z^T [N,Np_num] column-major
    # Phase2+3: per xyz, solve Z=(dr[i]*S^-1)^T then assemble drSH[n,i]
    for i in range(3):
        # dr[i][Np,N] -> dr^T[N,Np_num] column-major (no physical transpose:
        # ScaLAPACK reads the [Np,N] home block as [N,Np] via +dr^T, factor=+1)
        sparse2dense_transpose(
            dr,i,dr_csr_ridx,dr_csr_cidx,drp_min,
            N,Np_min,Np_max,Np_num,1.0,drT
        )
        pdgemr2d(   # 1D Np-column split -> 2D block-cyclic RHS
            &N_mkl,&Np_mkl,drT,&i1,&i1,descRHSp,
            Zbuf,&i1,&i1,descRHS,&ictxt
        )
        pdpotrs(   # solve S*Zbuf=dr^T -> Zbuf = Z^T = (dr[i]*S^-1)^T [N,Np]
            'U',&N_mkl,&Np_mkl,Sbuf,&i1,&i1,descS,
            Zbuf,&i1,&i1,descRHS,&info
        )
        pdgemr2d(   # back to [N,Np_num] column-major (this proc's Np columns)
            &N_mkl,&Np_mkl,Zbuf,&i1,&i1,descRHS,
            drT,&i1,&i1,descRHSp,&ictxt2
        )
        if Np_num>0:
            for n in range(4):
                for m in range(2):
                    # C = H[m,n]*Z^T [N,Np_num] col-major; its memory read as
                    # row-major [Np_num,N] is Z*H = dr*S^-1*H (H symmetric),
                    # i.e. the drSH[n,i] row block -> no explicit transpose
                    mkl_sparse_d_mm(
                        SPARSE_OPERATION_NON_TRANSPOSE,1.0,csrH[m*4+n],
                        descrH,SPARSE_LAYOUT_COLUMN_MAJOR,drT,
                        Np_num,N_mkl,0.0,Cbuf,N_mkl
                    )
                    # local reshape [Np_num,N]->[nR,Np_num*Np] + real/imag(m)
                    mat_reshape_new(Cbuf,m,N,Np,Np_num,drSH[n,i])

#    for i in range(8):
#        mkl_sparse_destroy(csrH[i])   # crashes on this MKL build, leak instead
    free(Nproc)
    free(Nproc_num)
    free(Sbuf)
    free(Zbuf)
    free(drT)
    free(Cbuf)

    blacs_gridexit(&ictxt)
    blacs_gridexit(&ictxt1)
    blacs_gridexit(&ictxt2)

    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("olpinv time: %.5fs.\n",endtime-starttime)
