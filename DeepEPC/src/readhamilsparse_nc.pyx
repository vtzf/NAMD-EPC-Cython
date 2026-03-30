#cython: language_level=3
#cython: cdivision=True

cimport cython
from epc cimport *
from hdf5 cimport *

cdef int norb_u
cdef int norbital_s
cdef int R_num
cdef int* atom_idx
cdef int* atom_idx_all
cdef double* data_buf
cdef int* key_buf
cdef double Hartree2eV = 27.211396641308


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readscfout(
    char* name, int ncell2, int[:,::1] key_num, int[:,::1] pub_key, 
    double* data_buf, double complex* hamil_buf
):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j ,k, spin, ct_AN, h_AN, Gh_AN, atomnum
    cdef int iR, jR, Rij, TNO1, TNO2, atomi, atomj, Rji, offset, offset_i
    cdef int ns = key_num[ncell2,3]
    cdef int* atv_ijk
    cdef int* FNAN
    cdef int** natn
    cdef int** ncn
    cdef int* key_idx = <int*>malloc(sizeof(int)*ncell2)

    memset(hamil_buf,0,sizeof(double complex)*4*ns)

    fp = fopen(name,'rb')
    fseek(fp,0,SEEK_SET)
    fread(i_vec,sizeof(int),6,fp)
    atomnum = i_vec[0]
    TCpyCell = i_vec[5]
    fseek(fp,4+(TCpyCell+1)*4*(8+4),SEEK_CUR)
    fseek(fp,atomnum*4,SEEK_CUR)

    FNAN = <int*>malloc(sizeof(int)*(atomnum+1))
    FNAN[0] = 0
    fread(&(FNAN[1]),sizeof(int),atomnum,fp)

    natn = <int**>malloc(sizeof(int*)*(atomnum+1));
    for ct_AN in range(1,atomnum+1):
        natn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(natn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    ncn = <int**>malloc(sizeof(int*)*(atomnum+1));
    for ct_AN in range(1,atomnum+1):
        ncn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(ncn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    fseek(fp,(3+3+atomnum)*4*8,SEEK_CUR)

    for spin in range(4):
        for i in range(ncell2):
            key_idx[i] = key_num[i,3]
        for ct_AN in range(1,atomnum+1):
            iR = <int>(atom_idx_all[ct_AN-1]/norb_u)
            TNO1 = atom_idx[ct_AN-1]
            for h_AN in range(FNAN[ct_AN]+1):
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = atom_idx[Gh_AN-1]
                if (ncn[ct_AN][h_AN]==0):
                    jR = <int>(atom_idx_all[Gh_AN-1]/norb_u)
                    Rij = iR*R_num+jR
                    offset = key_idx[Rij]
                    fread(data_buf,sizeof(double),TNO1*TNO2,fp)
                    if spin == 0:
                        for i in range(TNO1*TNO2):
                            hamil_buf[i+offset] = data_buf[i]*Hartree2eV
                    elif spin == 1:
                        for i in range(TNO1*TNO2):
                            hamil_buf[3*ns+i+offset] = data_buf[i]*Hartree2eV
                    elif spin == 2:
                        for i in range(TNO1*TNO2):
                            hamil_buf[ns+i+offset] += data_buf[i]*Hartree2eV*0.0
                    else:
                        for i in range(TNO1*TNO2):
                            hamil_buf[ns+i+offset] += 1j*data_buf[i]*Hartree2eV*0.0
                    key_idx[Rij] += TNO1*TNO2
                else:
                    fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    for spin in range(3):
        for i in range(ncell2):
            key_idx[i] = key_num[i,3]
        for ct_AN in range(1,atomnum+1):
            iR = <int>(atom_idx_all[ct_AN-1]/norb_u)
            TNO1 = atom_idx[ct_AN-1]
            for h_AN in range(FNAN[ct_AN]+1):
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = atom_idx[Gh_AN-1]
                if (ncn[ct_AN][h_AN]==0):
                    jR = <int>(atom_idx_all[Gh_AN-1]/norb_u)
                    Rij = iR*R_num+jR
                    offset = key_idx[Rij]
                    fread(data_buf,sizeof(double),TNO1*TNO2,fp)
                    if spin == 0:
                        for i in range(TNO1*TNO2):
                            hamil_buf[i+offset] += 1j*data_buf[i]*Hartree2eV
                    elif spin == 1:
                        for i in range(TNO1*TNO2):
                            hamil_buf[3*ns+i+offset] += 1j*data_buf[i]*Hartree2eV
                    else:
                        for i in range(TNO1*TNO2):
                            hamil_buf[ns+i+offset] += 1j*data_buf[i]*Hartree2eV*0.0
                    key_idx[Rij] += TNO1*TNO2
                else:
                    fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    for i in range(ncell2):
        key_idx[i] = key_num[i,3]
    for ct_AN in range(1,atomnum+1):
        atomi = ct_AN-1
        iR = <int>(atom_idx_all[ct_AN-1]/norb_u)
        TNO1 = atom_idx[ct_AN-1]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            atomj = Gh_AN-1
            TNO2 = atom_idx[Gh_AN-1]
            if (ncn[ct_AN][h_AN]==0):
                jR = <int>(atom_idx_all[Gh_AN-1]/norb_u)
                Rij = iR*R_num+jR
                # get H[Rji][i(dn),j(up)] offset_i
                Rji = jR*R_num+iR
                offset_i = -1
                for i in range(key_num[Rji,2],key_num[Rji+1,2]):
                    if (pub_key[i,0]==atomj and pub_key[i,1]==atomi):
                        offset_i = pub_key[i,5]
                        break
                # for one system, H[i,j] and H[j,i] must both exist
                # H[Rji][i(dn),j(up)] = H[Rij][j(up),i(dn)].conj().T
                offset = key_idx[Rij]
                for i in range(TNO1):
                    for j in range(TNO2):
                        hamil_buf[2*ns+j*TNO1+i+offset_i] \
                        = conj(hamil_buf[ns+i*TNO2+j+offset])

                key_idx[Rij] += TNO1*TNO2

    fclose(fp)
    for ct_AN in range(1,atomnum+1):
        free(natn[ct_AN])
        free(ncn[ct_AN])
    free(natn)
    free(ncn)
    free(FNAN)
    free(key_idx)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readscfout_keymap(
    char* name, int ncell2, int[:,::1] key_num, int[:,::1] pub_key, int* key_map
):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j ,k, ct_AN, h_AN, Gh_AN, atomnum
    cdef int atomi, atomj, iR, jR, Rij, TNO1, TNO2, offset
    cdef int* atv_ijk
    cdef int* FNAN
    cdef int** natn
    cdef int** ncn

    for i in range((key_num[ncell2,2]+40)*3):
        key_map[i] = -1
    fp = fopen(name,'rb')
    fseek(fp,0,SEEK_SET)
    fread(i_vec,sizeof(int),6,fp)
    atomnum = i_vec[0]
    TCpyCell = i_vec[5]
    fseek(fp,4+(TCpyCell+1)*4*(8+4),SEEK_CUR)
    fseek(fp,atomnum*4,SEEK_CUR)

    FNAN = <int*>malloc(sizeof(int)*(atomnum+1))
    FNAN[0] = 0
    fread(&(FNAN[1]),sizeof(int),atomnum,fp)

    natn = <int**>malloc(sizeof(int*)*(atomnum+1));
    for ct_AN in range(1,atomnum+1):
        natn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(natn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    ncn = <int**>malloc(sizeof(int*)*(atomnum+1));
    for ct_AN in range(1,atomnum+1):
        ncn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(ncn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    fseek(fp,(3+3+atomnum)*4*8,SEEK_CUR)

    j = 0
    for ct_AN in range(1,atomnum+1):
        atomi = ct_AN-1
        iR = <int>(atom_idx_all[ct_AN-1]/norb_u)
        TNO1 = atom_idx[atomi]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            atomj = Gh_AN-1
            TNO2 = atom_idx[atomj]
            if (ncn[ct_AN][h_AN]==0):
                jR = <int>(atom_idx_all[Gh_AN-1]/norb_u)
                Rij = iR*R_num+jR
                for i in range(key_num[Rij,2],key_num[Rij+1,2]):
                    if (pub_key[i,0]==atomi and pub_key[i,1]==atomj):
                        key_map[j*3] = i
                        key_map[j*3+1] = atomi
                        key_map[j*3+2] = atomj
                        break
                j += 1

            fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    fclose(fp)
    for ct_AN in range(1,atomnum+1):
        free(natn[ct_AN])
        free(ncn[ct_AN])
    free(natn)
    free(ncn)
    free(FNAN)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readscfout1(
    char* name, int ncell2, int[:,::1] key_num, int[:,::1] pub_key, 
    int[:,::1] key_info1, double* data_buf, double complex* hamil_buf, 
    double complex[:,::1] epc, double f, bint LADD
):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j, k, l, spin, ct_AN, h_AN, Gh_AN, atomnum
    cdef int iR, jR, Rij, TNO1, TNO2, atomi, atomj, Rji, offset, offset_i
    cdef int ns = key_num[ncell2,3]
    cdef int* atv_ijk
    cdef int* FNAN
    cdef int** natn
    cdef int** ncn
    cdef int* key_map = <int*>malloc(sizeof(int)*3*(key_num[ncell2,2]+40))
    cdef char kname[500]

    if not LADD:
        memset(hamil_buf,0,sizeof(double complex)*4*ns)

    readscfout_keymap(name,ncell2,key_num,pub_key,key_map)
    
    fp = fopen(name,'rb')
    fseek(fp,0,SEEK_SET)
    fread(i_vec,sizeof(int),6,fp)
    atomnum = i_vec[0]
    TCpyCell = i_vec[5]
    fseek(fp,4+(TCpyCell+1)*4*(8+4),SEEK_CUR)
    fseek(fp,atomnum*4,SEEK_CUR)

    FNAN = <int*>malloc(sizeof(int)*(atomnum+1))
    FNAN[0] = 0
    fread(&(FNAN[1]),sizeof(int),atomnum,fp)

    natn = <int**>malloc(sizeof(int*)*(atomnum+1));
    for ct_AN in range(1,atomnum+1):
        natn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(natn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    ncn = <int**>malloc(sizeof(int*)*(atomnum+1));
    for ct_AN in range(1,atomnum+1):
        ncn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(ncn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    fseek(fp,(3+3+atomnum)*4*8,SEEK_CUR)

    for spin in range(4):
        j = 0
        for ct_AN in range(1,atomnum+1):
            TNO1 = atom_idx[ct_AN-1]
            for h_AN in range(FNAN[ct_AN]+1):
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = atom_idx[Gh_AN-1]
                if (ncn[ct_AN][h_AN]==0):
                    if key_map[j*3] >= 0:
                        offset = pub_key[key_map[j*3],5]
                        fread(data_buf,sizeof(double),TNO1*TNO2,fp)
                        if LADD:
                            if spin == 0:
                                for i in range(TNO1*TNO2):
                                    epc[0,key_info1[i+offset,1]] \
                                    = data_buf[i]*Hartree2eV
                            elif spin == 1:
                                for i in range(TNO1*TNO2):
                                    epc[3,key_info1[i+offset,1]] \
                                    = data_buf[i]*Hartree2eV
                            elif spin == 2:
                                for i in range(TNO1*TNO2):
                                    epc[1,key_info1[i+offset,1]] \
                                    += data_buf[i]*Hartree2eV*0.0
                            else:
                                for i in range(TNO1*TNO2):
                                    epc[1,key_info1[i+offset,1]] \
                                    += 1j*data_buf[i]*Hartree2eV*0.0
                        else:
                            if spin == 0:
                                for i in range(TNO1*TNO2):
                                    hamil_buf[i+offset] \
                                    = data_buf[i]*Hartree2eV
                            elif spin == 1:
                                for i in range(TNO1*TNO2):
                                    hamil_buf[3*ns+i+offset] \
                                    = data_buf[i]*Hartree2eV
                            elif spin == 2:
                                for i in range(TNO1*TNO2):
                                    hamil_buf[ns+i+offset] \
                                    += data_buf[i]*Hartree2eV
                            else:
                                for i in range(TNO1*TNO2):
                                    hamil_buf[ns+i+offset] \
                                    += 1j*data_buf[i]*Hartree2eV
                    else:
                        fseek(fp,TNO1*TNO2*8,SEEK_CUR)
                    j += 1
                else:
                    fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    for spin in range(3):
        j = 0
        for ct_AN in range(1,atomnum+1):
            TNO1 = atom_idx[ct_AN-1]
            for h_AN in range(FNAN[ct_AN]+1):
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = atom_idx[Gh_AN-1]
                if (ncn[ct_AN][h_AN]==0):
                    if key_map[j*3] >= 0:
                        offset = pub_key[key_map[j*3],5]
                        fread(data_buf,sizeof(double),TNO1*TNO2,fp)
                        if LADD:
                            if spin == 0:
                                for i in range(TNO1*TNO2):
                                    epc[0,key_info1[i+offset,1]] \
                                    += 1j*data_buf[i]*Hartree2eV
                            elif spin == 1:
                                for i in range(TNO1*TNO2):
                                    epc[3,key_info1[i+offset,1]] \
                                    += 1j*data_buf[i]*Hartree2eV
                            else:
                                for i in range(TNO1*TNO2):
                                    epc[1,key_info1[i+offset,1]] \
                                    += 1j*data_buf[i]*Hartree2eV*0.0
                        else:
                            if spin == 0:
                                for i in range(TNO1*TNO2):
                                    hamil_buf[i+offset] \
                                    += 1j*data_buf[i]*Hartree2eV
                            elif spin == 1:
                                for i in range(TNO1*TNO2):
                                    hamil_buf[3*ns+i+offset] \
                                    += 1j*data_buf[i]*Hartree2eV
                            else:
                                for i in range(TNO1*TNO2):
                                    hamil_buf[ns+i+offset] \
                                    += 1j*data_buf[i]*Hartree2eV
                    else:
                        fseek(fp,TNO1*TNO2*8,SEEK_CUR)
                    j += 1
                else:
                    fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    j = 0
    for ct_AN in range(1,atomnum+1):
        atomi = ct_AN-1
        TNO1 = atom_idx[ct_AN-1]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            atomj = Gh_AN-1
            TNO2 = atom_idx[Gh_AN-1]
            if (ncn[ct_AN][h_AN]==0):
                # if H[j(up),i(dn)] exists
                if key_map[j*3] >= 0:
                    # find H[i(dn),j(up)] offset_i
                    offset_i = -1
                    for i in range(key_num[ncell2,2]+40):
                        if (key_map[i*3+1]==atomj and key_map[i*3+2]==atomi):
                            offset_i = pub_key[key_map[i*3],5]
                            break
                    offset = pub_key[key_map[j*3],5]
                    # make sure H[i(dn),j(up)] exists
                    if offset_i >= 0:
                        if LADD:
                            for k in range(TNO1):
                                for l in range(TNO2):
                                    epc[2,key_info1[l*TNO1+k+offset_i,1]] \
                                    = conj(epc[1,key_info1[k*TNO2+l+offset,1]])
                        else:
                            for k in range(TNO1):
                                for l in range(TNO2):
                                    hamil_buf[2*ns+l*TNO1+k+offset_i] \
                                    = conj(hamil_buf[ns+k*TNO2+l+offset])
                    # if H[i(dn),j(up)] does not exist, 
                    # set H[j(up),i(dn)] = 0.0 to ensure symmetry
                    else:
                        if LADD:
                            for k in range(4):
                                for l in range(TNO1*TNO2):
                                    epc[k,key_info1[l+offset,1]] = 0.0
                        else:
                            for k in range(4):
                                for l in range(TNO1*TNO2):
                                    hamil_buf[k*ns+l+offset] = 0.0

                j += 1

    if LADD:
        j = 0
        for ct_AN in range(1,atomnum+1):
            TNO1 = atom_idx[ct_AN-1]
            for h_AN in range(FNAN[ct_AN]+1):
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = atom_idx[Gh_AN-1]
                if (ncn[ct_AN][h_AN]==0):
                    if key_map[j*3] >= 0:
                        offset = pub_key[key_map[j*3],5]
                        for k in range(4):
                            for i in range(TNO1*TNO2):
                                l = key_info1[i+offset,1]
                                epc[k,l] = (epc[k,l]-hamil_buf[k*ns+i+offset])*f
                    j += 1

    fclose(fp)
    for ct_AN in range(1,atomnum+1):
        free(natn[ct_AN])
        free(ncn[ct_AN])
    free(natn)
    free(ncn)
    free(FNAN)
    free(key_map)


cdef void readh5_p0(
    mpi.MPI_Comm comm_b, char* h5_name, int nprocs_b, int myid_b, 
    int ncell2, int[:,::1] key_num, int[:,::1] pub_key, 
    double* data_buf, double complex* hamil_buf
):
    cdef hid_t f, data_id
    cdef herr_t status
    cdef int h, i, j, k, key_min, key_max, offset, TNO1, TNO2
    cdef int ns = key_num[ncell2,3]
    cdef double complex* databuf = <double complex*>data_buf
    cdef char key_t[100]

    if (myid_b==0):
        memset(hamil_buf,0,sizeof(double complex)*4*ns)

    mpi.MPI_Barrier(comm_b)
    key_min = <int>((key_num[ncell2,2]*myid_b)/nprocs_b)
    key_max = <int>((key_num[ncell2,2]*(myid_b+1))/nprocs_b)

    f = H5Fopen(h5_name,H5F_ACC_RDONLY,H5P_DEFAULT)
    for h in range(key_min,key_max):
        sprintf(key_t,"[0, 0, 0, %d, %d]",pub_key[h,0]+1,pub_key[h,1]+1)
        data_id = H5Dopen(f,key_t,H5P_DEFAULT)
        status = H5Dread(
            data_id,H5Dget_type(data_id),H5S_ALL,
            H5S_ALL,H5P_DEFAULT,databuf
        )
        TNO1 = pub_key[h,2]
        TNO2 = pub_key[h,3]
        offset = pub_key[h,5]
        for i in range(TNO1):
            for j in range(TNO2):
                k = i*TNO2+j+offset
                hamil_buf[k] = databuf[i*TNO2*2+j]
                hamil_buf[ns+k] = databuf[i*TNO2*2+j+TNO2]
                hamil_buf[2*ns+k] = databuf[(i+TNO1)*TNO2*2+j]
                hamil_buf[3*ns+k] = databuf[(i+TNO1)*TNO2*2+j+TNO2]

        status = H5Dclose(data_id)

    status = H5Fclose(f)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readh5_p1(
    mpi.MPI_Comm comm_b, char* h5_name, int nprocs_b, int myid_b, 
    int ncell2, int[:,::1] key_num, int[:,::1] pub_key, 
    int[:,::1] key_info1, double factor, double* data_buf, 
    double complex* hamil_buf, double complex[:,::1] dhamil, bint LADD
):
    cdef hid_t f, data_id
    cdef herr_t status
    cdef int h, i, j, k, l, key_min, key_max, offset, TNO1, TNO2
    cdef int ns = key_num[ncell2,3]
    cdef double complex* databuf = <double complex*>data_buf
    cdef char key_t[100]
    cdef char key_t1[100]

    if (myid_b==0):
        if not LADD:
            memset(hamil_buf,0,sizeof(double complex)*4*ns)

    mpi.MPI_Barrier(comm_b)
    key_min = <int>((key_num[ncell2,2]*myid_b)/nprocs_b)
    key_max = <int>((key_num[ncell2,2]*(myid_b+1))/nprocs_b)

    f = H5Fopen(h5_name,H5F_ACC_RDONLY,H5P_DEFAULT)
    for h in range(key_min,key_max):
        sprintf(key_t,"[0, 0, 0, %d, %d]",pub_key[h,0]+1,pub_key[h,1]+1)
        if H5Lexists(f,key_t,H5P_DEFAULT):
            sprintf(key_t1,"[0, 0, 0, %d, %d]",pub_key[h,1]+1,pub_key[h,0]+1)
            if H5Lexists(f,key_t1,H5P_DEFAULT):
                data_id = H5Dopen(f,key_t,H5P_DEFAULT)
                status = H5Dread(
                    data_id,H5Dget_type(data_id),H5S_ALL,
                    H5S_ALL,H5P_DEFAULT,databuf
                )
                TNO1 = pub_key[h,2]
                TNO2 = pub_key[h,3]
                offset = pub_key[h,5]
                if LADD:
                    for i in range(TNO1):
                        for j in range(TNO2):
                            k = i*TNO2+j+offset
                            l = key_info1[k,1]
                            dhamil[0,l] = (databuf[i*TNO2*2+j] \
                                         - hamil_buf[k])*factor
                            dhamil[1,l] = (databuf[i*TNO2*2+j+TNO2] \
                                         - hamil_buf[ns+k])*factor
                            dhamil[2,l] = (databuf[(i+TNO1)*TNO2*2+j] \
                                         - hamil_buf[2*ns+k])*factor
                            dhamil[3,l] = (databuf[(i+TNO1)*TNO2*2+j+TNO2] \
                                         - hamil_buf[3*ns+k])*factor
                else:
                    for i in range(TNO1):
                        for j in range(TNO2):
                            k = i*TNO2+j+offset
                            hamil_buf[k] = databuf[i*TNO2*2+j]
                            hamil_buf[ns+k] = databuf[i*TNO2*2+j+TNO2]
                            hamil_buf[2*ns+k] = databuf[(i+TNO1)*TNO2*2+j]
                            hamil_buf[3*ns+k] = databuf[(i+TNO1)*TNO2*2+j+TNO2]
                status = H5Dclose(data_id)

    status = H5Fclose(f)


@cython.boundscheck(False)
@cython.wraparound(False)
def deltahamil_b(
    MPI.Comm comm, MPI.Comm shm_comm, int nmodes, int nm_min,  
    int dH_block, int norbital_u, int ncell, int norbital_m, 
    int atomnum_py, int[::1] atom_idx_py, 
    int[::1] atom_idx_all_py, int[::1] catom, int[:,::1] key_num, 
    int[:,::1] pub_key, int[:,::1] key_info1, double dQ1, 
    double complex[:,:,::1] dhamil, char* inDir, char* dhamilDir, 
    char* H5HamName, char* dhamil_method, bint IsH5
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm c_shm_comm = shm_comm.ob_mpi
    cdef mpi.MPI_Comm comm_b, comm_gp
    cdef mpi.MPI_Group group_all, group
    cdef mpi.MPI_Datatype DB_NOB
    cdef mpi.MPI_Aint l_hamil_buf
    cdef mpi.MPI_Win win
    cdef int h, i, j, k, l, m, n, myid, nprocs, ierr, shm_id, nprocs_p_shm, \
             nprocs_p, color, myid_b, nprocs_b, nprocs_shm, nnodes
    cdef int s_d = sizeof(double)
    cdef int s_dcplx = sizeof(double complex)
    cdef int s_int = sizeof(int)
    cdef int ncell2 = ncell*ncell
    cdef int* nodelist
    cdef int* imodes_num
    cdef int* imodes
    cdef int* iprocs_num
    cdef int* iprocs
    cdef int* imodes_num_shm
    cdef int* imodes_shm
    cdef int* imodes_num_gather
    cdef int* imodes_gather
    cdef int imodes_min, imodes_max, iprocs_min, iprocs_max

    cdef double complex* hamil_buf
    cdef double ndQ1 = -1.0*dQ1
    cdef double dQ2 = dQ1/2.0
    cdef double ndQ2 = -1.0*dQ2
    cdef double factor[2]
    cdef double starttime, endtime
    cdef char data_name[500]
    cdef char* delta = "xyz"
    cdef char* diff = "FBC"
    cdef char* plus = "+-"

    global norb_u
    global norbital_s
    global atom_idx
    global atom_idx_all
    global key_buf
    global data_buf
    global R_num

    starttime = mpi.MPI_Wtime()

    norb_u = norbital_u
    norbital_s = norbital_u
    norbital_s *= ncell
    R_num = ncell

    atom_idx = <int*>malloc(atomnum_py*sizeof(int))
    for i in range(atomnum_py):
        atom_idx[i] = atom_idx_py[i]
    atom_idx_all = <int*>malloc((atomnum_py+1)*sizeof(int))
    for i in range(atomnum_py+1):
        atom_idx_all[i] = atom_idx_all_py[i]
    key_buf = <int*>malloc(ncell2*sizeof(int))

    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)
    ierr = mpi.MPI_Comm_size(c_shm_comm,&nprocs_shm)
    ierr = mpi.MPI_Comm_rank(c_shm_comm,&shm_id)
    nnodes = <int>(nprocs/nprocs_shm)

    nprocs_p_shm = <int>(nprocs_shm/dH_block)
    nprocs_p = nprocs_p_shm*nnodes
    imodes_num = <int*>malloc(sizeof(int)*nprocs_p)
    imodes = <int*>malloc(sizeof(int)*(nprocs_p+1))
    iprocs_num = <int*>malloc(sizeof(int)*nprocs_p)
    iprocs = <int*>malloc(sizeof(int)*(nprocs_p+1))

    imodes[0] = 0
    iprocs[0] = 0
    for i in range(nprocs_p):
        iprocs_min = <int>((nprocs_shm*i)/nprocs_p_shm)
        iprocs_max = <int>((nprocs_shm*(i+1))/nprocs_p_shm)
        imodes_min = <int>((nmodes*i)/nprocs_p)
        imodes_max = <int>((nmodes*(i+1))/nprocs_p)
        iprocs_num[i] = iprocs_max - iprocs_min
        imodes_num[i] = imodes_max - imodes_min
        iprocs[i+1] = iprocs_max
        imodes[i+1] = imodes_max
        if (iprocs_min<=myid and iprocs_max>myid):
            color = i
    imodes_min = imodes[color]
    imodes_max = imodes[color+1]

    nodelist = <int*>malloc(sizeof(int)*nnodes)
    imodes_num_shm = <int*>malloc(sizeof(int)*nnodes)
    imodes_shm = <int*>malloc(sizeof(int)*(nnodes+1))
    imodes_shm[0] = 0
    l = 0
    for i in range(nnodes):
        nodelist[i] = i*nprocs_shm
        k = nprocs_shm*(i+1)
        for j in range(l,nprocs_p):
            if (iprocs[j]<k and iprocs[j+1]>=k):
                break
        imodes_num_shm[i] = (imodes[j+1]-imodes[l])
        imodes_shm[i+1] = imodes[j+1]
        l = j+1

    mpi.MPI_Comm_group(c_comm,&group_all)
    mpi.MPI_Group_incl(group_all,nnodes,nodelist,&group)
    ierr = mpi.MPI_Comm_create(c_comm,group,&comm_gp)
    mpi.MPI_Comm_split(c_comm,color,myid,&comm_b)
    ierr = mpi.MPI_Comm_size(comm_b,&nprocs_b)
    ierr = mpi.MPI_Comm_rank(comm_b,&myid_b)

    data_buf = <double*>malloc(norbital_m*norbital_m*8*s_d)

    if (myid_b==0):
        l_hamil_buf = 4*key_num[ncell2,3]*sizeof(double complex)
    else:
        l_hamil_buf = 0
    mpi.MPI_Win_allocate_shared(
        l_hamil_buf,s_dcplx,mpi.MPI_INFO_NULL,comm_b,&hamil_buf,&win
    )
    if (myid_b!=0):
        mpi.MPI_Win_shared_query(win,0,&l_hamil_buf,&s_dcplx,&hamil_buf)
    mpi.MPI_Barrier(c_comm)

    if (dhamil_method[0]!=diff[2]):
        if (dhamil_method[0]==diff[0]):
            factor[0] = dQ1
            h = 0
        else:
            factor[1] = ndQ1
            h = 1
        if IsH5:
            sprintf(data_name,"%s/%s.h5",inDir,H5HamName)
            readh5_p0(
                comm_b,data_name,nprocs_b,myid_b,
                ncell2,key_num,pub_key,data_buf,hamil_buf
            )
        else:
            if (myid_b==0):
                sprintf(data_name,"%s/openmx.scfout",inDir)
                readscfout(
                    data_name,ncell2,key_num,pub_key,data_buf,hamil_buf
                )
        mpi.MPI_Barrier(comm_b)
        for i in range(imodes_min,imodes_max):
            j = <int>((i+nm_min)/3)
            k = (i+nm_min)%3
            if IsH5:
                #sprintf(data_name,"%s/%s.h5",inDir,H5HamName)
                sprintf(
                    data_name,"%s/%s/%d%c%c/%s.h5",inDir,dhamilDir,
                    catom[j],plus[h],delta[k],H5HamName
                )
                readh5_p1(
                    comm_b,data_name,nprocs_b,myid_b,ncell2,
                    key_num,pub_key,key_info1,factor[h],
                    data_buf,hamil_buf,dhamil[i],1
                )
            else:
                if (myid_b==0):
                    sprintf(
                        data_name,"%s/%s/%d%c%c/openmx.scfout",
                        inDir,dhamilDir,catom[j],plus[h],delta[k]
                    )
                    readscfout1(
                        data_name,ncell2,key_num,pub_key,key_info1,
                        data_buf,hamil_buf,dhamil[i],factor[h],1
                    )
            mpi.MPI_Barrier(comm_b)
    else:
        for i in range(imodes_min,imodes_max):
            j = <int>((i+nm_min)/3)
            k = (i+nm_min)%3
            # read catom-xyz
            if IsH5:
                sprintf(
                    data_name,"%s/%s/%d-%c/%s.h5",inDir,
                    dhamilDir,catom[j],delta[k],H5HamName
                )
                readh5_p1(
                    comm_b,data_name,nprocs_b,myid_b,ncell2,
                    key_num,pub_key,key_info1,dQ2,
                    data_buf,hamil_buf,dhamil[i],0
                )
            else:
                if (myid_b==0):
                    sprintf(
                        data_name,"%s/%s/%d-%c/openmx.scfout",
                        inDir,dhamilDir,catom[j],delta[k]
                    )
                    readscfout1(
                        data_name,ncell2,key_num,pub_key,key_info1,
                        data_buf,hamil_buf,dhamil[i],dQ2,0
                    )
            mpi.MPI_Barrier(comm_b)
            # read catom+xyz
            if IsH5:
                sprintf(
                    data_name,"%s/%s/%d+%c/%s.h5",inDir,
                    dhamilDir,catom[j],delta[k],H5HamName
                )
                readh5_p1(
                    comm_b,data_name,nprocs_b,myid_b,ncell2,
                    key_num,pub_key,key_info1,dQ2,
                    data_buf,hamil_buf,dhamil[i],1
                )
            else:
                if (myid_b==0):
                    sprintf(
                        data_name,"%s/%s/%d+%c/openmx.scfout",
                        inDir,dhamilDir,catom[j],delta[k]
                    )
                    readscfout1(
                        data_name,ncell2,key_num,pub_key,key_info1,
                        data_buf,hamil_buf,dhamil[i],dQ2,1
                    )
            mpi.MPI_Barrier(comm_b)

    mpi.MPI_Barrier(c_shm_comm)
    if (nnodes>1):
        mpi.MPI_Type_contiguous(key_num[ncell2,3],mpi.MPI_DOUBLE_COMPLEX,&DB_NOB)
        mpi.MPI_Type_commit(&DB_NOB)
        if (comm_gp!=mpi.MPI_COMM_NULL):
            for i in range(nnodes):
                for j in range(imodes_shm[i],imodes_shm[i+1]):
                    for k in range(4):
                        mpi.MPI_Bcast(&dhamil[j,k,0],1,DB_NOB,i,comm_gp)

        mpi.MPI_Type_free(&DB_NOB)

    mpi.MPI_Win_free(&win)

    free(nodelist)
    free(imodes_num)
    free(imodes)
    free(iprocs_num)
    free(iprocs)
    free(imodes_num_shm)
    free(imodes_shm)
    free(atom_idx)
    free(atom_idx_all)
    free(key_buf)
    free(data_buf)

    mpi.MPI_Barrier(c_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("dhamil time in mode[%4d:%4d]:%12.4fs.\n",nm_min,nm_min+nmodes,endtime-starttime)
