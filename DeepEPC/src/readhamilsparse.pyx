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
#cdef double Hartree2eV = 27.2113845


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
cdef void readscfout_key0(
    char* name, int[:,::1] key_num
):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j ,k, ct_AN, h_AN, Gh_AN, iR, jR, Rij, atomnum
    cdef int* atv_ijk
    cdef int* FNAN
    cdef int** natn
    cdef int** ncn

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

    natn = <int**>malloc(sizeof(int*)*(atomnum+1))
    for ct_AN in range(1,atomnum+1):
        natn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(natn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    ncn = <int**>malloc(sizeof(int*)*(atomnum+1));
    for ct_AN in range(1,atomnum+1):
        ncn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(ncn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    fseek(fp,(3+3+atomnum)*4*8,SEEK_CUR)

    for ct_AN in range(1,atomnum+1):
        iR = <int>(atom_idx_all[ct_AN-1]/norb_u)
        TNO1 = atom_idx[ct_AN-1]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            TNO2 = atom_idx[Gh_AN-1]
            if (ncn[ct_AN][h_AN]==0):
                jR = <int>(atom_idx_all[Gh_AN-1]/norb_u)
                Rij = iR*R_num+jR
                key_num[Rij,0] += 1
                key_num[Rij,1] += TNO1*TNO2

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
cdef void readscfout_key1(
    char* name, int ncell2, int[:,::1] key_num, int[:,::1] pub_key
):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j ,k, ct_AN, h_AN, Gh_AN, \
             iR, jR, Rij, atomi, atomj, atomnum
    cdef int* atv_ijk
    cdef int* FNAN
    cdef int** natn
    cdef int** ncn
    cdef int* key_idx = <int*>malloc(sizeof(int)*ncell2)

    for i in range(ncell2):
        key_idx[i] = key_num[i,2]
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

    natn = <int**>malloc(sizeof(int*)*(atomnum+1))
    for ct_AN in range(1,atomnum+1):
        natn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(natn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    ncn = <int**>malloc(sizeof(int*)*(atomnum+1))
    for ct_AN in range(1,atomnum+1):
        ncn[ct_AN] = <int*>malloc(sizeof(int)*(FNAN[ct_AN]+1))
        fread(ncn[ct_AN],sizeof(int),FNAN[ct_AN]+1,fp)
    fseek(fp,(3+3+atomnum)*4*8,SEEK_CUR)

    for ct_AN in range(1,atomnum+1):
        iR = <int>(atom_idx_all[ct_AN-1]/norb_u)
        atomi = ct_AN-1
        TNO1 = atom_idx[atomi]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            atomj = Gh_AN-1
            TNO2 = atom_idx[atomj]
            if (ncn[ct_AN][h_AN]==0):
                jR = <int>(atom_idx_all[Gh_AN-1]/norb_u)
                Rij = iR*R_num+jR
                pub_key[key_idx[Rij],0] = atomi
                pub_key[key_idx[Rij],1] = atomj
                pub_key[key_idx[Rij],2] = TNO1
                pub_key[key_idx[Rij],3] = TNO2
                pub_key[key_idx[Rij],4] = TNO1*TNO2
                key_idx[Rij] += 1

            fseek(fp,TNO1*TNO2*8,SEEK_CUR)

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
cdef herr_t readh5engine_key0(
    hid_t loc_id, char* name, H5O_info1_t* info, void* key_num
):
    cdef int idx[5]
    cdef herr_t status
    cdef hid_t data_id
    cdef int i, j, iR, jR, Rij

    sscanf(name,"[%d, %d, %d, %d, %d]",\
           &idx[0],&idx[1],&idx[2],&idx[3],&idx[4])
    if (idx[0]==0 and idx[1]==0 and idx[2]==0):
        iR = <int>(atom_idx_all[idx[3]-1]/norb_u)
        jR = <int>(atom_idx_all[idx[4]-1]/norb_u)
        Rij = iR*R_num+jR
        (<int*>(key_num))[Rij*4] += 1
        (<int*>(key_num))[Rij*4+1] += atom_idx[idx[3]-1]\
                                    * atom_idx[idx[4]-1]
        return 0
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef herr_t readh5engine_key1(
    hid_t loc_id, char* name, H5O_info1_t* info, void* pub_key
):
    cdef int idx[5]
    cdef herr_t status
    cdef hid_t data_id
    cdef int i, j, iR, jR, Rij, atomi, atomj, TNOi, TNOj
    global key_buf

    sscanf(name,"[%d, %d, %d, %d, %d]",\
           &idx[0],&idx[1],&idx[2],&idx[3],&idx[4])
    if (idx[0]==0 and idx[1]==0 and idx[2]==0):
        iR = <int>(atom_idx_all[idx[3]-1]/norb_u)
        jR = <int>(atom_idx_all[idx[4]-1]/norb_u)
        Rij = iR*R_num+jR
        atomi = idx[3]-1
        atomj = idx[4]-1
        TNOi = atom_idx[atomi]
        TNOj = atom_idx[atomj]
        (<int*>(pub_key))[key_buf[Rij]*6] = atomi
        (<int*>(pub_key))[key_buf[Rij]*6+1] = atomj
        (<int*>(pub_key))[key_buf[Rij]*6+2] = TNOi
        (<int*>(pub_key))[key_buf[Rij]*6+3] = TNOj
        (<int*>(pub_key))[key_buf[Rij]*6+4] = TNOi*TNOj
        key_buf[Rij] += 1
        return 0
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readh5_key0(char* h5_name, int[:,::1] key_num):
    cdef int i, j
    cdef hid_t f
    cdef herr_t status

    f = H5Fopen(h5_name,H5F_ACC_RDONLY,H5P_DEFAULT)
    status = H5Ovisit1(
        f,H5_INDEX_NAME,H5_ITER_NATIVE,readh5engine_key0,&key_num[0,0]
    )
    status = H5Fclose(f)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readh5_key1(
    char* h5_name, int ncell2, int[:,::1] key_num, int[:,::1] pub_key
):
    cdef int i, j
    cdef hid_t f
    cdef herr_t status
    global key_buf

    for i in range(ncell2):
        key_buf[i] = key_num[i,2]
    f = H5Fopen(h5_name,H5F_ACC_RDONLY,H5P_DEFAULT)
    status = H5Ovisit1(
        f,H5_INDEX_NAME,H5_ITER_NATIVE,readh5engine_key1,&pub_key[0,0]
    )
    status = H5Fclose(f)


@cython.boundscheck(False)
@cython.wraparound(False)
def GetSparseNum(
    char* inDir, char* H5HamName, int[:,::1] key_num, 
    int[::1] atom_idx_py, int[::1] atom_idx_all_py, 
    int atomnum, int norbital_u, int ncell, int ncell2, bint IsH5
):
    cdef int i, j
    cdef char data_name[500]
    global atom_idx
    global atom_idx_all
    global R_num
    global norb_u

    R_num = ncell
    norb_u = norbital_u
    atom_idx = <int*>malloc(atomnum*sizeof(int))
    for i in range(atomnum):
        atom_idx[i] = atom_idx_py[i]
    atom_idx_all = <int*>malloc((atomnum+1)*sizeof(int))
    for i in range(atomnum+1):
        atom_idx_all[i] = atom_idx_all_py[i]
    memset(&key_num[0,0],0,(ncell2+1)*4*sizeof(int))
    if IsH5:
        sprintf(data_name,"%s/%s.h5",inDir,H5HamName)
        readh5_key0(data_name,key_num)
    else:
        sprintf(data_name,"%s/openmx.scfout",inDir)
        readscfout_key0(data_name,key_num)
    for i in range(ncell2):
        for j in range(i+1,ncell2+1):
            key_num[j,2] += key_num[i,0]
            key_num[j,3] += key_num[i,1]

    free(atom_idx)
    free(atom_idx_all)


@cython.boundscheck(False)
@cython.wraparound(False)
def GetSparseIdx(
    char* inDir, char* H5HamName, int norbital_u, int ncell, int ncell2,
    int[:,::1] key_num, int[:,::1] pub_key, int[:,::1] key_info, 
    int[:,::1] key_info1, int[::1] atom_idx_py, int[::1] atom_idx_all_py, 
    int atomnum, bint IsH5
):
    cdef int h, i, j, k, offset
    cdef char data_name[500]
    global atom_idx
    global atom_idx_all
    global key_buf
    global R_num
    global norb_u

    R_num = ncell
    norb_u = norbital_u
    atom_idx = <int*>malloc(atomnum*sizeof(int))
    for i in range(atomnum):
        atom_idx[i] = atom_idx_py[i]
    atom_idx_all = <int*>malloc((atomnum+1)*sizeof(int))
    for i in range(atomnum+1):
        atom_idx_all[i] = atom_idx_all_py[i]
    key_buf = <int*>malloc(ncell2*sizeof(int))
    if IsH5:
        sprintf(data_name,"%s/%s.h5",inDir,H5HamName)
        readh5_key1(data_name,ncell2,key_num,pub_key)
    else:
        sprintf(data_name,"%s/openmx.scfout",inDir)
        readscfout_key1(data_name,ncell2,key_num,pub_key)
    
    for h in range(key_num[ncell2,2]):
        pub_key[h,5] = 0
    for h in range(key_num[ncell2,2]):
        for i in range(h+1,key_num[ncell2,2]):
            pub_key[i,5] += pub_key[h,4]

    for h in range(key_num[ncell2,2]):
        for i in range(pub_key[h,2]):
            for j in range(pub_key[h,3]):
                k = i*pub_key[h,3]+j
                key_info[k+pub_key[h,5],0] \
                = (i+atom_idx_all[pub_key[h,0]]%norb_u)*norbital_u \
                + (j+atom_idx_all[pub_key[h,1]]%norb_u)

    for h in range(key_num[ncell2,3]):
        key_info1[h,1] = h
    for h in range(ncell2):
        for i in range(key_num[h,1]):
            key_info[i+key_num[h,3],1] = i
        qsort(&key_info[key_num[h,3],0],key_num[h,1],sizeof(int)*2,&Cmp)
        for i in range(key_num[h,1]):
            key_info1[i+key_num[h,3],0] = key_info[i+key_num[h,3],1]
        qsort(&key_info1[key_num[h,3],0],key_num[h,1],sizeof(int)*2,&Cmp)

    free(atom_idx)
    free(atom_idx_all)
    free(key_buf)


@cython.boundscheck(False)
@cython.wraparound(False)
def GetSparseNumSum(
    int ncell, int[:,::1] key_num, 
    int[:,::1] key_num_s, int[:,::1] key_info
):
    cdef int i, j, key_min, key_max, nkey, key_num_u, tmp
    cdef int * key_buf

    memset(&key_num_s[0,0],0,(ncell+1)*2*sizeof(int))
    for i in range(ncell):
        key_min = key_num[ncell*i,3]
        key_max = key_num[ncell*(i+1),3]
        nkey = key_max - key_min
        if nkey > 0:
            key_buf = <int*>malloc(nkey*sizeof(int))
            for j in range(key_min,key_max):
                key_buf[j-key_min] = key_info[j,0]
            qsort(key_buf,nkey,sizeof(int),&Cmp)
            tmp = key_buf[0]
            key_num_u = 1
            for j in range(1,nkey):
                if (key_buf[j]!=tmp):
                    tmp = key_buf[j]
                    key_num_u += 1
            free(key_buf)
            key_num_s[i,0] = key_num_u

        for j in range(i+1,ncell+1):
            key_num_s[j,1] += key_num_s[i,0]


@cython.boundscheck(False)
@cython.wraparound(False)
def GetSparseIdxSum(
    int ncell, int[:,::1] key_num, int[:,::1] key_num_s, 
    int[:,::1] key_info, int[::1] key_info_s
):
    cdef int i, j, key_num_i, key_min, key_max, nkey, key_num_u, tmp
    cdef int * key_buf
    cdef int * mapidx

    for i in range(ncell):
        key_num_i = key_num_s[i,1]
        key_min = key_num[ncell*i,3]
        key_max = key_num[ncell*(i+1),3]
        nkey = key_max - key_min
        if nkey > 0:
            key_buf = <int*>malloc(nkey*sizeof(int))
            for j in range(key_min,key_max):
                key_buf[j-key_min] = key_info[j,0]
            qsort(key_buf,nkey,sizeof(int),&Cmp)
            tmp = key_buf[0]
            key_info_s[key_num_i] = tmp
            key_num_u = 1
            for j in range(1,nkey):
                if (key_buf[j]!=tmp):
                    tmp = key_buf[j]
                    key_info_s[key_num_i+key_num_u] = tmp
                    key_num_u += 1

            mapidx = <int*>malloc(key_buf[nkey-1]*sizeof(int))
            for j in range(key_num_u):
                mapidx[key_info_s[key_num_i+j]] = j
            for j in range(key_min,key_max):
                key_info[j,1] = mapidx[key_info[j,0]]

            free(mapidx)
            free(key_buf)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readscfout(
    char* name, int ncell2, int[:,::1] key_num, 
    double* data_buf, double* hamil_buf, int Ispin
):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j ,k, ct_AN, h_AN, Gh_AN, atomnum
    cdef int iR, jR, Rij, TNO1, TNO2, spin
    cdef int ns = key_num[ncell2,3]
    cdef int* atv_ijk
    cdef int* FNAN
    cdef int** natn
    cdef int** ncn
    cdef int* key_idx = <int*>malloc(sizeof(int)*ncell2)

    memset(hamil_buf,0,sizeof(double)*(Ispin+1)*ns)

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

    for spin in range(Ispin+1):
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
                    for i in range(TNO1*TNO2):
                        hamil_buf[spin*ns+i+offset] = data_buf[i]*Hartree2eV
                    key_idx[Rij] += TNO1*TNO2
                else:
                    fseek(fp,TNO1*TNO2*8,SEEK_CUR)

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
    int[:,::1] key_info1, double* data_buf, double* hamil_buf, 
    double[:,::1] epc, double f, int Ispin, bint LADD
):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j ,k, ct_AN, h_AN, Gh_AN, atomnum
    cdef int iR, jR, Rij, TNO1, TNO2, atomi, atomj, offset, offset_i, spin
    cdef int ns = key_num[ncell2,3]
    cdef int* atv_ijk
    cdef int* FNAN
    cdef int** natn
    cdef int** ncn
    cdef int* key_map = <int*>malloc(sizeof(int)*3*(key_num[ncell2,2]+40))
    cdef char kname[500]

    if not LADD:
        memset(hamil_buf,0,sizeof(double)*(Ispin+1)*ns)

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

    for spin in range(Ispin+1):
        j = 0
        for ct_AN in range(1,atomnum+1):
            atomi = ct_AN-1
            TNO1 = atom_idx[ct_AN-1]
            for h_AN in range(FNAN[ct_AN]+1):
                Gh_AN = natn[ct_AN][h_AN]
                atomj = Gh_AN-1
                TNO2 = atom_idx[Gh_AN-1]
                if (ncn[ct_AN][h_AN]==0):
                    if key_map[j*3] >= 0:
                        # find H[i(dn),j(up)] offset_i
                        offset_i = -1
                        for i in range(key_num[ncell2,2]+40):
                            if (key_map[i*3+1]==atomj and key_map[i*3+2]==atomi):
                                offset_i = pub_key[key_map[i*3],5]
                                break
                        if offset_i >= 0:
                            offset = pub_key[key_map[j*3],5]
                            fread(data_buf,sizeof(double),TNO1*TNO2,fp)
                            if LADD:
                                for i in range(TNO1*TNO2):
                                    epc[spin,key_info1[i+offset,1]] \
                                    = (data_buf[i]*Hartree2eV \
                                     - hamil_buf[spin*ns+i+offset])*f
                            else:
                                for i in range(TNO1*TNO2):
                                    hamil_buf[spin*ns+i+offset] \
                                    = data_buf[i]*Hartree2eV
                        else:
                            fseek(fp,TNO1*TNO2*8,SEEK_CUR)
                    else:
                        fseek(fp,TNO1*TNO2*8,SEEK_CUR)
                    j += 1
                else:
                    fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    fclose(fp)
    for ct_AN in range(1,atomnum+1):
        free(natn[ct_AN])
        free(ncn[ct_AN])
    free(natn)
    free(ncn)
    free(FNAN)
    free(key_map)


cdef void readh5_p0(
    mpi.MPI_Comm comm_b, char* h5_name, int nprocs_b, 
    int myid_b, int ncell2, int[:,::1] key_num, 
    int[:,::1] pub_key, double* data_buf, double* hamil_buf
):
    cdef hid_t f, data_id
    cdef herr_t status
    cdef int h, i, j, key_min, key_max, offset, spin
    cdef int ns = key_num[ncell2,3]
    cdef char key_t[100]

    if (myid_b==0):
        memset(hamil_buf,0,sizeof(double)*ns)

    mpi.MPI_Barrier(comm_b)
    key_min = <int>((key_num[ncell2,2]*myid_b)/nprocs_b)
    key_max = <int>((key_num[ncell2,2]*(myid_b+1))/nprocs_b)

    if (key_min<key_max):
        f = H5Fopen(h5_name,H5F_ACC_RDONLY,H5P_DEFAULT)
        for h in range(key_min,key_max):
            sprintf(key_t,"[0, 0, 0, %d, %d]",pub_key[h,0]+1,pub_key[h,1]+1)
            data_id = H5Dopen(f,key_t,H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_DOUBLE,H5S_ALL,
                H5S_ALL,H5P_DEFAULT,data_buf
            )
            offset = pub_key[h,5]
            for i in range(pub_key[h,4]):
                hamil_buf[i+offset] = data_buf[i]

            status = H5Dclose(data_id)

        status = H5Fclose(f)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readh5_p1(
    mpi.MPI_Comm comm_b, char* h5_name, int nprocs_b, int myid_b, 
    int ncell2, int[:,::1] key_num, int[:,::1] pub_key, 
    int[:,::1] key_info1, double factor, double* data_buf, 
    double* hamil_buf, double[::1] dhamil, bint LADD
):
    cdef hid_t f, data_id
    cdef herr_t status
    cdef int h, i, j, key_min, key_max, offset
    cdef int ns = key_num[ncell2,3]
    cdef char key_t[100]
    cdef char key_t1[100]

    if (myid_b==0):
        if not LADD:
            memset(hamil_buf,0,sizeof(double)*ns)

    mpi.MPI_Barrier(comm_b)
    key_min = <int>((key_num[ncell2,2]*myid_b)/nprocs_b)
    key_max = <int>((key_num[ncell2,2]*(myid_b+1))/nprocs_b)

    if (key_min<key_max):
        f = H5Fopen(h5_name,H5F_ACC_RDONLY,H5P_DEFAULT)
        for h in range(key_min,key_max):
            sprintf(key_t,"[0, 0, 0, %d, %d]",pub_key[h,0]+1,pub_key[h,1]+1)
            if H5Lexists(f,key_t,H5P_DEFAULT):
                sprintf(key_t1,"[0, 0, 0, %d, %d]",pub_key[h,1]+1,pub_key[h,0]+1)
                if H5Lexists(f,key_t1,H5P_DEFAULT):
                    data_id = H5Dopen(f,key_t,H5P_DEFAULT)
                    status = H5Dread(
                        data_id,H5T_NATIVE_DOUBLE,H5S_ALL,
                        H5S_ALL,H5P_DEFAULT,data_buf
                    )
                    offset = pub_key[h,5]
                    if LADD:
                        for i in range(pub_key[h,4]):
                            dhamil[key_info1[i+offset,1]] \
                            = (data_buf[i]-hamil_buf[i+offset])*factor
                    else:
                        for i in range(pub_key[h,4]):
                            hamil_buf[i+offset] = data_buf[i]
                    status = H5Dclose(data_id)

        status = H5Fclose(f)


@cython.boundscheck(False)
@cython.wraparound(False)
def ReadHamil0(
    MPI.Comm shm_comm_py, int norbital_u, int ncell, int norbital_m, 
    int atomnum, int[:,::1] key_num, int[:,::1] pub_key, 
    int[::1] atom_idx_py, int[::1] atom_idx_all_py, double[::1] hamil_buf,
    char* inDir, char* H5HamName, int Ispin, bint IsH5
):
    cdef int i, j, ierr, shm_nprocs, shm_id
    cdef int ncell2 = ncell*ncell
    cdef mpi.MPI_Comm shm_comm = shm_comm_py.ob_mpi
    cdef char data_name[500]
    global data_buf
    global atom_idx
    global atom_idx_all
    global R_num
    global norb_u

    R_num = ncell
    norb_u = norbital_u
    data_buf = <double*>malloc(norbital_m*norbital_m*sizeof(double))
    atom_idx = <int*>malloc(atomnum*sizeof(int))
    for i in range(atomnum):
        atom_idx[i] = atom_idx_py[i]
    atom_idx_all = <int*>malloc((atomnum+1)*sizeof(int))
    for i in range(atomnum+1):
        atom_idx_all[i] = atom_idx_all_py[i]

    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)

    if IsH5:
        sprintf(data_name,"%s/%s.h5",inDir,H5HamName)
        readh5_p0(
            shm_comm,data_name,shm_nprocs,shm_id,
            ncell2,key_num,pub_key,data_buf,&hamil_buf[0]
        )
    else:
        if (shm_id==0):
            sprintf(data_name,"%s/openmx.scfout",inDir)
            readscfout(data_name,ncell2,key_num,data_buf,&hamil_buf[0],Ispin)

    free(data_buf)
    free(atom_idx)
    free(atom_idx_all)


@cython.boundscheck(False)
@cython.wraparound(False)
def deltahamil_b(
    MPI.Comm comm, MPI.Comm shm_comm, int nmodes, int nm_min,  
    int dH_block, int norbital_u, int ncell, int norbital_m, 
    int atomnum_py, int[::1] atom_idx_py, int[::1] atom_idx_all_py, 
    int[::1] catom, int[:,::1] key_num, int[:,::1] pub_key, 
    int[:,::1] key_info1, double dQ1, double[::1] hamil_buf0, 
    double[:,:,::1] dhamil, char* inDir, char* dhamilDir, 
    char* H5HamName, char* dhamil_method, int Ispin, bint IsH5
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

    cdef double* hamil_buf
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

    data_buf = <double*>malloc(norbital_m*norbital_m*sizeof(double))

    # hamil_buf is only used in FD C method
    # and imodes_min < imodes_max
    if dhamil_method[0]==diff[2] and imodes_min < imodes_max:
        if (myid_b==0):
            l_hamil_buf = (Ispin+1)*key_num[ncell2,3]*sizeof(double)
        else:
            l_hamil_buf = 0
        mpi.MPI_Win_allocate_shared(
            l_hamil_buf,s_d,mpi.MPI_INFO_NULL,comm_b,&hamil_buf,&win
        )
        if (myid_b!=0):
            mpi.MPI_Win_shared_query(win,0,&l_hamil_buf,&s_d,&hamil_buf)
    mpi.MPI_Barrier(c_comm)

    if (dhamil_method[0]!=diff[2]):
        if (dhamil_method[0]==diff[0]):
            factor[0] = dQ1
            h = 0
        else:
            factor[1] = ndQ1
            h = 1
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
                    data_buf,&hamil_buf0[0],dhamil[i,0],1
                )
            else:
                if (myid_b==0):
                    sprintf(
                        data_name,"%s/%s/%d%c%c/openmx.scfout",
                        inDir,dhamilDir,catom[j],plus[h],delta[k]
                    )
                    readscfout1(
                        data_name,ncell2,key_num,pub_key,key_info1,
                        data_buf,&hamil_buf0[0],dhamil[i],factor[h],Ispin,1
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
                    data_buf,hamil_buf,dhamil[i,0],0
                )
            else:
                if (myid_b==0):
                    sprintf(
                        data_name,"%s/%s/%d-%c/openmx.scfout",
                        inDir,dhamilDir,catom[j],delta[k]
                    )
                    readscfout1(
                        data_name,ncell2,key_num,pub_key,key_info1,
                        data_buf,hamil_buf,dhamil[i],dQ2,Ispin,0
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
                    data_buf,hamil_buf,dhamil[i,0],1
                )
            else:
                if (myid_b==0):
                    sprintf(
                        data_name,"%s/%s/%d+%c/openmx.scfout",
                        inDir,dhamilDir,catom[j],delta[k]
                    )
                    readscfout1(
                        data_name,ncell2,key_num,pub_key,key_info1,
                        data_buf,hamil_buf,dhamil[i],dQ2,Ispin,1
                    )
            mpi.MPI_Barrier(comm_b)

    mpi.MPI_Barrier(c_shm_comm)
    if (nnodes>1):
        mpi.MPI_Type_contiguous(key_num[ncell2,3],mpi.MPI_DOUBLE,&DB_NOB)
        mpi.MPI_Type_commit(&DB_NOB)
        if (comm_gp!=mpi.MPI_COMM_NULL):
#            imodes_num_gather = <int*>malloc(sizeof(int)*nnodes)
#            imodes_gather = <int*>malloc(sizeof(int)*(nnodes+1))
#            for i in range(nnodes):
#                imodes_num_gather[i] = imodes_num_shm[i]*ncell2*(Ispin+1)
#            for i in range(nnodes+1):
#                imodes_gather[i] = imodes_shm[i]*ncell2*(Ispin+1)
#            mpi.MPI_Allgatherv(
#                mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,&dhamil[0,0,0],
#                imodes_num_gather,imodes_gather,DB_NOB,comm_gp
#            )
#            free(imodes_num_gather)
#            free(imodes_gather)

            for i in range(nnodes):
                for j in range(imodes_shm[i],imodes_shm[i+1]):
                    for k in range(Ispin+1):
                        mpi.MPI_Bcast(&dhamil[j,k,0],1,DB_NOB,i,comm_gp)

        mpi.MPI_Type_free(&DB_NOB)

    if dhamil_method[0]==diff[2] and imodes_min < imodes_max:
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

    mpi.MPI_Comm_free(&comm_b)
    if comm_gp != mpi.MPI_COMM_NULL:
        mpi.MPI_Comm_free(&comm_gp)
    mpi.MPI_Group_free(&group)
    mpi.MPI_Group_free(&group_all)

    mpi.MPI_Barrier(c_comm)
    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("dhamil time in mode[%4d:%4d]:%12.4fs.\n",nm_min,nm_min+nmodes,endtime-starttime)
