#cython: language_level=3
#cython: cdivision=True

cimport cython
from mpi4py cimport MPI
from mpi4py cimport libmpi as mpi
from libc.math cimport cos, sin, sqrt, exp, fabs, fmod, fmax, round, M_PI
from libc.stdio cimport printf, sscanf, sprintf, FILE, \
     SEEK_SET, SEEK_CUR, SEEK_END, fopen, fseek, fread, fwrite, fclose
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, calloc, free, qsort
from libc.time cimport time, time_t


cdef extern from "hdf5.h":
    ctypedef long hid_t
    ctypedef int herr_t
    cdef int H5T_NATIVE_INT
    cdef int H5T_NATIVE_DOUBLE
    cdef hid_t H5S_ALL
    cdef unsigned int H5F_ACC_RDONLY
    cdef unsigned int H5P_DEFAULT
    cdef hid_t H5Fopen(
        char* filename, unsigned int flags, hid_t access_plist
    )
    cdef hid_t H5Dopen(
        hid_t file_id, const char* name, hid_t dapl_id
    )
    cdef herr_t H5Dread(
        hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
        hid_t file_space_id, hid_t plist_id, void* buf
    )
    cdef herr_t H5Dclose(hid_t dset_id)
    cdef herr_t H5Fclose(hid_t file_id)

    ctypedef signed long long haddr_t
    ctypedef enum H5O_type_t:
        H5O_TYPE_UNKNOWN = -1,
        H5O_TYPE_GROUP,
        H5O_TYPE_DATASET,
        H5O_TYPE_NAMED_DATATYPE,
        H5O_TYPE_NTYPES
    ctypedef long long hsize_t
    ctypedef struct space:
        hsize_t total
        hsize_t meta
        hsize_t mesg
        hsize_t free
    ctypedef struct mesg:
        unsigned long present
        unsigned long shared
    ctypedef struct hdr:
        unsigned version
        unsigned nmesgs
        unsigned nchunks
        unsigned flags
        space space
        mesg mesg
    ctypedef struct H5_ih_info_t:
        hsize_t     index_size,
        hsize_t     heap_size
    cdef struct meta_size:
        H5_ih_info_t   obj,
        H5_ih_info_t   attr
    ctypedef struct H5O_info_t:
        unsigned long   fileno
        haddr_t         addr
        H5O_type_t      type
        unsigned        rc
        time_t          atime
        time_t          mtime
        time_t          ctime
        time_t          btime
        hsize_t         num_attrs
        hdr             hdr
        meta_size       meta_size

    ctypedef enum H5_index_t:
        H5_INDEX_UNKNOWN = -1,
        H5_INDEX_NAME,
        H5_INDEX_CRT_ORDER,
        H5_INDEX_N
    ctypedef enum H5_iter_order_t:
        H5_ITER_UNKNOWN = -1,
        H5_ITER_INC,
        H5_ITER_DEC,
        H5_ITER_NATIVE,
        H5_ITER_N
    ctypedef herr_t (*H5O_iterate_t)(
        hid_t obj, char* name, H5O_info_t* info, void* op_data
    ) except *
    cdef herr_t H5Ovisit(
        hid_t obj_id, H5_index_t idx_type,
        H5_iter_order_t order, H5O_iterate_t op, void* op_data
    )
    ctypedef int htri_t
    cdef htri_t H5Lexists(hid_t loc_id, char* name, hid_t lapl_id)


cdef extern from "complex.h":
    double complex conj(double complex)
    double creal(double complex)
    double cimag(double complex)
    double complex cexp(double complex)
    double complex ccos(double complex)
    double complex csin(double complex)
    double cabs(double complex)

cdef extern from "mkl_types.h" nogil:
    ctypedef int MKL_INT


cdef extern from "mkl_pblas.h" nogil:
    cdef void pdgemm(
        char* transa, char* transb, MKL_INT* m, MKL_INT* n, MKL_INT* k,
        double* alpha, double *a, MKL_INT *ia, MKL_INT *ja, MKL_INT* desca,
        double* b, MKL_INT* ib, MKL_INT* jb, MKL_INT* descb, double* beta,
        double* c, MKL_INT* ic, MKL_INT* jc, MKL_INT* descc
    )
    cdef void pdsymm(
        char* side, char* uplo, MKL_INT* m, MKL_INT* n, double* alpha,
        double* a, MKL_INT* ia, MKL_INT* ja, MKL_INT* desca,
        double* b, MKL_INT* ib, MKL_INT* jb, MKL_INT* descb,
        double* beta, double* c, MKL_INT* ic, MKL_INT* jc,
        MKL_INT* descc
    )
    cdef void pdgeadd(
        char* trans, MKL_INT* m, MKL_INT* n, double* alpha,
        double* a, MKL_INT* ia, MKL_INT* ja, MKL_INT* desca, double* beta,
        double* c, MKL_INT* ic, MKL_INT* jc, MKL_INT* descc
    )
    cdef void pdtran(
        MKL_INT *m, MKL_INT *n, double *alpha, double *a, 
        MKL_INT *ia, MKL_INT *ja, MKL_INT *desca, double *beta, 
        double *c, MKL_INT *ic, MKL_INT *jc, MKL_INT *descc
    )


cdef extern from "mkl_scalapack.h" nogil:
    cdef void pdgesv(
        MKL_INT* n, MKL_INT* nrhs, double* a, MKL_INT* ia, MKL_INT* ja,
        MKL_INT* desca, MKL_INT* ipiv, double* b, MKL_INT* ib,
        MKL_INT* jb, MKL_INT* descb, MKL_INT* info
    )
    cdef MKL_INT numroc(
        MKL_INT* n, MKL_INT* nb, MKL_INT* iproc,
        MKL_INT* isrcproc, MKL_INT* nprocs
    )
    cdef void descinit(
        MKL_INT* desc, MKL_INT* m, MKL_INT* n, MKL_INT* mb, MKL_INT* nb,
        MKL_INT* irsrc, MKL_INT* icsrc, MKL_INT* ictxt, MKL_INT* lld, MKL_INT* info
    )
    cdef void pdgemr2d(
        MKL_INT* m, MKL_INT* n, double* a, MKL_INT* ia, MKL_INT* ja, MKL_INT* desca,
        double* b, MKL_INT* ib, MKL_INT* jb, MKL_INT* descb, MKL_INT* ictxt
    )
    cdef void pdposv(
        char* uplo, MKL_INT* n, MKL_INT* nrhs, double* a, MKL_INT* ia,
        MKL_INT* ja, MKL_INT* desca, double* b, MKL_INT* ib, MKL_INT* jb, 
        MKL_INT* descb, MKL_INT* info
    )
    cdef void pdpotrf(
        char* uplo, MKL_INT* n, double* a, MKL_INT* ia, 
        MKL_INT* ja, MKL_INT* desca, MKL_INT* info
    )


cdef extern from "mkl_blacs.h" nogil:
    cdef void blacs_pinfo(MKL_INT* mypnum, MKL_INT* nprocs)
    cdef void blacs_get(MKL_INT* ConTxt, MKL_INT* what, MKL_INT* val)
    cdef void blacs_gridinit(MKL_INT* ConTxt, char* layout, MKL_INT* nprow, MKL_INT* npcol)
    cdef void blacs_gridinfo(
        MKL_INT* ConTxt, MKL_INT* nprow, MKL_INT* npcol,
        MKL_INT* myrow, MKL_INT* mycol
    )
    cdef void blacs_gridexit(MKL_INT*)


cdef extern from "mkl_spblas.h" nogil:
    cdef enum sparse_status_t:
        SPARSE_STATUS_SUCCESS           = 0
        # the operation was successful
        SPARSE_STATUS_NOT_INITIALIZED   = 1
        # empty handle or matrix arrays
        SPARSE_STATUS_ALLOC_FAILED      = 2
        # internal error: memory allocation failed
        SPARSE_STATUS_INVALID_VALUE     = 3
        # invalid input value
        SPARSE_STATUS_EXECUTION_FAILED  = 4
        # e.g. 0-diagonal element for triangular solver, etc.
        SPARSE_STATUS_INTERNAL_ERROR    = 5
        # internal error
        SPARSE_STATUS_NOT_SUPPORTED     = 6
        # e.g. operation for double precision doesn't support other types

    cdef struct sparse_matrix
    ctypedef sparse_matrix* sparse_matrix_t
    cdef enum sparse_index_base_t:
        SPARSE_INDEX_BASE_ZERO  = 0 # C-style
        SPARSE_INDEX_BASE_ONE   = 1 # Fortran-style

    cdef sparse_status_t mkl_sparse_d_create_coo(
        sparse_matrix_t *A, sparse_index_base_t indexing,
        MKL_INT rows, MKL_INT cols, MKL_INT nnz,
        MKL_INT *row_indx, MKL_INT *col_indx, double *values
    )
    cdef sparse_status_t mkl_sparse_d_create_csr(
        sparse_matrix_t *A, sparse_index_base_t indexing, 
        MKL_INT rows, MKL_INT cols, MKL_INT *rows_start, 
        MKL_INT *rows_end, MKL_INT *col_indx, double *values
    )
    cdef sparse_status_t mkl_sparse_optimize(sparse_matrix_t A)
    cdef sparse_status_t mkl_sparse_destroy(sparse_matrix_t  A)

    cdef enum sparse_operation_t:
        SPARSE_OPERATION_NON_TRANSPOSE       = 10
        SPARSE_OPERATION_TRANSPOSE           = 11
        SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12
    cdef enum sparse_matrix_type_t:
        SPARSE_MATRIX_TYPE_GENERAL           = 20
        #General case
        SPARSE_MATRIX_TYPE_SYMMETRIC         = 21
        #Triangular part of
        SPARSE_MATRIX_TYPE_HERMITIAN         = 22
        #the matrix is to be processed
        SPARSE_MATRIX_TYPE_TRIANGULAR        = 23
        SPARSE_MATRIX_TYPE_DIAGONAL          = 24
        #diagonal matrix; only diagonal elements will be processed
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR  = 25
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL    = 26
        #block-diagonal matrix; only diagonal blocks will be processed
    cdef enum sparse_fill_mode_t:
        SPARSE_FILL_MODE_LOWER  = 40
        # lower triangular part of the matrix is stored
        SPARSE_FILL_MODE_UPPER  = 41
        # upper triangular part of the matrix is stored
        SPARSE_FILL_MODE_FULL   = 42
        # upper triangular part of the matrix is stored
    cdef enum sparse_diag_type_t:
        SPARSE_DIAG_NON_UNIT    = 50
        #triangular matrix with non-unit diagonal
        SPARSE_DIAG_UNIT        = 51
        #triangular matrix with unit diagonal
    cdef struct matrix_descr:
        sparse_matrix_type_t  type
        #matrix type: general, diagonal or triangular / symmetric / hermitian
        sparse_fill_mode_t    mode
        #upper or lower triangular part of the matrix ( for triangular / symmetric / hermitian case)
        sparse_diag_type_t    diag
        # unit or non-unit diagonal ( for triangular / symmetric / hermitian case)

    cdef sparse_status_t mkl_sparse_d_mv(
        sparse_operation_t operation, double alpha, sparse_matrix_t A,
        matrix_descr descr, double *x, double beta, double *y
    )
    cdef enum sparse_layout_t:
        SPARSE_LAYOUT_ROW_MAJOR    = 101 #C-style
        SPARSE_LAYOUT_COLUMN_MAJOR = 102 #Fortran-style
    cdef sparse_status_t mkl_sparse_d_mm(
        sparse_operation_t operation, double alpha, sparse_matrix_t A,
        matrix_descr descr,
        # sparse_matrix_type_t + sparse_fill_mode_t + sparse_diag_type_t
        sparse_layout_t layout,
        #storage scheme for the dense matrix: C-style or Fortran-style
        double *x, MKL_INT columns, MKL_INT ldx, double beta,
        double *y, MKL_INT ldy
    )


cdef long norb
cdef int* atom_idx
cdef int* atom_idx_sum
cdef int key_num_p
cdef double Hartree2eV = 27.211386245988
cdef double Bohr2Ang = 0.529177249


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int Cmp(void * pa, void * pb) nogil:
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
cdef herr_t readh5engine_key0(
    hid_t loc_id, char* name, H5O_info_t* info, void* key_num
):
    cdef int idx[5]
    cdef herr_t status
    cdef hid_t data_id

    sscanf(name,"[%d, %d, %d, %d, %d]",\
           &idx[0],&idx[1],&idx[2],&idx[3],&idx[4])
    if (idx[0]==0 and idx[1]==0 and idx[2]==0):
        (<int*>(key_num))[0] += 1
        (<int*>(key_num))[1] += atom_idx[idx[3]-1]*atom_idx[idx[4]-1]
        return 0
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readh5_key0(char* h5_name, int nfile, int[:,::1] key_num):
    cdef int i, j
    cdef hid_t f
    cdef herr_t status
    cdef char h5name[500]

    for i in range(nfile+1):
        for j in range(4):
            key_num[i,j] = 0

    if nfile>1:
        for i in range(nfile):
            sprintf(h5name,"%s_%d.h5",h5_name,i)
            f = H5Fopen(h5name,H5F_ACC_RDONLY,H5P_DEFAULT)
            status = H5Ovisit(
                f,H5_INDEX_NAME,H5_ITER_NATIVE,
                readh5engine_key0,&key_num[i,0]
            )
            status = H5Fclose(f)
    else:
        sprintf(h5name,"%s.h5",h5_name)
        f = H5Fopen(h5name,H5F_ACC_RDONLY,H5P_DEFAULT)
        status = H5Ovisit(
            f,H5_INDEX_NAME,H5_ITER_NATIVE,
            readh5engine_key0,&key_num[0,0]
        )
        status = H5Fclose(f)

    for i in range(nfile):
        for j in range(i+1,nfile+1):
            key_num[j,2] += key_num[i,0]
            key_num[j,3] += key_num[i,1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readscfout_key0(char* name, int[:,::1] key_num):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j ,k, ct_AN, h_AN, Gh_AN, \
             iR, jR, Rij, atomnum, TNO1, TNO2
    cdef int* atv_ijk
    cdef int* FNAN
    cdef int** natn
    cdef int** ncn

    key_num[0,0] = 0
    key_num[0,1] = 0
    key_num[0,2] = 0
    key_num[0,3] = 0

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
        TNO1 = atom_idx[ct_AN-1]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            TNO2 = atom_idx[Gh_AN-1]
            if (ncn[ct_AN][h_AN]==0):
                key_num[0,0] += 1
                key_num[0,1] += TNO1*TNO2
    #        fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    fclose(fp)
    for ct_AN in range(1,atomnum+1):
        free(natn[ct_AN])
        free(ncn[ct_AN])
    free(natn)
    free(ncn)
    free(FNAN)

    key_num[1,2] = key_num[0,0]
    key_num[1,3] = key_num[0,1]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy_key(int[:,::1] key_num_src, int[:,::1] key_num_dst):
    key_num_dst[0,0] = key_num_src[0,0]
    key_num_dst[0,1] = key_num_src[0,1]
    key_num_dst[0,2] = 0
    key_num_dst[0,3] = 0
    key_num_dst[1,2] = key_num_dst[0,0]
    key_num_dst[1,3] = key_num_dst[0,1]


@cython.boundscheck(False)
@cython.wraparound(False)
def GetSparseNum(
    char* inDir, char* H5HamName, char* H5OlpName, char* H5DrName, 
    int nfileham, int nfileolp, int nfiledr, int atomnum,
    int[:,::1] key_num_h, int[:,::1] key_num_o, int[:,::1] key_num_dr, 
    int[::1] atom_idx_py, bint IsH5
):
    cdef int i, j
    cdef char data_name[500]
    global atom_idx

    atom_idx = <int*>malloc(atomnum*sizeof(int))
    for i in range(atomnum):
        atom_idx[i] = atom_idx_py[i]
    if IsH5:
        readh5_key0(H5HamName,nfileham,key_num_h)
        readh5_key0(H5OlpName,nfileolp,key_num_o)
        sprintf(data_name,"%sx",H5DrName)
        readh5_key0(data_name,nfiledr,key_num_dr)
    else:
        sprintf(data_name,"%s/openmx.scfout",inDir)
        readscfout_key0(data_name,key_num_h)
        copy_key(key_num_h,key_num_o)
        copy_key(key_num_h,key_num_dr)

    free(atom_idx)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void GetKeyInfo(
    int nfile, int[:,::1] key_num, 
    int[:,::1] pub_key, long[:,::1] key_info
):
    cdef int h, i, j, k, offset
    cdef long* key_info_t

    key_info_t = <long*>malloc(key_num[nfile,3]*2*sizeof(long))

    offset = 0
    for h in range(key_num[nfile,2]):
        for i in range(pub_key[h,2]):
            for j in range(pub_key[h,3]):
                k = i*pub_key[h,3]+j
                key_info_t[(k+offset)*2] \
                = (atom_idx_sum[pub_key[h,0]]+i)*norb \
                + (atom_idx_sum[pub_key[h,1]]+j)
        offset += pub_key[h,2]*pub_key[h,3]

    for i in range(key_num[nfile,3]):
        key_info_t[i*2+1] = i
        key_info[i,1] = i
    qsort(key_info_t,key_num[nfile,3],sizeof(long)*2,&Cmp)
    for i in range(key_num[nfile,3]):
        key_info[i,0] = key_info_t[i*2+1]
    qsort(&key_info[0,0],key_num[nfile,3],sizeof(long)*2,&Cmp)
    for i in range(key_num[nfile,3]):
        key_info[i,0] = key_info_t[i*2]

    free(key_info_t)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef herr_t readh5engine_key1(
    hid_t loc_id, char* name, H5O_info_t* info, void* pub_key
):
    cdef int idx[5]
    cdef herr_t status
    cdef hid_t data_id
    cdef int i, j, atomi, atomj
    global key_num_p

    sscanf(name,"[%d, %d, %d, %d, %d]",\
           &idx[0],&idx[1],&idx[2],&idx[3],&idx[4])
    if (idx[0]==0 and idx[1]==0 and idx[2]==0):
        atomi = idx[3]-1
        atomj = idx[4]-1
        (<int*>(pub_key))[key_num_p*4] = atomi
        (<int*>(pub_key))[key_num_p*4+1] = atomj
        (<int*>(pub_key))[key_num_p*4+2] = atom_idx[atomi]
        (<int*>(pub_key))[key_num_p*4+3] = atom_idx[atomj]
        key_num_p += 1
        return 0
    else:
        return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readh5_key1(
    char* h5_name, int nfile, int[:,::1] key_num, 
    int[:,::1] pub_key, long[:,::1] key_info
):
    cdef int h, i, j, offset
    cdef hid_t f
    cdef herr_t status
    cdef char h5name[500]
    global key_num_p

    if nfile>1:
        for i in range(nfile):
            key_num_p = key_num[i,2]
            sprintf(h5name,"%s_%d.h5",h5_name,i)
            f = H5Fopen(h5name,H5F_ACC_RDONLY,H5P_DEFAULT)
            status = H5Ovisit(
                f,H5_INDEX_NAME,H5_ITER_NATIVE,
                readh5engine_key1,&pub_key[0,0]
            )
            status = H5Fclose(f)
    else:
        key_num_p = 0
        sprintf(h5name,"%s.h5",h5_name)
        f = H5Fopen(h5name,H5F_ACC_RDONLY,H5P_DEFAULT)
        status = H5Ovisit(
            f,H5_INDEX_NAME,H5_ITER_NATIVE,
            readh5engine_key1,&pub_key[0,0]
        )
        status = H5Fclose(f)

    GetKeyInfo(nfile,key_num,pub_key,key_info)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readscfout_key1(
    char* name, int nfile, int[:,::1] key_num, 
    int[:,::1] pub_key, long[:,::1] key_info
):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j ,k, ct_AN, h_AN, Gh_AN, \
             atomi, atomj, atomnum, TNO1, TNO2
    cdef int* atv_ijk
    cdef int* FNAN
    cdef int** natn
    cdef int** ncn
    global key_num_p

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

    key_num_p = 0
    for ct_AN in range(1,atomnum+1):
        atomi = ct_AN-1
        TNO1 = atom_idx[atomi]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            atomj = Gh_AN-1
            TNO2 = atom_idx[atomj]
            if (ncn[ct_AN][h_AN]==0):
                pub_key[key_num_p,0] = atomi
                pub_key[key_num_p,1] = atomj
                pub_key[key_num_p,2] = TNO1
                pub_key[key_num_p,3] = TNO2
                key_num_p += 1
    #        fseek(fp,TNO1*TNO2*8,SEEK_CUR)

    fclose(fp)
    for ct_AN in range(1,atomnum+1):
        free(natn[ct_AN])
        free(ncn[ct_AN])
    free(natn)
    free(ncn)
    free(FNAN)

    GetKeyInfo(nfile,key_num,pub_key,key_info)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy_key1(
    int nfile, int[:,::1] key_num,
    int[:,::1] pub_key_src, int[:,::1] pub_key_dst, 
    long[:,::1] key_info_src, long[:,::1] key_info_dst
):
    memcpy(
        &pub_key_dst[0,0],&pub_key_src[0,0],
        key_num[nfile,2]*4*sizeof(int)
    )
    memcpy(
        &key_info_dst[0,0],&key_info_src[0,0],
        key_num[nfile,3]*2*sizeof(long)
    )


@cython.boundscheck(False)
@cython.wraparound(False)
def GetSparseIdx(
    char* inDir, char* H5HamName, char* H5OlpName, char* H5DrName,
    int nfileham, int nfileolp, int nfiledr, int atomnum, int norbital, 
    int[:,::1] key_num_h, int[:,::1] key_num_o, int[:,::1] key_num_dr,
    int[:,::1] pub_key_h, int[:,::1] pub_key_o, int[:,::1] pub_key_dr,
    long[:,::1] keyinfo_h, long[:,::1] keyinfo_o, long[:,::1] keyinfo_dr,
    int[::1] atom_idx_py, int[::1] atom_idx_sum_py, bint IsH5
):
    cdef int h, i, j, offset
    cdef char data_name[500]
    global atom_idx
    global atom_idx_sum
    global norb

    norb = norbital
    atom_idx = <int*>malloc(atomnum*sizeof(int))
    for i in range(atomnum):
        atom_idx[i] = atom_idx_py[i]
    atom_idx_sum = <int*>malloc((atomnum+1)*sizeof(int))
    for i in range(atomnum+1):
        atom_idx_sum[i] = atom_idx_sum_py[i]

    if IsH5:
        readh5_key1(H5HamName,nfileham,key_num_h,pub_key_h,keyinfo_h)
        readh5_key1(H5OlpName,nfileolp,key_num_o,pub_key_o,keyinfo_o)
        sprintf(data_name,"%sx",H5DrName)
        readh5_key1(data_name,nfiledr,key_num_dr,pub_key_dr,keyinfo_dr)
    else:
        sprintf(data_name,"%s/openmx.scfout",inDir)
        readscfout_key1(data_name,nfileham,key_num_h,pub_key_h,keyinfo_h)
        copy_key1(1,key_num_h,pub_key_h,pub_key_o,keyinfo_h,keyinfo_o)
        copy_key1(1,key_num_h,pub_key_h,pub_key_dr,keyinfo_h,keyinfo_dr)

    free(atom_idx)
    free(atom_idx_sum)


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
            offset = 0
            for j in range(key_min):
                offset += pub_key[j,2]*pub_key[j,3]
            sprintf(h5name,"%s_%d.h5",h5_name,i)
            f = H5Fopen(h5name,H5F_ACC_RDONLY,H5P_DEFAULT)
            for j in range(key_min,key_max):
                sprintf(key_t,"[0, 0, 0, %d, %d]",pub_key[j,0]+1,pub_key[j,1]+1)
                data_id = H5Dopen(f,key_t,H5P_DEFAULT)
                status = H5Dread(
                    data_id,H5T_NATIVE_DOUBLE,H5S_ALL,
                    H5S_ALL,H5P_DEFAULT,data_buf
                )
                for k in range(pub_key[j,2]*pub_key[j,3]):
                    data[key_info[k+offset,1]] = data_buf[k]*factor
                status = H5Dclose(data_id)
                offset += pub_key[j,2]*pub_key[j,3]
            status = H5Fclose(f)
    else:
        mpi.MPI_Barrier(shm_comm)
        key_min = (key_num[0,0]*shm_id)/shm_nprocs
        key_max = (key_num[0,0]*(shm_id+1))/shm_nprocs
        sprintf(h5name,"%s.h5",h5_name)
        f = H5Fopen(h5name,H5F_ACC_RDONLY,H5P_DEFAULT)
        offset = 0
        for j in range(key_min):
            offset += pub_key[j,2]*pub_key[j,3]
        for j in range(key_min,key_max):
            sprintf(key_t,"[0, 0, 0, %d, %d]",pub_key[j,0]+1,pub_key[j,1]+1)
            data_id = H5Dopen(f,key_t,H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_DOUBLE,H5S_ALL,
                H5S_ALL,H5P_DEFAULT,data_buf
            )
            for k in range(pub_key[j,2]*pub_key[j,3]):
                data[key_info[k+offset,1]] = data_buf[k]*factor
            status = H5Dclose(data_id)
            offset += pub_key[j,2]*pub_key[j,3]
        status = H5Fclose(f)

    free(data_buf)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void readscfout(
    char* name, int norb_m, long[:,::1] key_info, 
    double[::1] data_h, double[::1] data_o, double[:,::1] data_dr, 
    double factor_h, double factor_o, double factor_dr
):
    cdef FILE * fp
    cdef int i_vec[6]
    cdef int TCpyCell
    cdef int i, j ,k, xyz, ct_AN, h_AN, Gh_AN, \
             atomnum, factor, offset, TNO1, TNO2
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

    offset = 0
    for ct_AN in range(1,atomnum+1):
        TNO1 = Total_NumOrbs[ct_AN]
        for h_AN in range(FNAN[ct_AN]+1):
            Gh_AN = natn[ct_AN][h_AN]
            TNO2 = Total_NumOrbs[Gh_AN]
            if (ncn[ct_AN][h_AN]==0):
                fread(data_buf,sizeof(double),TNO1*TNO2,fp)
                for i in range(TNO1*TNO2):
                    data_h[key_info[i+offset,1]] \
                    = data_buf[i]*factor_h
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


@cython.boundscheck(False)
@cython.wraparound(False)
def GetSparseData(
    MPI.Comm shm_comm_py, 
    char* inDir, char* H5HamName, char* H5OlpName, char* H5DrName,
    int nfileham, int nfileolp, int nfiledr, int atomnum, int norb_m, 
    int[:,::1] key_num_h, int[:,::1] key_num_o, int[:,::1] key_num_dr,
    int[:,::1] pub_key_h, int[:,::1] pub_key_o, int[:,::1] pub_key_dr,
    long[:,::1] keyinfo_h, long[:,::1] keyinfo_o, long[:,::1] keyinfo_dr,
    double[::1] data_h, double[::1] data_o, double[:,::1] data_dr, bint IsH5
):
    cdef int shm_nprocs, shm_id, ierr
    cdef char data_name[500]
    cdef double f_dr = 1.0/Bohr2Ang

    cdef mpi.MPI_Comm shm_comm = shm_comm_py.ob_mpi
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)

    if IsH5:
        readh5(
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
                data_name,norb_m,keyinfo_h,data_h,
                data_o,data_dr,Hartree2eV,1.0,f_dr
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
def coo2csridx(
    int N, int Nsparse, long[:,::1] coo_idx,
    int[:,::1] csr_ridx, int[::1] csr_cidx
):
    cdef int i, idx, ridx
    cdef int s_int = sizeof(int)
    cdef int * coo_ridx = <int*>malloc(s_int*Nsparse)
    cdef int * coo_cidx = <int*>malloc(s_int*Nsparse)

    for i in range(Nsparse):
        coo_ridx[i] = <int>(coo_idx[i,0]/N)
        coo_cidx[i] = <int>(coo_idx[i,0]%N)

    idx = -1
    for i in range(Nsparse):
        ridx = coo_ridx[i]
        if idx<ridx:
            csr_ridx[0,ridx] = i
            idx = ridx
        csr_cidx[i] = coo_cidx[i]
    for i in range(N-1):
        csr_ridx[1,i] = csr_ridx[0,i+1]
    csr_ridx[1,N-1] = Nsparse

    free(coo_ridx)
    free(coo_cidx)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sparse2dense_csr(
    double[::1] olp, int[:,::1] csr_ridx, int[::1] csr_cidx, 
    int N, int Nsparse, int Nproc_min, int Nproc_max, 
    int Nproc_len, double factor, double* olp_f
):
    cdef int i
    cdef int irow, icol

    memset(olp_f,0,sizeof(double)*N*Nproc_len)
    for i in range(Nsparse):
        icol = csr_cidx[i]
        if icol<Nproc_min or icol>=Nproc_max:
            continue
        else:
            for j in range(N):
                if(i>=csr_ridx[0,j] and i<csr_ridx[1,j]):
                    irow = j
                    break
            olp_f[irow*Nproc_len+icol-Nproc_min] = olp[i]*factor


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double sparse1norm(
    double[::1] olp, int[:,::1] keyinfo, int Nsparse
):
    cdef int i
    cdef int irow, icol, colidx
    cdef double coldata, colsum, colsum_max

    irow = -1
    colsum = 0.0
    colsum_max = 0.0
    for i in range(Nsparse):
        colidx = keyinfo[0,i]
        coldata = fabs(olp[i])
        if irow<colidx:
            if colsum>colsum_max:
                colsum_max = colsum
            colsum = coldata
            irow = colidx
        else:
            colsum += coldata

    return colsum_max


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
    mpi.MPI_Comm comm, double* olp, 
    int nprocs, int myid, int N, int Nproc_len, int Nsplit, 
    int* Nproc, int* Nproc_num, double[:,::1] olp_r
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
def olp_inv(
    MPI.Comm comm, int nprocs, int myid, 
    double[::1] olp, int[:,::1] olp_keyinfo, int Nsparse_o, 
    double[::1] ham, int[:,::1] ham_keyinfo, int Nsparse_h,
    double[:,::1] dr, int[:,::1] dr_csr_ridx, int[::1] dr_csr_cidx, 
    int Nsparse_dr, int drp_min, int N, int Nsplit, 
    MKL_INT Mb, MKL_INT Nb, double[:,:,::1] drSH
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
    cdef double lv, starttime, endtime
    cdef matrix_descr descrdr
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
    descinit(descSp,&N_mkl,&N_mkl,&Nproc_mkl,&N_mkl,&i0,&i0,&ictxt1,&lldSp,&info)
    olpbuf_pb = <double*>malloc(s_d*mb*nb)
    hambuf_pb = <double*>malloc(s_d*mb*nb)

    # init ham_pb,olp_pb
    sparse2dense_coo(
        olp,olp_keyinfo,N,Nsparse_o,Nproc_min,
        Nproc_max,Nproc_len,1.0,matbuf
    )
    pdgemr2d(
        &N_mkl,&N_mkl,matbuf,&i1,&i1,descSp,
        olpbuf_pb,&i1,&i1,descS,&ictxt
    )
    sparse2dense_coo(
        ham,ham_keyinfo,N,Nsparse_h,Nproc_min,
        Nproc_max,Nproc_len,1.0,matbuf
    )
    pdgemr2d(
        &N_mkl,&N_mkl,matbuf,&i1,&i1,descSp,
        hambuf_pb,&i1,&i1,descS,&ictxt
    )
    # calculate S^-1*H
    pdposv(
        'U',&N_mkl,&N_mkl,olpbuf_pb,&i1,&i1,
        descS,hambuf_pb,&i1,&i1,descS,&info
    )
    free(olpbuf_pb)
    olpinvham = <double*>malloc(s_d*N*Nproc_len)
    pdgemr2d(
        &N_mkl,&N_mkl,hambuf_pb,&i1,&i1,descS,
        olpinvham,&i1,&i1,descSp,&ictxt1
    )
    free(hambuf_pb)
    # transpose S^-1*H from fortran to C order
    mat_tran(
        c_comm,nprocs,N,Nproc_len,Nproc,
        Nproc_num,olpinvham,matbuf
    )
#    cdef FILE * fp
#    if myid == 0:
#        fp = fopen("olpinvham.dat","wb")
#        fwrite(olpinvham,s_d,N*Nproc_len,fp)
#        fclose(fp)
    descrdr.type = SPARSE_MATRIX_TYPE_GENERAL
    for i in range(3):
        mkl_sparse_d_create_csr(
            &csrdr,SPARSE_INDEX_BASE_ZERO,Np_mkl,N_mkl,
            &dr_csr_ridx[0,0],&dr_csr_ridx[1,0],
            &dr_csr_cidx[drp_min],&dr[i,drp_min]
        )
        mkl_sparse_optimize(csrdr)
        # dot(dr,S^-1*H) [Np,N]*[N,Nproc_len]
        mkl_sparse_d_mm(
            SPARSE_OPERATION_NON_TRANSPOSE,1.0,csrdr,
            descrdr,SPARSE_LAYOUT_ROW_MAJOR,olpinvham,
            lldSp,lldSp,0.0,matbuf,lldSp
        )
        # reshape dr*S^-1*H [Np,N]->[nR,(Np)_p,Np]
        mat_reshape(
            c_comm,matbuf,nprocs,myid,N,Nproc_len,
            Nsplit,Nproc,Nproc_num,drSH[i]
        )
        mkl_sparse_destroy(csrdr)

    free(Nproc)
    free(Nproc_num)
    free(olpinvham)
    free(matbuf)

    blacs_gridexit(&ictxt)
    blacs_gridexit(&ictxt1)

    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("olpinv time: %.5fs.\n",endtime-starttime)


@cython.boundscheck(False)
@cython.wraparound(False)
def drSH2dhamil(
    MPI.Comm comm, int nprocs, int myid, int natom,
    int N, int Nsplit, int[::1] norb_u, int[::1] norb_u_num,
    double[:,:,::1] drSH, double[:,:,:,:,::1] dhamil
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int i, j, k, l, m, n, xyz, Np_num
    cdef int s_int = sizeof(int)
    cdef int s_d = sizeof(double)
    cdef int Np = N/Nsplit
    cdef int icell = Nsplit/2
    cdef double * dh_buf
    cdef int * count = <int*>malloc(nprocs*s_int)
    cdef int * displ = <int*>malloc(nprocs*s_int)
    cdef int * Npproc = <int*>calloc(s_int,nprocs+1)
    cdef int * Npproc_num = <int*>calloc(s_int,nprocs)
    cdef double dh, starttime, endtime

    starttime = mpi.MPI_Wtime()
    for i in range(nprocs):
        Np_min = (Np*i)/nprocs
        Np_max = (Np*(i+1))/nprocs
        Npproc_num[i] = Np_max-Np_min
        Npproc[i+1] = Np_max
    Np_num = Npproc_num[myid]

    for i in range(nprocs):
        count[i] = Np*Npproc_num[i]
        displ[i] = Np*Npproc[i]

    if (myid==0):
        dh_buf = <double*>malloc(s_d*Np*Np)
    mpi.MPI_Barrier(c_comm)

    for xyz in range(3):
        for i in range(Nsplit):
            mpi.MPI_Gatherv(
                &drSH[xyz,i,0],count[myid],mpi.MPI_DOUBLE,
                dh_buf,count,displ,mpi.MPI_DOUBLE,0,c_comm
            )
            if (myid==0):
                for k in range(natom):
                    for l in range(norb_u_num[k]):
                        n = l+norb_u[k]
                        for m in range(Np):
                            dh = dh_buf[n*Np+m]
                            dhamil[k*3+xyz,icell,i,n,m] += dh
                            dhamil[k*3+xyz,i,icell,m,n] += dh

    free(Npproc)
    free(Npproc_num)
    free(count)
    free(displ)
    if (myid==0):
        free(dh_buf)
