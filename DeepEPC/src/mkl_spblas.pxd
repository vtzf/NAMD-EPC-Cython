cdef extern from "mkl_types.h":
    ctypedef int MKL_INT
    cdef struct _MKL_Complex16:
        double real
        double imag
    ctypedef _MKL_Complex16 MKL_Complex16


cdef extern from "mkl_spblas.h" nogil:
    cdef enum sparse_status_t:
        SPARSE_STATUS_SUCCESS           = 0
        SPARSE_STATUS_NOT_INITIALIZED   = 1
        SPARSE_STATUS_ALLOC_FAILED      = 2
        SPARSE_STATUS_INVALID_VALUE     = 3
        SPARSE_STATUS_EXECUTION_FAILED  = 4
        SPARSE_STATUS_INTERNAL_ERROR    = 5
        SPARSE_STATUS_NOT_SUPPORTED     = 6

    cdef struct sparse_matrix
    ctypedef sparse_matrix* sparse_matrix_t
    cdef enum sparse_index_base_t:
        SPARSE_INDEX_BASE_ZERO  = 0
        SPARSE_INDEX_BASE_ONE   = 1

    cdef sparse_status_t mkl_sparse_d_create_coo(
        sparse_matrix_t *A, sparse_index_base_t indexing,
        MKL_INT rows, MKL_INT cols, MKL_INT nnz,
        MKL_INT *row_indx, MKL_INT *col_indx, double *values
    )
    cdef sparse_status_t mkl_sparse_z_create_coo(
        sparse_matrix_t *A, sparse_index_base_t indexing,
        MKL_INT rows, MKL_INT cols, MKL_INT nnz,
        MKL_INT *row_indx, MKL_INT *col_indx, MKL_Complex16 *values
    )
    cdef sparse_status_t mkl_sparse_optimize(sparse_matrix_t A)
    cdef sparse_status_t mkl_sparse_destroy(sparse_matrix_t A)

    cdef enum sparse_operation_t:
        SPARSE_OPERATION_NON_TRANSPOSE       = 10
        SPARSE_OPERATION_TRANSPOSE           = 11
        SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 12
    cdef enum sparse_matrix_type_t:
        SPARSE_MATRIX_TYPE_GENERAL           = 20
        SPARSE_MATRIX_TYPE_SYMMETRIC         = 21
        SPARSE_MATRIX_TYPE_HERMITIAN         = 22
        SPARSE_MATRIX_TYPE_TRIANGULAR        = 23
        SPARSE_MATRIX_TYPE_DIAGONAL          = 24
        SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR  = 25
        SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL    = 26
    cdef enum sparse_fill_mode_t:
        SPARSE_FILL_MODE_LOWER  = 40
        SPARSE_FILL_MODE_UPPER  = 41
        SPARSE_FILL_MODE_FULL   = 42
    cdef enum sparse_diag_type_t:
        SPARSE_DIAG_NON_UNIT    = 50
        SPARSE_DIAG_UNIT        = 51
    cdef struct matrix_descr:
        sparse_matrix_type_t  type
        sparse_fill_mode_t    mode
        sparse_diag_type_t    diag

    cdef sparse_status_t mkl_sparse_d_mv(
        sparse_operation_t operation, double alpha, sparse_matrix_t A,
        matrix_descr descr, double *x, double beta, double *y
    )
    cdef sparse_status_t mkl_sparse_z_mv(
        sparse_operation_t operation, MKL_Complex16 alpha, sparse_matrix_t A,
        matrix_descr descr, MKL_Complex16 *x, MKL_Complex16 beta, MKL_Complex16 *y
    )
    cdef enum sparse_layout_t:
        SPARSE_LAYOUT_ROW_MAJOR    = 101
        SPARSE_LAYOUT_COLUMN_MAJOR = 102
    cdef sparse_status_t mkl_sparse_d_mm(
        sparse_operation_t operation, double alpha, sparse_matrix_t A,
        matrix_descr descr, sparse_layout_t layout,
        double *x, MKL_INT columns, MKL_INT ldx, double beta,
        double *y, MKL_INT ldy
    )
    cdef sparse_status_t mkl_sparse_z_mm(
        sparse_operation_t operation, MKL_Complex16 alpha,
        sparse_matrix_t A, matrix_descr descr, sparse_layout_t layout,
        MKL_Complex16 *x, MKL_INT columns, MKL_INT ldx,
        MKL_Complex16 beta, MKL_Complex16 *y, MKL_INT ldy
    )
