cdef extern from "mkl_types.h":
    ctypedef int MKL_INT
    cdef struct _MKL_Complex16:
        double real
        double imag
    ctypedef _MKL_Complex16 MKL_Complex16


cdef extern from "mkl_cblas.h" nogil:
    cdef enum CBLAS_LAYOUT:
        CblasRowMajor=101
        CblasColMajor=102
    cdef enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113
    cdef enum CBLAS_SIDE:
        CblasLeft=141
        CblasRight=142
    cdef enum CBLAS_UPLO:
        CblasUpper=121
        CblasLower=122

    cdef void cblas_dgemm(
        CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB, MKL_INT M, MKL_INT N, MKL_INT K,
        double alpha, double* A, MKL_INT lda, double* B, MKL_INT ldb,
        double beta, double* C, MKL_INT ldc
    )
    cdef void cblas_zgemm(
        CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB, MKL_INT M, MKL_INT N, MKL_INT K,
        void* alpha, void* A, MKL_INT lda, void* B, MKL_INT ldb,
        void* beta, void* C, MKL_INT ldc
    )
    cdef void cblas_zgemm3m(
        CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
        CBLAS_TRANSPOSE TransB, MKL_INT M, MKL_INT N, MKL_INT K,
        void* alpha, void* A, MKL_INT lda, void* B, MKL_INT ldb,
        void* beta, void* C, MKL_INT ldc
    )
    cdef void cblas_dgemv(
        CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA, MKL_INT M,
        MKL_INT N, double alpha, double *A, MKL_INT lda, double *X,
        MKL_INT incX, double beta, double *Y, MKL_INT incY
    )
    cdef void cblas_zgemv(
        CBLAS_LAYOUT Layout, CBLAS_TRANSPOSE TransA,
        MKL_INT M, MKL_INT N, void *alpha, void *A, MKL_INT lda,
        void *X, MKL_INT incX, void *beta, void *Y, MKL_INT incY
    )
    cdef void cblas_zdotc_sub(
        MKL_INT N, void *X, MKL_INT incX,
        void *Y, MKL_INT incY, void *dotc
    )
    cdef void cblas_zdotu_sub(
        MKL_INT N, void *X, MKL_INT incX,
        void *Y, MKL_INT incY, void *dotu
    )
