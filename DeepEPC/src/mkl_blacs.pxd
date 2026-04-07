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
