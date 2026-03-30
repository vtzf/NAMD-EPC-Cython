cimport cython
from mpi4py cimport MPI
from mpi4py cimport libmpi as mpi
from libc.math cimport cos, sin, sqrt, exp, fabs, fmod, fmax, round, M_PI
from libc.stdio cimport printf, sscanf, sprintf, FILE, \
     SEEK_SET, SEEK_CUR, SEEK_END, fopen, fseek, fread, fwrite, fclose
from libc.string cimport memcpy, memset, strcpy
from libc.stdlib cimport malloc, calloc, free, qsort
from libc.time cimport time, time_t

cdef extern from "complex.h":
    double complex conj(double complex)
    double creal(double complex)
    double cimag(double complex)
    double complex cexp(double complex)
    double complex ccos(double complex)
    double complex csin(double complex)
    double cabs(double complex)
