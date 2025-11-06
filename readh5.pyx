#cython: language_level=3
#cython: cdivision=True

import sys
cimport cython
import numpy as np
cimport numpy as np
from mpi4py import MPI
from mpi4py cimport MPI
from mpi4py cimport libmpi as mpi
from libc.math cimport fmod, round, fabs, M_PI
from libc.stdio cimport sprintf, printf, FILE, \
     SEEK_SET, SEEK_CUR, SEEK_END, fopen, fseek, fread, fwrite, fclose
from libc.string cimport memcpy, memset, strcmp
from libc.stdlib cimport malloc, calloc, free, qsort

cdef extern from "hdf5.h":
    ctypedef long hid_t
    ctypedef int herr_t
    cdef int H5T_NATIVE_INT
    cdef int H5T_NATIVE_DOUBLE
    cdef hid_t H5S_ALL
    cdef unsigned int H5F_ACC_RDONLY
    cdef unsigned int H5P_DEFAULT
    cdef hid_t H5Fopen(
        char *filename, unsigned int flags, hid_t access_plist
    )
    cdef hid_t H5Dopen(
        hid_t file_id, const char *name, hid_t dapl_id
    )
    cdef herr_t H5Dread(
        hid_t dset_id, hid_t mem_type_id, hid_t mem_space_id,
        hid_t file_space_id, hid_t plist_id, void *buf
    )
    cdef herr_t H5Dclose(hid_t dset_id)
    cdef herr_t H5Fclose(hid_t file_id)


cdef extern from "complex.h":
    double complex conj(double complex)
    double cabs(double complex)
    double creal(double complex)
    double cimag(double complex)


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
cdef void Cross3(double[::1] a, double[::1] b, double[::1] c):
    c[0] = a[1]*b[2] - a[2]*b[1]
    c[1] = a[2]*b[0] - a[0]*b[2]
    c[2] = a[0]*b[1] - a[1]*b[0]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double Dot3(double[::1] a, double[::1] b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void Inv3(double[:,::1] a, double[:,::1] ainv):
    cdef double deta

    deta = a[0,0]*(a[1,1]*a[2,2]-a[1,2]*a[2,1])\
          -a[0,1]*(a[1,0]*a[2,2]-a[1,2]*a[2,0])\
          +a[0,2]*(a[1,0]*a[2,1]-a[1,1]*a[2,0])
    deta = 1.0/deta

    ainv[0,0] = (a[1,1]*a[2,2]-a[1,2]*a[2,1])*deta
    ainv[0,1] = (-(a[0,1]*a[2,2]-a[0,2]*a[2,1]))*deta
    ainv[0,2] = (a[0,1]*a[1,2]-a[0,2]*a[1,1])*deta
    ainv[1,0] = (-(a[1,0]*a[2,2]-a[1,2]*a[2,0]))*deta
    ainv[1,1] = (a[0,0]*a[2,2]-a[0,2]*a[2,0])*deta
    ainv[1,2] = (-(a[0,0]*a[1,2]-a[0,2]*a[1,0]))*deta
    ainv[2,0] = (a[1,0]*a[2,1]-a[1,1]*a[2,0])*deta
    ainv[2,1] = (-(a[0,0]*a[2,1]-a[0,1]*a[2,0]))*deta
    ainv[2,2] = (a[0,0]*a[1,1]-a[0,1]*a[1,0])*deta


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void RabcInv(double[:,::1] abc, double[:,::1] rabcinv):
    cdef double[:,::1] rabc = np.empty((3,3),dtype=np.float64)
    cdef double tmp[3]
    cdef double TpiCellV

    Cross3(abc[1],abc[2],tmp)
    TpiCellV = 2*M_PI/Dot3(abc[0],tmp)
    
    rabc[0,0] = tmp[0]*TpiCellV
    rabc[0,1] = tmp[1]*TpiCellV
    rabc[0,2] = tmp[2]*TpiCellV
    
    Cross3(abc[2],abc[0],tmp)
    rabc[1,0] = tmp[0]*TpiCellV
    rabc[1,1] = tmp[1]*TpiCellV
    rabc[1,2] = tmp[2]*TpiCellV

    Cross3(abc[0],abc[1],tmp)
    rabc[2,0] = tmp[0]*TpiCellV
    rabc[2,1] = tmp[1]*TpiCellV
    rabc[2,2] = tmp[2]*TpiCellV

    Inv3(rabc,rabcinv)


@cython.boundscheck(False)
@cython.wraparound(False)
def GetKqidxEF(
    int myid, int nqx, int nqy, int nqz ,double[:,::1] abc, double dt, 
    double hbar, double EFX, double EFY, double EFZ, int[:,::1] k1kidxE
):
    cdef double[:,::1] rabcinv = np.empty((3,3),dtype=np.float64)
    cdef double dkec[3]
    cdef double dked[3]
    cdef int dke[3]

    cdef int i, j, k, l, m, n, kxidx, kyidx, kzidx, kidx, kidx1

    # electric field: eV/$\AA$
    dkec[0] = EFX*dt/hbar
    dkec[1] = EFY*dt/hbar
    dkec[2] = EFZ*dt/hbar

    RabcInv(abc,rabcinv)
    dked[0] = dkec[0]*rabcinv[0,0]\
             +dkec[1]*rabcinv[1,0]+dkec[2]*rabcinv[2,0]
    dked[1] = dkec[0]*rabcinv[0,1]\
             +dkec[1]*rabcinv[1,1]+dkec[2]*rabcinv[2,1]
    dked[2] = dkec[0]*rabcinv[0,2]\
             +dkec[1]*rabcinv[1,2]+dkec[2]*rabcinv[2,2]

    dke[0] = <int>round(dked[0]*nqx)
    dke[1] = <int>round(dked[1]*nqy)
    dke[2] = <int>round(dked[2]*nqz)
    dked[0] = <double>(dke[0])/nqx
    dked[1] = <double>(dke[1])/nqy
    dked[2] = <double>(dke[2])/nqz

    if myid == 0:
        printf("dkcx = %f, dkcy = %f, dkcz = %f\n",dkec[0],dkec[1],dkec[2])
        printf("dkdx = %f, dkdy = %f, dkdz = %f\n",dked[0],dked[1],dked[2])
        printf("dkx = %d, dky = %d, dkz = %d\n",dke[0],dke[1],dke[2])

    for i in range(nqx):
        for j in range(nqy):
            for k in range(nqz):
                kxidx = (i+dke[0])%nqx
                kyidx = (j+dke[1])%nqy
                kzidx = (k+dke[2])%nqz
                if kxidx < 0: kxidx += nqx
                if kyidx < 0: kyidx += nqy
                if kzidx < 0: kzidx += nqz
                kidx = (kxidx*nqy+kyidx)*nqz+kzidx
                kxidx = (i-dke[0])%nqx
                kyidx = (j-dke[1])%nqy
                kzidx = (k-dke[2])%nqz
                if kxidx < 0: kxidx += nqx
                if kyidx < 0: kyidx += nqy
                if kzidx < 0: kzidx += nqz
                kidx1 = (kxidx*nqy+kyidx)*nqz+kzidx

                l = (i*nqy+j)*nqz+k
                k1kidxE[l,0] = kidx
                k1kidxE[l,1] = kidx1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void GetKqidx(
    int myid, int nq, int k_p_m, 
    double[:,::1] k_list, double[:,::1] q_list,
    int nqx, int nqy, int nqz, int nk_set, int nk_p_set,
    int * ekidx_set_a, int * ekidx_count_a, 
    int * ekidx_set, int * ekidx_count,
    int[:,::1] kqidx, int[:,::1] k1qidx
):
    cdef int i, j, k, l, m, n, qxidx, qyidx, qzidx, \
             qidx0, qidx1, idx_s, idx_e, idx_t0, idx_t1, k0, k1
    cdef double k_list_x, k_list_y, k_list_z, kkx, kky, kkz,\
                q_list_x, q_list_y, q_list_z
    cdef int * q2qmap = <int*>malloc(nq*2*sizeof(int))

    for i in range(nq):
        q_list_x = fmod(q_list[i,0],1.0)
        if q_list_x<0:
            q_list_x += 1
        q_list_y = fmod(q_list[i,1],1.0)
        if q_list_y<0:
            q_list_y += 1
        q_list_z = fmod(q_list[i,2],1.0)
        if q_list_z<0:
            q_list_z += 1

        qxidx = <int>round(q_list_x*nqx)
        qyidx = <int>round(q_list_y*nqy)
        qzidx = <int>round(q_list_z*nqz)

        q2qmap[i*2+1] = i
        q2qmap[i*2] = (qxidx*nqy+qyidx)*nqz+qzidx

    qsort(q2qmap,nq,sizeof(int)*2,&Cmp)

    k = 0
    for i in range(nk_p_set):
        k0 = ekidx_set[i]
        k_list_x = k_list[k0,0]
        k_list_y = k_list[k0,1]
        k_list_z = k_list[k0,2]
        l = 0
        for j in range(nk_set):
            k1 = ekidx_set_a[j]
            kkx = fmod(k_list[k1,0]-k_list_x,1.0)
            if kkx<0:
                kkx += 1
            kky = fmod(k_list[k1,1]-k_list_y,1.0)
            if kky<0:
                kky += 1
            kkz = fmod(k_list[k1,2]-k_list_z,1.0)
            if kkz<0:
                kkz += 1

            qxidx = <int>round(kkx*nqx)
            qyidx = <int>round(kky*nqy)
            qzidx = <int>round(kkz*nqz)

            qidx0 = (qxidx*nqy+qyidx)*nqz+qzidx
            if (qidx0<q2qmap[0] or qidx0>q2qmap[nq*2-2]):
                printf("[k,k\']->q map incomplete!\n")
                sys.exit()
            else:
                idx_s = 0
                idx_e = nq
                while 1:
                    if ((idx_e-idx_s)<=1):
                        if (qidx0==q2qmap[idx_s*2]):
                            idx_t0 = idx_s
                        elif (qidx0==q2qmap[idx_e*2]):
                            idx_t0 = idx_e
                        else:
                            printf("[k,k\']->q map incomplete!\n")
                            sys.exit()
                        break
                    idx_t0 = (idx_s+idx_e)/2
                    if (qidx0<q2qmap[idx_t0*2]):
                        idx_e = idx_t0
                    else:
                        idx_s = idx_t0

            qidx1 = (((nqx-qxidx)%nqx)*nqy+(nqy-qyidx)%nqy)*nqz+(nqz-qzidx)%nqz
            if (qidx1<q2qmap[0] or qidx1>q2qmap[nq*2-2]):
                printf("[k,k\']->q map incomplete!\n")
                sys.exit()
            else:
                idx_s = 0
                idx_e = nq
                while 1:
                    if ((idx_e-idx_s)<=1):
                        if (qidx1==q2qmap[idx_s*2]):
                            idx_t1 = idx_s
                        elif (qidx1==q2qmap[idx_e*2]):
                            idx_t1 = idx_e
                        else:
                            printf("[k,k\']->q map incomplete!\n")
                            sys.exit()
                        break
                    idx_t1 = (idx_s+idx_e)/2
                    if (qidx1<q2qmap[idx_t1*2]):
                        idx_e = idx_t1
                    else:
                        idx_s = idx_t1

            for m in range(ekidx_count[i]):
                for n in range(ekidx_count_a[j]):
                    kqidx[k+m,l+n] = q2qmap[idx_t0*2+1]
                    k1qidx[k_p_m+k+m,l+n] = q2qmap[idx_t1*2+1]

            l += ekidx_count_a[j]
        k += ekidx_count[i]

    free(q2qmap)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void KProcSplit(
    mpi.MPI_Comm comm, int myid, int nprocs, int Num, int nk_s, int nq, 
    int nqx, int nqy, int nqz, int nbands, int[::1] ekidx, int[::1] ebidx, 
    int * k_range, double[:,::1] k_list_a, double[:,::1] q_list, 
    int[::1] k_proc, int[::1] k_proc_num, int nk_proc, 
    int k_p_m, int[:,::1] kqidx_p, int[:,::1] k1qidx_p, int **** k_proc_range
):
    cdef mpi.MPI_Datatype INT2
    cdef int k_proc_min = k_proc[myid]
    cdef int k_proc_max = k_proc[myid+1]
    cdef int i, j, k, l, m, n, ekset, n_ekset_a, n_ekset, \
             k_s, k_e, nk_min, nk_max, idx_s, idx_e, \
             idx_s0, idx_e0, nidx, k_range_min, k_range_max
    cdef int * ekidx_set_a = <int*>malloc(nk_s*sizeof(int))
    cdef int * ekidx_count_a = <int*>calloc(nk_s,sizeof(int))
    cdef int * ekidx_set = <int*>malloc(nk_proc*sizeof(int))
    cdef int * ekidx_count = <int*>calloc(nk_proc,sizeof(int))
    cdef int * ekidx_sum
    cdef int * ebidx_count

    n_ekset_a = -1
    ekset = -1
    for i in range(nk_s):
        if ekidx[i] == ekset:
            ekidx_count_a[n_ekset_a] += 1
        else:
            ekset = ekidx[i]
            n_ekset_a += 1
            ekidx_set_a[n_ekset_a] = ekset
            ekidx_count_a[n_ekset_a] += 1
    n_ekset_a += 1

    n_ekset = -1
    ekset = -1
    for i in range(k_proc_min,k_proc_max):
        if ekidx[i] == ekset:
            ekidx_count[n_ekset] += 1
        else:
            ekset = ekidx[i]
            n_ekset += 1
            ekidx_set[n_ekset] = ekset
            ekidx_count[n_ekset] += 1
    n_ekset += 1

    GetKqidx(
        myid,nq,k_p_m,k_list_a,q_list,nqx,nqy,nqz,
        n_ekset_a,n_ekset,ekidx_set_a,ekidx_count_a,
        ekidx_set,ekidx_count,kqidx_p,k1qidx_p
    )

    ekidx_sum = <int*>calloc(n_ekset+1,sizeof(int))
    for i in range(n_ekset):
        for j in range(i,n_ekset):
            ekidx_sum[j+1] += ekidx_count[i]

    ebidx_count = <int*>malloc(n_ekset*nbands*sizeof(int))
    for i in range(n_ekset):
        for j in range(ekidx_sum[i],ekidx_sum[i+1]):
            ebidx_count[i*nbands+j-ekidx_sum[i]] \
            = ebidx[k_proc_min+j]

    k_s = -1
    k_e = -1
    nk_min = ekidx[k_proc_min]
    nk_max = ekidx[k_proc_max-1]
    for i in range(Num):
        if (nk_min>=k_range[i] \
        and nk_min<k_range[i+1]):
            k_s = i
        if (nk_max>=k_range[i] \
        and nk_max<k_range[i+1]):
            k_e = i
        if (k_s>=0 and k_e>=0):
            break

    k_proc_range[0] = <int***>malloc((k_e-k_s+1)*sizeof(int**))
    idx_s = 0
    idx_e = n_ekset
    idx_s0 = 0
    for i in range(k_e-k_s+1):
        k = i+k_s
        k_proc_range[0][i] = <int**>malloc(4*sizeof(int*))
        k_proc_range[0][i][0] = <int*>malloc(3*sizeof(int))
        k_proc_range[0][i][0][0] = k_e-k_s+1
        k_proc_range[0][i][0][1] = k

        k_range_min = k_range[k]
        if k_s == k_e:
            idx_e0 = idx_e
        else:
            k_range_max = k_range[k+1]
            idx_e0 = idx_e
            for j in range(idx_s,idx_e):
                if (ekidx_set[j]>=k_range_min):
                    idx_s0 = j
                    break
            for j in range(idx_s0,idx_e):
                if (ekidx_set[j]>=k_range_max):
                    idx_e0 = j
                    break
        nidx = idx_e0-idx_s0

        k_proc_range[0][i][0][2] = nidx
        k_proc_range[0][i][1] = <int*>malloc(nidx*sizeof(int))
        k_proc_range[0][i][2] = <int*>malloc(nidx*sizeof(int))
        k_proc_range[0][i][3] = <int*>malloc(nidx*nbands*sizeof(int))
        for j in range(nidx):
            k_proc_range[0][i][1][j] = ekidx_set[idx_s0+j]-k_range_min
            k_proc_range[0][i][2][j] = ekidx_count[idx_s0+j]
            for l in range(k_proc_range[0][i][2][j]):
                k_proc_range[0][i][3][j*nbands+l] \
                = ebidx_count[(idx_s0+j)*nbands+l]

        idx_s0 = idx_e0
        idx_s = idx_e0

    free(ekidx_set_a)
    free(ekidx_count_a)
    free(ekidx_set)
    free(ekidx_count)
    free(ekidx_sum)
    free(ebidx_count)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void TransposeBlock(
    mpi.MPI_Comm c_comm, int nprocs, int myid, int nmodes, 
    int nm_block, int nk_a, int k_p_m, double complex[:,:,::1] epc_a,
    int[::1] k_proc, int[::1] k_proc_num, double f, int * TRANS
):
    cdef int h, i, j, k, l, nm_min, nm_max, nm_num, nm_loop, \
             nm_buffer, k_proc_i, nk_proc, idx0, idx1, idx2, idx3
    cdef int k_p = k_proc[myid]
    cdef int nk_p = k_proc_num[myid]
    cdef int * scount = <int*>malloc(nprocs*sizeof(int))
    cdef int * sdispl = <int*>malloc(nprocs*sizeof(int))
    cdef int * rcount = <int*>malloc(nprocs*sizeof(int))
    cdef int * rdispl = <int*>malloc(nprocs*sizeof(int))
    cdef double complex * epc_sbuf
    cdef double complex * epc_rbuf
    cdef double complex epc_t

    nm_loop = <int>(nmodes/nm_block)
    nm_buffer = nm_block
    if nmodes%nm_block != 0: 
        nm_loop += 1
        nm_buffer += 1
    epc_sbuf = <double complex*>malloc(\
        (nk_a*nk_p*nm_buffer)*sizeof(double complex))
    epc_rbuf = <double complex*>malloc(\
        (nk_p*nk_a*nm_buffer)*sizeof(double complex))
    for j in range(nprocs):
        scount[j] = nm_buffer*nk_p*k_proc_num[j]
        rcount[j] = nm_buffer*nk_p*k_proc_num[j]
        sdispl[j] = nm_buffer*nk_p*k_proc[j]
        rdispl[j] = nm_buffer*nk_p*k_proc[j]

    for h in range(nm_loop):
        nm_min = <int>((nmodes*h)/nm_loop)
        nm_max = <int>((nmodes*(h+1))/nm_loop)
        nm_num = nm_max - nm_min
        # Only keep Upper/Lower part
        # prepare send buffer
        memset(epc_sbuf,0,(nk_a*nk_p*nm_buffer)*sizeof(double complex))
        if TRANS[0] == 0:
            for i in range(nm_num):
                for j in range(nk_p):
                    for k in range(k_p+j+1):
                        epc_t = epc_a[i+nm_min,j+k_p_m,k]*f
                        epc_a[i+nm_min,j+k_p_m,k] = conj(epc_t)
                        epc_sbuf[(k*nk_p+j)*nm_buffer+i] = epc_t
                    for k in range(k_p+j+1,nk_a):
                        epc_a[i+nm_min,j+k_p_m,k] = 0
        elif TRANS[1] == 0:
            for i in range(nm_num):
                for j in range(nk_p):
                    for k in range(k_p+j):
                        epc_a[i+nm_min,j+k_p_m,k] = 0
                    for k in range(k_p+j,nk_a):
                        epc_t = epc_a[i+nm_min,j+k_p_m,k]*f
                        epc_a[i+nm_min,j+k_p_m,k] = conj(epc_t)
                        epc_sbuf[(k*nk_p+j)*nm_buffer+i] = epc_t
        else:
            for i in range(nm_num):
                for j in range(nk_p):
                    for k in range(nk_a):
                        epc_t = epc_a[i+nm_min,j+k_p_m,k]*f
                        epc_a[i+nm_min,j+k_p_m,k] = conj(epc_t)
                        epc_sbuf[(k*nk_p+j)*nm_buffer+i] = epc_t
        mpi.MPI_Alltoallv(
            epc_sbuf,scount,sdispl,
            mpi.MPI_DOUBLE_COMPLEX,epc_rbuf,rcount,rdispl,
            mpi.MPI_DOUBLE_COMPLEX,c_comm
        )

        for i in range(nprocs):
            k_proc_i = k_proc[i]
            nk_proc = k_proc_num[i]
            idx0 = rdispl[i]
            for j in range(nk_p):
                for k in range(nk_proc):
                    idx1 = j*nk_proc+k
                    for l in range(nm_num):
                        epc_a[l+nm_min,j+k_p_m,k+k_proc_i]\
                        += epc_rbuf[idx0+idx1*nm_buffer+l]

    if TRANS[2] != 0:
        for k in range(nmodes):
            for i in range(nk_p):
                epc_a[k,i+k_p_m,i+k_p] /= 2.0
    
    for k in range(nmodes):
        for i in range(nk_p):
            epc_a[k,i+k_p_m,i+k_p] = cabs(epc_a[k,i+k_p_m,i+k_p])

    free(epc_sbuf)
    free(epc_rbuf)
    free(scount)
    free(sdispl)
    free(rcount)
    free(rdispl)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void AllGather(
    mpi.MPI_Comm c_comm, int nprocs, int shm_nprocs, int myid,
    int nmodes, int nk_a, int[::1] k_proc_num, int[:,::1] k1qidx,
    double complex[:,:,::1] epc_a
):
    cdef mpi.MPI_Comm remote_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef mpi.MPI_Datatype CPLX_N, INT_N
    cdef int i, j, k, nnode, node_id, ierr
    cdef int * nodelist
    cdef int * epc_remote
    cdef int * epc_remote_num

    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(sizeof(int)*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&remote_comm)
    # Allgatherv
    mpi.MPI_Barrier(c_comm)
    if (nnode>1):
        epc_remote = <int*>calloc((nnode+1),sizeof(int))
        epc_remote_num = <int*>calloc(nnode,sizeof(int))
        for i in range(nnode):
            for j in range(i*shm_nprocs,(i+1)*shm_nprocs):
                epc_remote_num[i] += k_proc_num[j]
            for j in range(i+1,nnode+1):
                epc_remote[j] += epc_remote_num[i]
        mpi.MPI_Type_contiguous(nk_a,mpi.MPI_DOUBLE_COMPLEX,&CPLX_N)
        mpi.MPI_Type_commit(&CPLX_N)
        if remote_comm != mpi.MPI_COMM_NULL:
            for i in range(nmodes):
                mpi.MPI_Allgatherv(
                    mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,&epc_a[i,0,0],
                    &epc_remote_num[0],&epc_remote[0],CPLX_N,remote_comm
                )
#            for j in range(nnode):
#                printf("%d:%d,%d\n",myid,epc_remote[j],epc_remote_num[j])
#                for i in range(5):
#                    mpi.MPI_Bcast(
#                        &epc_a[i,epc_remote[j],0],
#                        epc_remote_num[j],CPLX_N,j,remote_comm
#                    )
        mpi.MPI_Type_free(&CPLX_N)

        mpi.MPI_Type_contiguous(nk_a,mpi.MPI_INT,&INT_N)
        mpi.MPI_Type_commit(&INT_N)
        if remote_comm != mpi.MPI_COMM_NULL:
            mpi.MPI_Allgatherv(
                mpi.MPI_IN_PLACE,0,mpi.MPI_DATATYPE_NULL,&k1qidx[0,0],
                &epc_remote_num[0],&epc_remote[0],INT_N,remote_comm
            )
        mpi.MPI_Type_free(&INT_N)

        free(epc_remote)
        free(epc_remote_num)
        mpi.MPI_Barrier(c_comm)

    free(nodelist)


@cython.boundscheck(False)
@cython.wraparound(False)
def ReadH5Info(
    MPI.Comm comm, int myid, 
    char* Dir, int Num, int nqx, int nqy, int nqz, double EMIN, double EMAX
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef hid_t file_id, data_id
    cdef herr_t status
    cdef int nk_a, nq, nk_s, nbands, i, kidx, bidx, \
             kmin_s, kmax_s, bmin_s, bmax_s
    cdef int info[4]
    cdef int info_buf[4]
    cdef int * k_range
    cdef double [:,::1] abc = np.zeros((3,3),dtype=np.float64)
    cdef char name[128]

    sprintf(name,"%s%d.h5",Dir,1)
    file_id = H5Fopen(name,H5F_ACC_RDONLY,H5P_DEFAULT)
    data_id = H5Dopen(file_id,"el_ph_band_info/information",H5P_DEFAULT)
    status = H5Dread(
        data_id,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,info
    )
    H5Dclose(data_id)
    data_id = H5Dopen(file_id,"el_ph_band_info/lattice_vec_angstrom",H5P_DEFAULT)
    status = H5Dread(
        data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,&abc[0,0]
    )
    H5Dclose(data_id)
    status = H5Fclose(file_id)

    nk_p = info[0]
    nq = info[1]
    nbands = info[2]
    nmodes = info[3]

    if myid == 0:
        k_range = <int*>calloc(Num+1,sizeof(int))
        for i in range(Num):
            sprintf(name,"%s%d.h5",Dir,i+1)
            file_id = H5Fopen(name,H5F_ACC_RDONLY,H5P_DEFAULT)
            data_id = H5Dopen(file_id,"el_ph_band_info/information",H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,info_buf
            )
            for j in range(i+1,Num+1):
                k_range[j] += info_buf[0]
            H5Dclose(data_id)
            status = H5Fclose(file_id)
        nk_a = k_range[Num]
    mpi.MPI_Bcast(&nk_a,1,mpi.MPI_INT,0,c_comm)

    if (nk_a>nq):
        printf('Unmatch of q and k points exists!\n')
        sys.exit()
    if (nqx*nqy*nqz!=nq):
        printf('Unmatch of nq and nqx*nqy*nqz!\n')
        sys.exit()

    if (myid==0):
        energy_buf = <double*>malloc(sizeof(double)*nk_a*nbands)
        for i in range(Num):
            sprintf(name,"%s%d.h5",Dir,i+1)
            file_id = H5Fopen(name,H5F_ACC_RDONLY,H5P_DEFAULT)
            data_id = H5Dopen(file_id,"el_ph_band_info/el_band_eV",H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,
                H5P_DEFAULT,&energy_buf[k_range[i]*nbands]
            )
            H5Dclose(data_id)
            status = H5Fclose(file_id)

        nk_s = 0
        kmin_s = 0
        kmax_s = 0
        bmin_s = 0
        bmax_s = 0
        for i in range(nk_a*nbands):
            en = energy_buf[i]
            if (en>=EMIN and en<=EMAX):
                kidx = <int>(i/nbands)
                bidx = i%nbands
                if kmin_s > kidx: kmin_s = kidx
                if kmax_s < kidx: kmax_s = kidx
                if bmin_s > bidx: bmin_s = bidx
                if bmax_s < bidx: bmax_s = bidx
                nk_s += 1
        free(energy_buf)
        free(k_range)
    mpi.MPI_Bcast(&nk_s,1,mpi.MPI_INT,0,c_comm)
    mpi.MPI_Bcast(&kmin_s,1,mpi.MPI_INT,0,c_comm)
    mpi.MPI_Bcast(&kmax_s,1,mpi.MPI_INT,0,c_comm)
    mpi.MPI_Bcast(&bmin_s,1,mpi.MPI_INT,0,c_comm)
    mpi.MPI_Bcast(&bmax_s,1,mpi.MPI_INT,0,c_comm)

    return nk_s, nk_a, nbands, nmodes, kmin_s, kmax_s, bmin_s, bmax_s, abc


@cython.boundscheck(False)
@cython.wraparound(False)
def ReadH5(
    MPI.Comm comm, char * Dir, int Num, int nmodes, int nm_block, int nk_s, 
    double PHCUT, double EMIN, double EMAX, int nqx, int nqy, int nqz,
    char * LTRANS, bint LEPCSHM, bint LPHSHM, int[:,::1] k1qidx_p, 
    double[:,::1] energy_buf, double[:,::1] phonon, double complex[:,:,::1] epc_a
):
    cdef hid_t file_id, data_id
    cdef herr_t status
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm shm_comm, node_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef int myid, shm_id, nprocs, shm_nprocs, ierr, \
             i, j, k, l, m, n, p, q, kidx, fnum, kqidx, eb,\
             phzero_num, phzero0, phzero1, nq, nk_p, nk_a, \
             nk_proc, k_proc_min, nbands, nnode, node_id
    cdef int * nodelist
    cdef int s_int = sizeof(int)
    cdef int n_p = nqx*nqy*nqz
    cdef int info[4]
    cdef int info_buf[4]
    cdef double * phonon_buf
    cdef double[:,::1] k_list_a, q_list
    cdef double[::1] energy
    cdef int[:,::1] phzero
    cdef int * k_range = <int*>calloc(Num+1,sizeof(int))
    cdef int[::1] ekidx, ebidx, k_proc, k_proc_num
    cdef int[:,::1] kqidx_p
    cdef int *** k_p_r
    cdef int * k_p_r0
    cdef long l_phzero, nmodes_l
    cdef double[:,:,:,::1] epc_p_r, epc_p_i
    cdef char name[128]
    cdef char epcname_r[128]
    cdef char epcname_i[128]
    cdef double starttime, endtime, en, f_r
    cdef int TRANS[3]
    TRANS[0] = strcmp(LTRANS,"U")
    TRANS[1] = strcmp(LTRANS,"L")
    TRANS[2] = strcmp(LTRANS,"S")
    if TRANS[0]!=0 and TRANS[1]!=0 and TRANS[2]!=0:
        printf("No LTRANS=\"%s\" option!\n",LTRANS)
        sys.exit()

    starttime = mpi.MPI_Wtime()
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)
    # get shm_comm
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    shm_comm_py = comm.Split_type(MPI.COMM_TYPE_SHARED)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&node_comm)

    sprintf(name,"%s%d.h5",Dir,1)
    file_id = H5Fopen(name,H5F_ACC_RDONLY,H5P_DEFAULT)
    data_id = H5Dopen(file_id,"el_ph_band_info/information",H5P_DEFAULT)
    status = H5Dread(
        data_id,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,info
    )
    H5Dclose(data_id)

    nk_p = info[0]
    nq = info[1]
    nbands = info[2]
    nmodes = info[3]
    if (n_p < nq):
        printf('q grid number doesn\'t match file nq!\n')
        sys.exit()

    q_list = np.zeros((nq,3),dtype=np.float64)
    data_id = H5Dopen(file_id,"el_ph_band_info/q_list",H5P_DEFAULT)
    status = H5Dread(
        data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,&q_list[0,0]
    )
    H5Dclose(data_id)

    # alloc phonon
    nmodes_l = nmodes
    if LPHSHM:
        if (shm_id==0):
            l_phzero = nmodes_l*nq*2*sizeof(int)
        else:
            l_phzero = 0
        win00 = MPI.Win.Allocate_shared(l_phzero,sizeof(int),comm=shm_comm_py)
        buf00,s_int = win00.Shared_query(0)
        phzero = np.ndarray(buffer=buf00,dtype=np.int32,shape=(nmodes_l*nq,2))
        mpi.MPI_Barrier(shm_comm)
        if (myid==0):
            phonon_buf = <double*>malloc(sizeof(double)*nq*nmodes)
            data_id = H5Dopen(file_id,"el_ph_band_info/ph_disp_meV",H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,phonon_buf
            )
            H5Dclose(data_id)

            phzero_num = 0
            for i in range(nq):
                for j in range(nmodes):
                    k = i*nmodes+j
                    if phonon_buf[k]<=0:
                        phonon[j,i] = 1e-13
                        phzero[phzero_num,0] = i
                        phzero[phzero_num,1] = j
                        phzero_num += 1
                    elif phonon_buf[k]<PHCUT:
                        phonon[j,i] = phonon_buf[k]/1000
                        phzero[phzero_num,0] = i
                        phzero[phzero_num,1] = j
                        phzero_num += 1
                    else:
                        phonon[j,i] = phonon_buf[k]/1000
            free(phonon_buf)

        mpi.MPI_Bcast(&phzero_num,1,mpi.MPI_INT,0,c_comm)
        mpi.MPI_Bcast(&phzero[0,0],phzero_num*2,mpi.MPI_INT,0,node_comm)
        mpi.MPI_Bcast(&phonon[0,0],nmodes*nq,mpi.MPI_DOUBLE,0,node_comm)
    else:
        phzero = np.zeros((nmodes*nq,2),dtype=np.int32)
        if myid == 0:
            phonon_buf = <double*>malloc(sizeof(double)*nq*nmodes)
            data_id = H5Dopen(file_id,"el_ph_band_info/ph_disp_meV",H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,phonon_buf
            )
            H5Dclose(data_id)
    
            phzero_num = 0
            for i in range(nq):
                for j in range(nmodes):
                    k = i*nmodes+j
                    if phonon_buf[k]<=0:
                        phonon[j,i] = 1e-13
                        phzero[phzero_num,0] = i
                        phzero[phzero_num,1] = j
                        phzero_num += 1
                    elif phonon_buf[k]<PHCUT:
                        phonon[j,i] = phonon_buf[k]/1000
                        phzero[phzero_num,0] = i
                        phzero[phzero_num,1] = j
                        phzero_num += 1
                    else:
                        phonon[j,i] = phonon_buf[k]/1000
            free(phonon_buf)

        mpi.MPI_Bcast(&phzero_num,1,mpi.MPI_INT,0,c_comm)
        mpi.MPI_Bcast(&phzero[0,0],phzero_num*2,mpi.MPI_INT,0,c_comm)
        mpi.MPI_Bcast(&phonon[0,0],nmodes*nq,mpi.MPI_DOUBLE,0,c_comm)

    status = H5Fclose(file_id)

    if myid == 0:
        for i in range(Num):
            sprintf(name,"%s%d.h5",Dir,i+1)
            file_id = H5Fopen(name,H5F_ACC_RDONLY,H5P_DEFAULT)
            data_id = H5Dopen(file_id,"el_ph_band_info/information",H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_INT,H5S_ALL,H5S_ALL,H5P_DEFAULT,info_buf
            )
            for j in range(i+1,Num+1):
                k_range[j] += info_buf[0]
            H5Dclose(data_id)
            status = H5Fclose(file_id)
    #printf("%d:%d,%d\n",myid,k_range[0],k_range[Num])
    mpi.MPI_Barrier(c_comm)
    mpi.MPI_Bcast(k_range,Num+1,mpi.MPI_INT,0,c_comm)
    nk_a = k_range[Num]
    #printf("%d:%d,%d,%d,%d\n",myid,Num,k_range[0],k_range[Num],nk_a)

    k_list_a = np.zeros((nk_a,3),dtype=np.float64)
    energy = np.zeros((nk_s),dtype=np.float64)
    ekidx = np.zeros((nk_s),dtype=np.int32)
    ebidx = np.zeros((nk_s),dtype=np.int32)
    if myid == 0:
        for i in range(Num):
            sprintf(name,"%s%d.h5",Dir,i+1)
            file_id = H5Fopen(name,H5F_ACC_RDONLY,H5P_DEFAULT)
            data_id = H5Dopen(file_id,"el_ph_band_info/el_band_eV",H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,
                H5P_DEFAULT,&energy_buf[k_range[i],0]
            )
            H5Dclose(data_id)
            data_id = H5Dopen(file_id,"el_ph_band_info/k_list",H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_DOUBLE,H5S_ALL,
                H5S_ALL,H5P_DEFAULT,&k_list_a[k_range[i],0]
            )
            H5Dclose(data_id)
            status = H5Fclose(file_id)
    
        k = 0
        for i in range(nk_a):
            for j in range(nbands):
                en = energy_buf[i,j]
                if (en>=EMIN and en<=EMAX):
                    energy[k] = en
                    ekidx[k] = i
                    ebidx[k] = j
                    k += 1
    mpi.MPI_Bcast(&energy_buf[0,0],nk_a*nbands,mpi.MPI_DOUBLE,0,node_comm)
    mpi.MPI_Bcast(&k_list_a[0,0],nk_a*3,mpi.MPI_DOUBLE,0,c_comm)
    mpi.MPI_Bcast(&energy[0],nk_s,mpi.MPI_DOUBLE,0,c_comm)
    mpi.MPI_Bcast(&ekidx[0],nk_s,mpi.MPI_INT,0,c_comm)
    mpi.MPI_Bcast(&ebidx[0],nk_s,mpi.MPI_INT,0,c_comm)

    k_proc = np.zeros((nprocs+1),dtype=np.int32)
    k_proc_num = np.zeros((nprocs),dtype=np.int32)
    for i in range(nprocs):
        k_proc[i+1] = ((i+1)*nk_s)/nprocs
        k_proc_num[i] = k_proc[i+1]-k_proc[i]
    nk_proc = k_proc_num[myid]
    if LEPCSHM: k_proc_min = k_proc[myid]
    else: k_proc_min = 0
    kqidx_p = np.zeros((nk_proc,nk_s),dtype=np.int32)
    KProcSplit(
        c_comm,myid,nprocs,Num,nk_s,nq,nqx,nqy,nqz,nbands,
        ekidx,ebidx,k_range,k_list_a,q_list,k_proc,
        k_proc_num,nk_proc,k_proc_min,kqidx_p,k1qidx_p,&k_p_r
    )
    k_list_a = None
    free(k_range)

    # unit conversion factor # meV to eV
    if TRANS[2] == 0:
        f_r = 1.0/2000.0
    else:
        f_r = 1.0/1000.0

    fnum = k_p_r[0][0][0]
    epc_p_r = np.zeros((nq,nmodes,nbands,nbands),dtype=np.float64)
    epc_p_i = np.zeros((nq,nmodes,nbands,nbands),dtype=np.float64)
    kidx = 0
    for i in range(fnum):
        sprintf(name,"%s%d.h5",Dir,k_p_r[i][0][1]+1)
        file_id = H5Fopen(name,H5F_ACC_RDONLY,H5P_DEFAULT)
        for j in range(k_p_r[i][0][2]):
            sprintf(epcname_r,"g_ephmat_total_meV/g_ik_r_%d",k_p_r[i][1][j]+1)
            sprintf(epcname_i,"g_ephmat_total_meV/g_ik_i_%d",k_p_r[i][1][j]+1)
            data_id = H5Dopen(file_id,epcname_r,H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,
                H5P_DEFAULT,&epc_p_r[0,0,0,0]
            )
            H5Dclose(data_id)
            data_id = H5Dopen(file_id,epcname_i,H5P_DEFAULT)
            status = H5Dread(
                data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,
                H5P_DEFAULT,&epc_p_i[0,0,0,0]
            )
            H5Dclose(data_id)

            for k in range(phzero_num):
                phzero0 = phzero[k,0]
                phzero1 = phzero[k,1]
                for l in range(nbands):
                    for m in range(nbands):
                        epc_p_r[phzero0,phzero1,l,m] = 0
                        epc_p_i[phzero0,phzero1,l,m] = 0

            n = k_p_r[i][2][j]
            k_p_r0 = &(k_p_r[i][3][j*nbands])
            for m in range(nk_s):
                kqidx = kqidx_p[kidx,m]
                eb = ebidx[m]
                for k in range(nmodes):
                    for l in range(n):
                        epc_a[k,k_proc_min+kidx+l,m] \
                        = epc_p_r[kqidx,k,k_p_r0[l],eb] \
                        + epc_p_i[kqidx,k,k_p_r0[l],eb]*1j
            
            kidx += n

        status = H5Fclose(file_id)

    TransposeBlock(
        c_comm,nprocs,myid,nmodes,nm_block,nk_s,
        k_proc_min,epc_a,k_proc,k_proc_num,f_r,TRANS
    )
    if LEPCSHM:
        AllGather(
            c_comm,nprocs,shm_nprocs,myid,nmodes,
            nk_s,k_proc_num,k1qidx_p,epc_a
        )

    for i in range(fnum):
        for j in range(4):
            free(k_p_r[i][j])
        free(k_p_r[i])
    free(k_p_r)

    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("Read epc time: %.6fs.\n",endtime-starttime)

    return nk_a,n_p,nbands,ekidx,ebidx,\
           k_proc,k_proc_num,energy


@cython.boundscheck(False)
@cython.wraparound(False)
def ReadNpy(
    MPI.Comm comm, int nmodes, int nm_block, int nbands, int nqx, 
    int nqy, int nqz, int nk_s, double PHCUT, double EMIN, double EMAX, 
    int[:,::1] k1qidx_p, double[:,::1] energy_buf, double[:,::1] phonon, 
    double complex[:,:,::1] epc_a, char * LTRANS, 
    str phname, char * epcname, bint LEPCSHM, bint LPHSHM
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm shm_comm, node_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef int myid, shm_id, nprocs, shm_nprocs, ierr, \
             i, j, k, l, m, n, p, q, kidx, kqidx, eb,\
             phzero_num, phzero0, phzero1, n_p,\
             nk_a, nk_proc, koffset, k_proc_min, nnode, node_id
    cdef int * nodelist
    cdef int s_int = sizeof(int)
    cdef int nq = nqx*nqy*nqz
    cdef double[:,::1] phonon_buf, k_list_a
    cdef double[::1] energy
    cdef int[:,::1] phzero
    cdef int * k_range = <int*>calloc(2,sizeof(int))
    cdef int[::1] ekidx, ebidx, k_proc, k_proc_num
    cdef int[:,::1] kqidx_p
    cdef int *** k_p_r
    cdef int * k_p_r0
    cdef long l_phzero, l_epc_t, nmodes_l
    cdef double complex[:,:,:,::1] epc_t
    cdef double starttime, endtime, en, f_r
    cdef int TRANS[3]
    cdef FILE * fp

    TRANS[0] = strcmp(LTRANS,"U")
    TRANS[1] = strcmp(LTRANS,"L")
    TRANS[2] = strcmp(LTRANS,"S")
    if TRANS[0]!=0 and TRANS[1]!=0 and TRANS[2]!=0:
        printf("No LTRANS=\"%s\" option!\n",LTRANS)
        sys.exit()

    starttime = mpi.MPI_Wtime()
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)
    # get shm_comm
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    shm_comm_py = comm.Split_type(MPI.COMM_TYPE_SHARED)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&node_comm)

    n_p = nq
    nk_a = nq
    k_range[1] = nq

    mpi.MPI_Bcast(&energy_buf[0,0],nk_a*nbands,mpi.MPI_DOUBLE,0,node_comm)
    # alloc phonon
    nmodes_l = nmodes
    if LPHSHM:
        if (shm_id==0):
            l_phzero = nmodes_l*nq*2*sizeof(int)
        else:
            l_phzero = 0
        win00 = MPI.Win.Allocate_shared(l_phzero,sizeof(int),comm=shm_comm_py)
        buf00,s_int = win00.Shared_query(0)
        phzero = np.ndarray(buffer=buf00,dtype=np.int32,shape=(nmodes_l*nq,2))
        mpi.MPI_Barrier(shm_comm)
        if (myid==0):
            phonon_buf = np.load(phname)
            phzero_num = 0
            for i in range(nq):
                for j in range(nmodes):
                    if phonon_buf[i,j]<=0:
                        phonon[j,i] = 1e-13
                        phzero[phzero_num,0] = i
                        phzero[phzero_num,1] = j
                        phzero_num += 1
                    elif phonon_buf[i,j]<PHCUT:
                        phonon[j,i] = phonon_buf[i,j]
                        phzero[phzero_num,0] = i
                        phzero[phzero_num,1] = j
                        phzero_num += 1
                    else:
                        phonon[j,i] = phonon_buf[i,j]
            phonon_buf = None

        mpi.MPI_Bcast(&phzero_num,1,mpi.MPI_INT,0,c_comm)
        mpi.MPI_Bcast(&phzero[0,0],phzero_num*2,mpi.MPI_INT,0,node_comm)
        mpi.MPI_Bcast(&phonon[0,0],nmodes*nq,mpi.MPI_DOUBLE,0,node_comm)
    else:
        phzero = np.zeros((nmodes*nq,2),dtype=np.int32)
        if myid == 0:
            phonon_buf = np.load(phname)
            phzero_num = 0
            for i in range(nq):
                for j in range(nmodes):
                    if phonon_buf[i,j]<=0:
                        phonon[j,i] = 1e-13
                        phzero[phzero_num,0] = i
                        phzero[phzero_num,1] = j
                        phzero_num += 1
                    elif phonon_buf[i,j]<PHCUT:
                        phonon[j,i] = phonon_buf[i,j]
                        phzero[phzero_num,0] = i
                        phzero[phzero_num,1] = j
                        phzero_num += 1
                    else:
                        phonon[j,i] = phonon_buf[i,j]
            phonon_buf = None
       
        mpi.MPI_Bcast(&phzero_num,1,mpi.MPI_INT,0,c_comm)
        mpi.MPI_Bcast(&phzero[0,0],phzero_num*2,mpi.MPI_INT,0,c_comm)
        mpi.MPI_Bcast(&phonon[0,0],nmodes*nq,mpi.MPI_DOUBLE,0,c_comm)

    energy = np.zeros((nk_s),dtype=np.float64)
    ekidx = np.zeros((nk_s),dtype=np.int32)
    ebidx = np.zeros((nk_s),dtype=np.int32)
    k = 0
    for i in range(nk_a):
        for j in range(nbands):
            en = energy_buf[i,j]
            if (en>=EMIN and en<=EMAX):
                energy[k] = en
                ekidx[k] = i
                ebidx[k] = j
                k += 1

    k_proc = np.zeros((nprocs+1),dtype=np.int32)
    k_proc_num = np.zeros((nprocs),dtype=np.int32)
    for i in range(nprocs):
        k_proc[i+1] = ((i+1)*nk_s)/nprocs
        k_proc_num[i] = k_proc[i+1]-k_proc[i]
    nk_proc = k_proc_num[myid]
    if LEPCSHM: k_proc_min = k_proc[myid]
    else: k_proc_min = 0
    kqidx_p = np.zeros((nk_proc,nk_s),dtype=np.int32)
    k_list_a = np.zeros((nq,3),dtype=np.float64)
    for i in range(nqx):
        for j in range(nqy):
            for k in range(nqz):
                l = (i*nqy+j)*nqz+k
                k_list_a[l,0] = (<double>i)/(<double>nqx)
                k_list_a[l,1] = (<double>j)/(<double>nqy)
                k_list_a[l,2] = (<double>k)/(<double>nqz)
    KProcSplit(
        c_comm,myid,nprocs,1,nk_s,nq,nqx,nqy,nqz,nbands,
        ekidx,ebidx,k_range,k_list_a,k_list_a,k_proc,
        k_proc_num,nk_proc,k_proc_min,kqidx_p,k1qidx_p,&k_p_r
    )
    k_list_a = None

    # unit conversion factor # meV to eV
    if TRANS[2] == 0:
        f_r = 0.5
    else:
        f_r = 1.0

    l_epc_t = nq*nmodes_l*nbands*nbands
    epc_t = np.zeros((nq,nmodes,nbands,nbands),dtype=np.complex128)

    kidx = 0
    koffset = 0
    fp = fopen(epcname,'rb')
    fseek(fp,0,SEEK_SET)
    for j in range(k_p_r[0][0][2]):
        fseek(fp,(k_p_r[0][1][j]-koffset)*l_epc_t*16,SEEK_CUR)
        fread(&epc_t[0,0,0,0],sizeof(double complex),l_epc_t,fp)
        koffset = k_p_r[0][1][j]+1

        for k in range(phzero_num):
            phzero0 = phzero[k,0]
            phzero1 = phzero[k,1]
            for l in range(nbands):
                for m in range(nbands):
                    epc_t[phzero0,phzero1,l,m] = 0

        n = k_p_r[0][2][j]
        k_p_r0 = &(k_p_r[0][3][j*nbands])
        for m in range(nk_s):
            kqidx = kqidx_p[kidx,m]
            eb = ebidx[m]
            for k in range(nmodes):
                for l in range(n):
                    epc_a[k,k_proc_min+kidx+l,m] \
                    = epc_t[kqidx,k,k_p_r0[l],eb]

        kidx += n

    fclose(fp)

    TransposeBlock(
        c_comm,nprocs,myid,nmodes,nm_block,nk_s,
        k_proc_min,epc_a,k_proc,k_proc_num,f_r,TRANS
    )
    if LEPCSHM:
        AllGather(
            c_comm,nprocs,shm_nprocs,myid,nmodes,
            nk_s,k_proc_num,k1qidx_p,epc_a
        )

    free(k_range)
    for j in range(4):
        free(k_p_r[0][j])
    free(k_p_r[0])
    free(k_p_r)

    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("Read epc time: %.6fs.\n",endtime-starttime)

    return nk_a,n_p,nbands,ekidx,ebidx,\
           k_proc,k_proc_num,energy


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void MakePhZero(
    int nk_s, int nk_p, int k_p, int nm_num, int nm_min, int nm_buffer,
    int[::1] k_proc, int[::1] ekidx, int[:,::1] kqidx, 
    double PHCUT, double[:,::1] phonon, double complex * epc_t
):
    cdef int h, i, j, k, l, m, n, qidx
    cdef int ekset, n_ekset_a, n_ekset
    cdef long epcidx
    cdef int * ekidx_set_a = <int*>malloc(nk_s*sizeof(int))
    cdef int * ekidx_count_a = <int*>calloc(nk_s,sizeof(int))
    cdef int * ekidx_set = <int*>malloc(nk_p*sizeof(int))
    cdef int * ekidx_count = <int*>calloc(nk_p,sizeof(int))

    n_ekset_a = -1
    ekset = -1
    for i in range(nk_s):
        if ekidx[i] == ekset:
            ekidx_count_a[n_ekset_a] += 1
        else:
            ekset = ekidx[i]
            n_ekset_a += 1
            ekidx_set_a[n_ekset_a] = ekset
            ekidx_count_a[n_ekset_a] += 1
    n_ekset_a += 1

    n_ekset = -1
    ekset = -1
    for i in range(k_p,k_p+nk_p):
        if ekidx[i] == ekset:
            ekidx_count[n_ekset] += 1
        else:
            ekset = ekidx[i]
            n_ekset += 1
            ekidx_set[n_ekset] = ekset
            ekidx_count[n_ekset] += 1
    n_ekset += 1

    k = 0
    for i in range(n_ekset):
        l = 0
        for j in range(n_ekset_a):
            for m in range(ekidx_count[i]):
                for n in range(ekidx_count_a[j]):
                    qidx = kqidx[k+m,l+n]
                    epcidx = (k+m)*nk_s+(l+n)
                    for h in range(nm_num):
                        if (phonon[h+nm_min,qidx]<PHCUT):
                            epc_t[epcidx*nm_buffer+h] = 0.0

            l += ekidx_count_a[j]
        k += ekidx_count[i]

    free(ekidx_set_a)
    free(ekidx_count_a)
    free(ekidx_set)
    free(ekidx_count)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void KProcSplitPart(
    mpi.MPI_Comm comm, int myid, int nprocs, int nk_s, int nq, 
    int nqx, int nqy, int nqz, int[::1] ekidx, int[::1] ebidx, 
    double[:,::1] k_list_a, double[:,::1] q_list, int[::1] k_proc, 
    int[::1] k_proc_num, int k_p_m, int[:,::1] kqidx_p, int[:,::1] k1qidx_p
):
    cdef mpi.MPI_Datatype INT2
    cdef int k_proc_min = k_proc[myid]
    cdef int k_proc_max = k_proc[myid+1]
    cdef int nk_proc = k_proc_num[myid]
    cdef int i, j, k, l, ekset, n_ekset_a, n_ekset
    cdef int * ekidx_set_a = <int*>malloc(nk_s*sizeof(int))
    cdef int * ekidx_count_a = <int*>calloc(nk_s,sizeof(int))
    cdef int * ekidx_set = <int*>malloc(nk_proc*sizeof(int))
    cdef int * ekidx_count = <int*>calloc(nk_proc,sizeof(int))

    n_ekset_a = -1
    ekset = -1
    for i in range(nk_s):
        if ekidx[i] == ekset:
            ekidx_count_a[n_ekset_a] += 1
        else:
            ekset = ekidx[i]
            n_ekset_a += 1
            ekidx_set_a[n_ekset_a] = ekset
            ekidx_count_a[n_ekset_a] += 1
    n_ekset_a += 1

    n_ekset = -1
    ekset = -1
    for i in range(k_proc_min,k_proc_max):
        if ekidx[i] == ekset:
            ekidx_count[n_ekset] += 1
        else:
            ekset = ekidx[i]
            n_ekset += 1
            ekidx_set[n_ekset] = ekset
            ekidx_count[n_ekset] += 1
    n_ekset += 1

    GetKqidx(
        myid,nq,k_p_m,k_list_a,q_list,nqx,nqy,nqz,
        n_ekset_a,n_ekset,ekidx_set_a,ekidx_count_a,
        ekidx_set,ekidx_count,kqidx_p,k1qidx_p
    )

    free(ekidx_set_a)
    free(ekidx_count_a)
    free(ekidx_set)
    free(ekidx_count)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void TransposePartBlock(
    mpi.MPI_Comm c_comm, int nprocs, int myid, int nmodes, 
    int nm_block, int nk_a, int k_p_m, double complex[:,:,::1] epc_a,
    int[::1] k_proc, int[::1] k_proc_num, int[::1] ekidx,
    int[:,::1] kqidx, double PHCUT, double[:,::1] phonon,
    double f, int * TRANS, char * epcname
):
    cdef int h, i, j, k, l, m, nm_min, nm_max, nm_num, nm_loop, \
             nm_buffer, k_proc_i, nk_proc, idx0, idx1, idx2, idx3
    cdef int k_p = k_proc[myid]
    cdef int nk_p = k_proc_num[myid]
    cdef long l_epc_set, l_epc_buf
    cdef int * scount = <int*>malloc(nprocs*sizeof(int))
    cdef int * sdispl = <int*>malloc(nprocs*sizeof(int))
    cdef int * rcount = <int*>malloc(nprocs*sizeof(int))
    cdef int * rdispl = <int*>malloc(nprocs*sizeof(int))
    cdef double complex * epc_sbuf
    cdef double complex * epc_rbuf
    cdef double complex * epc_t
    cdef double complex * epc_buf
    cdef double complex epc_t0
    cdef FILE * fp = fopen(epcname,'rb')

    nm_loop = <int>(nmodes/nm_block)
    nm_buffer = nm_block
    if nmodes%nm_block != 0:
        nm_loop += 1
        nm_buffer += 1
    epc_sbuf = <double complex*>malloc(\
        (nk_a*nk_p*nm_buffer)*sizeof(double complex))
    epc_rbuf = <double complex*>malloc(\
        (nk_p*nk_a*nm_buffer)*sizeof(double complex))
    for j in range(nprocs):
        scount[j] = nm_buffer*nk_p*k_proc_num[j]
        rcount[j] = nm_buffer*nk_p*k_proc_num[j]
        sdispl[j] = nm_buffer*nk_p*k_proc[j]
        rdispl[j] = nm_buffer*nk_p*k_proc[j]

    l_epc_set = k_p*nk_a*<long>(nmodes)
    l_epc_buf = nk_p*nk_a*<long>(nm_buffer)
    epc_t = <double complex*>malloc(l_epc_buf*sizeof(double complex))
    epc_buf = <double complex*>malloc(nk_a*nmodes*sizeof(double complex))

    for h in range(nm_loop):
        nm_min = <int>((nmodes*h)/nm_loop)
        nm_max = <int>((nmodes*(h+1))/nm_loop)
        nm_num = nm_max - nm_min
        # read epc
        fseek(fp,l_epc_set*sizeof(double complex),SEEK_SET)
        for i in range(nk_p):
            fread(epc_buf,sizeof(double complex),nk_a*nmodes,fp)
            for j in range(nk_a):
                for k in range(nm_num):
                    epc_t[(i*nk_a+j)*nm_buffer+k] \
                    = epc_buf[j*nmodes+k+nm_min]
        # make phonon zero
        MakePhZero(
            nk_a,nk_p,k_p,nm_num,nm_min,nm_buffer,
            k_proc,ekidx,kqidx,PHCUT,phonon,epc_t
        )
        # Only keep Upper/Lower part
        # prepare send buffer
        memset(epc_sbuf,0,(nk_a*nk_p*nm_buffer)*sizeof(double complex))
        if TRANS[0] == 0:
            for j in range(nk_p):
                for k in range(k_p+j+1):
                    for i in range(nm_num):
                        l = (j*nk_a+k)*nm_buffer+i
                        epc_t0 = epc_t[l]*f
                        epc_t[l] = conj(epc_t0)
                        epc_sbuf[(k*nk_p+j)*nm_buffer+i] = epc_t0
                for k in range(k_p+j+1,nk_a):
                    for i in range(nm_num):
                        l = (j*nk_a+k)*nm_buffer+i
                        epc_t[l] = 0
        elif TRANS[1] == 0:
            for j in range(nk_p):
                for k in range(k_p+j):
                    for i in range(nm_num):
                        l = (j*nk_a+k)*nm_buffer+i
                        epc_t[l] = 0
                for k in range(k_p+j,nk_a):
                    for i in range(nm_num):
                        l = (j*nk_a+k)*nm_buffer+i
                        epc_t0 = epc_t[l]*f
                        epc_t[l] = conj(epc_t0)
                        epc_sbuf[(k*nk_p+j)*nm_buffer+i] = epc_t0
        else:
            for j in range(nk_p):
                for k in range(nk_a):
                    for i in range(nm_num):
                        l = (j*nk_a+k)*nm_buffer+i
                        epc_t0 = epc_t[l]*f
                        epc_t[l] = conj(epc_t0)
                        epc_sbuf[(k*nk_p+j)*nm_buffer+i] = epc_t0
        mpi.MPI_Alltoallv(
            epc_sbuf,scount,sdispl,
            mpi.MPI_DOUBLE_COMPLEX,epc_rbuf,rcount,rdispl,
            mpi.MPI_DOUBLE_COMPLEX,c_comm
        )

        for i in range(nprocs):
            k_proc_i = k_proc[i]
            nk_proc = k_proc_num[i]
            idx0 = rdispl[i]
            for j in range(nk_p):
                for k in range(nk_proc):
                    idx1 = j*nk_proc+k
                    for l in range(nm_num):
                        epc_t[(j*nk_a+k+k_proc_i)*nm_buffer+l]\
                        += epc_rbuf[idx0+idx1*nm_buffer+l]

        if TRANS[2] != 0:
            for k in range(nm_num):
                for i in range(nk_p):
                    epc_t[(i*nk_a+i+k_p)*nm_buffer+k] /= 2.0
    
        for k in range(nm_num):
            for i in range(nk_p):
                l = (i*nk_a+i+k_p)*nm_buffer+k
                epc_t[l] = cabs(epc_t[l])
    
        for i in range(nm_num):
            for j in range(nk_p):
                for k in range(nk_a):
                    epc_a[i+nm_min,j+k_p_m,k] \
                    = epc_t[(j*nk_a+k)*nm_buffer+i]

    fclose(fp)
    free(epc_buf)
    free(epc_t)
    free(epc_sbuf)
    free(epc_rbuf)
    free(scount)
    free(sdispl)
    free(rcount)
    free(rdispl)


@cython.boundscheck(False)
@cython.wraparound(False)
def ReadNpyPart(
    MPI.Comm comm, int nmodes, int nm_block, int nbands, int nqx, 
    int nqy, int nqz, int nk_s, int[::1] ekidx, int[::1] ebidx, 
    double PHCUT, double EMIN, double EMAX, int[:,::1] k1qidx_p, 
    double[:,::1] energy_buf, double[::1] energy, double[:,::1] phonon, 
    double complex[:,:,::1] epc_a, char * LTRANS, str phname, 
    char * epcname, bint LEPCSHM, bint LPHSHM
):
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef mpi.MPI_Comm shm_comm, node_comm
    cdef mpi.MPI_Group split_gp, shm_gp
    cdef int myid, shm_id, nprocs, shm_nprocs, ierr, \
             i, j, k, l, n_p, nk_a, nk_proc, \
             k_proc_min, nnode, node_id
    cdef int * nodelist
    cdef int s_int = sizeof(int)
    cdef int nq = nqx*nqy*nqz
    cdef double[:,::1] phonon_buf, k_list_a
    cdef int[::1] k_proc, k_proc_num
    cdef int[:,::1] kqidx_p
    cdef double starttime, endtime, en, f_r
    cdef int TRANS[3]

    TRANS[0] = strcmp(LTRANS,"U")
    TRANS[1] = strcmp(LTRANS,"L")
    TRANS[2] = strcmp(LTRANS,"S")
    if TRANS[0]!=0 and TRANS[1]!=0 and TRANS[2]!=0:
        printf("No LTRANS=\"%s\" option!\n",LTRANS)
        sys.exit()

    starttime = mpi.MPI_Wtime()
    ierr = mpi.MPI_Comm_size(c_comm,&nprocs)
    ierr = mpi.MPI_Comm_rank(c_comm,&myid)
    # get shm_comm
    mpi.MPI_Comm_split_type(
        c_comm,mpi.MPI_COMM_TYPE_SHARED,0,mpi.MPI_INFO_NULL,&shm_comm
    )
    ierr = mpi.MPI_Comm_rank(shm_comm,&shm_id)
    ierr = mpi.MPI_Comm_size(shm_comm,&shm_nprocs)
    shm_comm_py = comm.Split_type(MPI.COMM_TYPE_SHARED)
    # create internode group/comm
    nnode = nprocs/shm_nprocs
    node_id = myid/shm_nprocs
    nodelist = <int*>malloc(s_int*nnode)
    for i in range(nnode):
        nodelist[i] = i*shm_nprocs
    mpi.MPI_Comm_group(c_comm,&split_gp)
    mpi.MPI_Group_incl(split_gp,nnode,nodelist,&shm_gp)
    ierr = mpi.MPI_Comm_create(c_comm,shm_gp,&node_comm)

    n_p = nq
    nk_a = nq

    mpi.MPI_Bcast(&energy_buf[0,0],nk_a*nbands,mpi.MPI_DOUBLE,0,node_comm)
    # alloc phonon
    if (myid==0):
        phonon_buf = np.load(phname)
        for i in range(nq):
            for j in range(nmodes):
                if phonon_buf[i,j]<=0:
                    phonon[j,i] = 1e-13
                else:
                    phonon[j,i] = phonon_buf[i,j]
        phonon_buf = None
    if LPHSHM:
        mpi.MPI_Bcast(&phonon[0,0],nmodes*nq,mpi.MPI_DOUBLE,0,node_comm)
        mpi.MPI_Barrier(shm_comm)
    else:
        mpi.MPI_Bcast(&phonon[0,0],nmodes*nq,mpi.MPI_DOUBLE,0,c_comm)

    k_proc = np.zeros((nprocs+1),dtype=np.int32)
    k_proc_num = np.zeros((nprocs),dtype=np.int32)
    for i in range(nprocs):
        k_proc[i+1] = ((i+1)*nk_s)/nprocs
        k_proc_num[i] = k_proc[i+1]-k_proc[i]
    nk_proc = k_proc_num[myid]
    if LEPCSHM: k_proc_min = k_proc[myid]
    else: k_proc_min = 0
    kqidx_p = np.zeros((nk_proc,nk_s),dtype=np.int32)
    k_list_a = np.zeros((nq,3),dtype=np.float64)
    for i in range(nqx):
        for j in range(nqy):
            for k in range(nqz):
                l = (i*nqy+j)*nqz+k
                k_list_a[l,0] = (<double>i)/(<double>nqx)
                k_list_a[l,1] = (<double>j)/(<double>nqy)
                k_list_a[l,2] = (<double>k)/(<double>nqz)
    KProcSplitPart(
        c_comm,myid,nprocs,nk_s,nq,nqx,nqy,nqz,
        ekidx,ebidx,k_list_a,k_list_a,k_proc,
        k_proc_num,k_proc_min,kqidx_p,k1qidx_p
    )
    k_list_a = None

    # unit conversion factor # meV to eV
    if TRANS[2] == 0:
        f_r = 0.5
    else:
        f_r = 1.0

    TransposePartBlock(
        c_comm,nprocs,myid,nmodes,nm_block,nk_s,
        k_proc_min,epc_a,k_proc,k_proc_num,ekidx,
        kqidx_p,PHCUT,phonon,f_r,TRANS,epcname
    )
    if LEPCSHM:
        AllGather(
            c_comm,nprocs,shm_nprocs,myid,nmodes,
            nk_s,k_proc_num,k1qidx_p,epc_a
        )

    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("Read epc time: %.6fs.\n",endtime-starttime)

    return nk_a,n_p,nbands,ekidx,ebidx,\
           k_proc,k_proc_num,energy
