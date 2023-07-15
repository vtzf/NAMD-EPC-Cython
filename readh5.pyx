#cython: language_level=3
#cython: cdivision=True

import sys
cimport cython
import numpy as np
cimport numpy as np
from mpi4py import MPI
from mpi4py cimport MPI
from mpi4py cimport libmpi as mpi
from libc.math cimport fmod, round
from libc.stdio cimport sprintf, printf
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
cdef void GetKqidx(
    int myid, int nk, int nq, 
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
                    k1qidx[k+m,l+n] = q2qmap[idx_t1*2+1]

            l += ekidx_count_a[j]
        k += ekidx_count[i]

    free(q2qmap)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void KProcSplit(
    int myid, int nprocs, int Num, int nk_s, int nq, 
    int nqx, int nqy, int nqz, int nbands, int[::1] ekidx,
    int[::1] ebidx, int * k_range, double[:,::1] k_list_a,
    double[:,::1] q_list, int[::1] k_proc, int[::1] k_proc_num, int nk_proc,
    int[:,::1] kqidx_p, int[:,::1] k1qidx_p, int **** k_proc_range
):
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
        myid,nk_s,nq,k_list_a,q_list,nqx,nqy,nqz,
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
cdef void Transpose(
    mpi.MPI_Comm c_comm, int nprocs, int myid, 
    int nmodes, int nk_a, double complex[:,:,::1] epc_a,
    int[::1] k_proc, int[::1] k_proc_num, double f, int * TRANS
):
    cdef int i, j, k, l, \
             k_proc_i, nk_proc, idx0, idx1, idx2, idx3
    cdef int k_p = k_proc[myid]
    cdef int nk_p = k_proc_num[myid]
    cdef int * scount = <int*>malloc(nprocs*sizeof(int))
    cdef int * sdispl = <int*>malloc(nprocs*sizeof(int))
    cdef int * rcount = <int*>malloc(nprocs*sizeof(int))
    cdef int * rdispl = <int*>malloc(nprocs*sizeof(int))
    cdef double complex * epc_sbuf
    cdef double complex * epc_rbuf
    cdef double complex epc_t

    # Only keep Upper/Lower part
    # prepare send buffer
    epc_sbuf = <double complex*>malloc(\
        (nk_a*nk_p*nmodes)*sizeof(double complex))
    if TRANS[0] == 0:
        for i in range(nmodes):
            for j in range(nk_p):
                for k in range(k_p+j+1):
                    epc_t = epc_a[i,j,k]*f
                    epc_a[i,j,k] = conj(epc_t)
                    epc_sbuf[(k*nk_p+j)*nmodes+i] = epc_t
                for k in range(k_p+j+1,nk_a):
                    epc_a[i,j,k] = 0
    elif TRANS[1] == 0:
        for i in range(nmodes):
            for j in range(nk_p):
                for k in range(k_p+j):
                    epc_a[i,j,k] = 0
                for k in range(k_p+j,nk_a):
                    epc_t = epc_a[i,j,k]*f
                    epc_a[i,j,k] = conj(epc_t)
                    epc_sbuf[(k*nk_p+j)*nmodes+i] = epc_t
    else:
        for i in range(nmodes):
            for j in range(nk_p):
                for k in range(nk_a):
                    epc_t = epc_a[i,j,k]*f
                    epc_a[i,j,k] = conj(epc_t)
                    epc_sbuf[(k*nk_p+j)*nmodes+i] = epc_t
    mpi.MPI_Barrier(c_comm)
    epc_rbuf = <double complex*>malloc(\
        (nk_p*nk_a*nmodes)*sizeof(double complex))
    for j in range(nprocs):
        scount[j] = nmodes*nk_p*k_proc_num[j]
        rcount[j] = nmodes*nk_p*k_proc_num[j]
        sdispl[j] = nmodes*nk_p*k_proc[j]
        rdispl[j] = nmodes*nk_p*k_proc[j]
    mpi.MPI_Alltoallv(
        epc_sbuf,scount,sdispl,
        mpi.MPI_DOUBLE_COMPLEX,epc_rbuf,rcount,rdispl,
        mpi.MPI_DOUBLE_COMPLEX,c_comm
    )
    free(epc_sbuf)

    for i in range(nprocs):
        k_proc_i = k_proc[i]
        nk_proc = k_proc_num[i]
        idx0 = rdispl[i]
        for j in range(nk_p):
            for k in range(nk_proc):
                idx1 = j*nk_proc+k
                for l in range(nmodes):
                    epc_a[l,j,k+k_proc_i]\
                    += epc_rbuf[idx0+idx1*nmodes+l]
    free(epc_rbuf)
    free(scount)
    free(sdispl)
    free(rcount)
    free(rdispl)

    if TRANS[2] != 0:
        for k in range(nmodes):
            for i in range(nk_p):
                epc_a[k,i,i+k_p] /= 2.0

    for k in range(nmodes):
        for i in range(nk_p):
            epc_a[k,i,i+k_p] = cabs(epc_a[k,i,i+k_p])


@cython.boundscheck(False)
@cython.wraparound(False)
def ReadH5(
    MPI.Comm comm, char * Dir, int Num, double PHCUT, 
    double EMIN, double EMAX, int nqx, int nqy, int nqz,
    char * LTRANS
):
    cdef hid_t file_id, data_id
    cdef herr_t status
    cdef mpi.MPI_Comm c_comm = comm.ob_mpi
    cdef int myid, nprocs, ierr, i, j, k, l, \
             m, n, p, q, kidx, fnum, kqidx, eb,\
             phzero_num, phzero0, phzero1,\
             nk_p, nk_a, nk_s, nk_proc, nq, nbands, nmodes
    cdef int n_p = nqx*nqy*nqz
    cdef int info[4]
    cdef int info_buf[4]
    cdef double * phonon_buf
    cdef double * energy_buf
    cdef double[:,::1] phonon, k_list_a, q_list
    cdef double[::1] energy
    cdef int * phzero
    cdef int * k_range = <int*>calloc(Num+1,sizeof(int))
    cdef int * eidx
    cdef int[::1] ekidx, ebidx, k_proc, k_proc_num
    cdef int[:,::1] kqidx_p, k1qidx_p
    cdef int *** k_p_r
    cdef int * k_p_r0
    cdef double[:,:,:,::1] epc_p_r, epc_p_i
    cdef double complex[:,:,::1] epc_a
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

    phonon_buf = <double*>malloc(sizeof(double)*nq*nmodes)
    phzero = <int*>malloc(sizeof(int)*nmodes*nq*2)
    phonon = np.zeros((nmodes,nq),dtype=np.float64)
    data_id = H5Dopen(file_id,"el_ph_band_info/ph_disp_meV",H5P_DEFAULT)
    status = H5Dread(
        data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,phonon_buf
    )
    H5Dclose(data_id)
    q_list = np.zeros((nq,3),dtype=np.float64)
    data_id = H5Dopen(file_id,"el_ph_band_info/q_list",H5P_DEFAULT)
    status = H5Dread(
        data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,&q_list[0,0]
    )
    H5Dclose(data_id)

    phzero_num = 0
    for i in range(nq):
        for j in range(nmodes):
            k = i*nmodes+j
            if phonon_buf[k]<=0:
                phonon[j,i] = 1e-13
                phzero[phzero_num*2] = i
                phzero[phzero_num*2+1] = j
                phzero_num += 1
            elif phonon_buf[k]<PHCUT:
                phonon[j,i] = phonon_buf[k]/1000
                phzero[phzero_num*2] = i
                phzero[phzero_num*2+1] = j
                phzero_num += 1
            else:
                phonon[j,i] = phonon_buf[k]/1000

    status = H5Fclose(file_id)

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
    if (nk_a>nq):
        printf('Unmatch of q and k points exists!\n')
        sys.exit()

    energy_buf = <double*>malloc(sizeof(double)*nk_a*nbands)
    k_list_a = np.zeros((nk_a,3),dtype=np.float64)
    for i in range(Num):
        sprintf(name,"%s%d.h5",Dir,i+1)
        file_id = H5Fopen(name,H5F_ACC_RDONLY,H5P_DEFAULT)
        data_id = H5Dopen(file_id,"el_ph_band_info/el_band_eV",H5P_DEFAULT)
        status = H5Dread(
            data_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,
            H5P_DEFAULT,&energy_buf[k_range[i]*nbands]
        )
        H5Dclose(data_id)
        data_id = H5Dopen(file_id,"el_ph_band_info/k_list",H5P_DEFAULT)
        status = H5Dread(
            data_id,H5T_NATIVE_DOUBLE,H5S_ALL,
            H5S_ALL,H5P_DEFAULT,&k_list_a[k_range[i],0]
        )
        H5Dclose(data_id)
        status = H5Fclose(file_id)

    energy = np.zeros((nk_a*nbands),dtype=np.float64)
    eidx = <int*>malloc(sizeof(int)*nk_a*nbands)
    ekidx = np.zeros((nk_a*nbands),dtype=np.int32)
    ebidx = np.zeros((nk_a*nbands),dtype=np.int32)
    nk_s = 0
    for i in range(nk_a*nbands):
        en = energy_buf[i]
        if (en>=EMIN and en<=EMAX):
            energy[nk_s] = en
            eidx[nk_s] = i
            ekidx[nk_s] = i/nbands
            ebidx[nk_s] = i%nbands
            nk_s += 1

    k_proc = np.zeros((nprocs+1),dtype=np.int32)
    k_proc_num = np.zeros((nprocs),dtype=np.int32)
    for i in range(nprocs):
        k_proc[i+1] = ((i+1)*nk_s)/nprocs
        k_proc_num[i] = k_proc[i+1]-k_proc[i]
    nk_proc = k_proc_num[myid]
    kqidx_p = np.zeros((nk_proc,nk_s),dtype=np.int32)
    k1qidx_p = np.zeros((nk_proc,nk_s),dtype=np.int32)
    KProcSplit( 
        myid,nprocs,Num,nk_s,nq,nqx,nqy,nqz,nbands,
        ekidx,ebidx,k_range,k_list_a,q_list,k_proc,
        k_proc_num,nk_proc,kqidx_p,k1qidx_p,&k_p_r
    )

    # unit conversion factor # meV to eV
    if TRANS[2] == 0:
        f_r = 1.0/2000.0
    else:
        f_r = 1.0/1000.0

    fnum = k_p_r[0][0][0]
    epc_a = np.zeros((nmodes,nk_proc,nk_s),dtype=np.complex128)
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
                phzero0 = phzero[k*2]
                phzero1 = phzero[k*2+1]
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
                        epc_a[k,kidx+l,m] \
                        = epc_p_r[kqidx,k,k_p_r0[l],eb] \
                        + epc_p_i[kqidx,k,k_p_r0[l],eb]*1j
            
            kidx += n

        status = H5Fclose(file_id)

    Transpose(
        c_comm,nprocs,myid,nmodes,nk_s,
        epc_a,k_proc,k_proc_num,f_r,TRANS
    )

    free(k_range)
    free(phonon_buf)
    free(phzero)
    free(energy_buf)
    free(eidx)

    for i in range(fnum):
        for j in range(4):
            free(k_p_r[i][j])
        free(k_p_r[i])
    free(k_p_r)

    endtime = mpi.MPI_Wtime()
    if myid == 0:
        printf("Read epc time: %.6fs.\n",endtime-starttime)

    return nk_a,nk_s,nq,n_p,nmodes,nbands,ekidx,ebidx,\
           k1qidx_p,k_proc,k_proc_num,energy,phonon,epc_a
