#!/usr/bin/env python
#-*- encoding:utf-8 -*-
import Args
import readh5
import readepc
import sys
import numpy as np
import mpi4py
from mpi4py import MPI
import os
import CalFunc
#import CalEFunc

def SurfHop():
    # INICON read
    inicon = np.loadtxt('INICON',dtype=np.int32)-1
    if len(inicon.shape) == 1:
        inicon = inicon.reshape(1,-1)
    if inicon.shape[0] < Args.NSAMPLE:
        print('Sample number in INICON is samller than NSAMPLE!')
        sys.exit()

    nel = (inicon.shape[1]-1)//2
    step_s = inicon[:,0]

    # get world comm and split comm to shm_comm
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    nprocs = comm.Get_size()
    shm_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    nprocs_shm = shm_comm.Get_size()
    shm_num = nprocs//nprocs_shm

    # get split num = GCD(shm_num,NSAMPLE)
    split_num = 1
    tmp1 = min(shm_num,Args.NSAMPLE)
    tmp2 = max(shm_num,Args.NSAMPLE)
    while (1):
        tmp2 = tmp2 - tmp1
        if tmp2 == 0:
            split_num = tmp1
            break
        else:
            tmp3 = min(tmp1,tmp2)
            tmp4 = max(tmp1,tmp2)
            tmp1 = tmp3
            tmp2 = tmp4
    iprocs = nprocs//split_num

    # split comm
    color= myid//iprocs
    isample = Args.NSAMPLE//split_num
    comm_split = comm.Split(color=color,key=myid)
    # comm_split rank_id and nprocs
    myid_split = comm_split.Get_rank()

    for i in range(split_num):
        if color == i:
            # get shm_comm
            shm_comm = comm_split.Split_type(MPI.COMM_TYPE_SHARED)
            shm_id = shm_comm.Get_rank()
            # get info
            if not Args.LHDF5:
                nk_s,nk_a,nmodes,nbands,kmin_s,kmax_s,bmin_s,bmax_s,abc = \
                readepc.ReadEpcInfo(
                    comm_split,myid_split,
                    Args.nqx,Args.nqy,Args.nqz,Args.EMIN,Args.EMAX
                )
                if myid_split == 0:
                    print(kmin_s,kmax_s,bmin_s,bmax_s,flush=True)
            else:
                nk_s,nk_a,nmodes,nbands,kmin_s,kmax_s,bmin_s,bmax_s,abc = \
                readh5.ReadH5Info(
                    comm_split,myid_split,
                    (Args.EPMDIR+'/'+Args.EPMPREF).encode('utf-8'),
                    Args.NPARTS,Args.nqx,Args.nqy,
                    Args.nqz,Args.EMIN,Args.EMAX
                )
            # alloc k1kidxE, energy_a
            nq = Args.nqx*Args.nqy*Args.nqz
            if (nk_a!=nq and Args.LEF):
                print('Energy grid incomplete under electric field!')
                sys.exit()
            if (shm_id==0):
                l_k1kidxE = nq*2*4
                l_energy_a = nk_a*nbands*8
            else:
                l_k1kidxE = 0
                l_energy_a = 0
            win2 = MPI.Win.Allocate_shared(l_k1kidxE,4,comm=shm_comm)
            buf2,s_int = win2.Shared_query(0)
            k1kidxE = np.ndarray(
                buffer=buf2,dtype=np.int32,shape=(nq,2)
            )
            win3 = MPI.Win.Allocate_shared(l_energy_a,8,comm=shm_comm)
            buf3,s_double = win3.Shared_query(0)
            energy_a = np.ndarray(
                buffer=buf3,dtype=np.float64,shape=(nk_a,nbands)
            )
            if (shm_id==0):
                readh5.GetKqidxEF(
                    myid_split,Args.nqx,Args.nqy,Args.nqz,abc,Args.dt,
                    Args.hbar,Args.EFX,Args.EFY,Args.EFZ,k1kidxE
                )
            # alloc kqidx, epc_a
            if Args.LEPCSHM:
                if (shm_id==0):
                    l_kqidx = nk_s*nk_s*4
                    l_epc = nk_s*nk_s*nmodes*16
                else:
                    l_kqidx = 0
                    l_epc = 0
                win1 = MPI.Win.Allocate_shared(l_kqidx,4,comm=shm_comm)
                buf1,s_int = win1.Shared_query(0)
                kqidx = np.ndarray(
                    buffer=buf1,dtype=np.int32,shape=(nk_s,nk_s)
                )
                win = MPI.Win.Allocate_shared(l_epc,16,comm=shm_comm)
                buf,s_dcplx = win.Shared_query(0)
                epc_a = np.ndarray(
                    buffer=buf,dtype=np.complex128,
                    shape=(nmodes,nk_s,nk_s)
                )
            else:
                nk_min = (myid_split*nk_s)//iprocs
                nk_max = ((myid_split+1)*nk_s)//iprocs
                nk_proc = nk_max-nk_min
                kqidx = np.zeros((nk_proc,nk_s),dtype=np.int32)
                epc_a = np.zeros((nmodes,nk_proc,nk_s),dtype=np.complex128)
            # alloc phonon
            if Args.LPHSHM:
                if (shm_id==0):
                    l_phonon = nmodes*nq*8
                else:
                    l_phonon = 0
                win0 = MPI.Win.Allocate_shared(l_phonon,8,comm=shm_comm)
                buf0,s_double = win0.Shared_query(0)
                phonon = np.ndarray(
                    buffer=buf0,dtype=np.float64,shape=(nmodes,nq)
                )
            else:
                phonon = np.zeros((nmodes,nq),dtype=np.float64)
            comm_split.Barrier()
            # readh5
            if not Args.LHDF5:
                nk,n_p,nbands,ekidx,ebidx,\
                k_proc,k_proc_num,energy\
                = readepc.ReadEpc(
                    comm_split,Args.PHCUT/1000.0,Args.EMIN,Args.EMAX,
                    Args.NM_BLOCK,nk_s,kqidx,energy_a,phonon,epc_a,
                    Args.LTRANS,Args.LEPCSHM,Args.LPHSHM
                )
            else:
                nk,n_p,nbands,ekidx,ebidx,\
                k_proc,k_proc_num,energy\
                = readh5.ReadH5(
                    comm_split,
                    (Args.EPMDIR+'/'+Args.EPMPREF).encode('utf-8'),
                    Args.NPARTS,nmodes,Args.NM_BLOCK,nk_s,Args.PHCUT,
                    Args.EMIN,Args.EMAX,Args.nqx,Args.nqy,Args.nqz,
                    Args.LTRANS.encode('utf-8'),Args.LEPCSHM,
                    Args.LPHSHM,kqidx,energy_a,phonon,epc_a
                )
            #np.save('epc-%d-%d.npy'%(myid,k_proc[myid]),epc_a)
            if color == 0 and myid_split == 0:
                #np.save('k1kidxE.npy',k1kidxE)
                Args.WriteInp(nbands,nk,n_p)
                np.save(Args.namddir+'/bassel.npy',\
                        np.vstack([ekidx,ebidx]).T)
                np.save(Args.namddir+'/energy.npy',energy)

            for j in range(i*isample,(i+1)*isample):
                starttime = MPI.Wtime()
                istep_s = step_s[j]
                ikstate_s = inicon[j,np.arange(1,nel*2+1,2)]
                ibstate_s = inicon[j,np.arange(2,nel*2+2,2)]
                istate_s = np.zeros((nel),dtype=np.int32)
                for iel in range(nel):
                    istate_s[iel] = CalFunc.GetIniIdx(\
                        ekidx,ebidx,ikstate_s[iel],ibstate_s[iel],nk_s
                    )
                    if istate_s[iel] < 0 and myid_split == 0:
                        print('Initial energy out of [EMIN,EMAX]!')
                        sys.exit()
                if Args.LEF:
                    if color == 0 and myid_split == 0:
                        print(kmin_s,kmax_s,bmin_s,bmax_s,flush=True)
#                    CalEFunc.fssh(
#                        comm_split,Args.namddir,j,istep_s,istate_s,
#                        ekidx,ebidx,Args.NTRAJ,Args.NSW,Args.NELM,Args.KbT,
#                        Args.edt,Args.hbar,Args.SIGMA,Args.dt,k_proc,k_proc_num,
#                        nk_s,nq,kmax_s-kmin_s+1,n_p,nmodes,bmax_s-bmin_s+1,bmin_s,
#                        nel,kqidx,k1kidxE,epc_a,energy,energy_a,phonon,Args.LHOLE,
#                        Args.LSPLIT,Args.LEPCSHM,Args.LPHSHM,Args.BANDDEG,
#                        Args.EFTYPE.encode('utf-8'),Args.EFSTART,Args.EFEND,Args.EFTIME
#)
                else:
                    CalFunc.fssh(
                        comm_split,Args.namddir,j,istep_s,istate_s,
                        Args.NTRAJ,Args.NSW,Args.NELM,Args.KbT,Args.edt,
                        Args.hbar,Args.SIGMA,Args.dt,k_proc,k_proc_num,
                        nk_s,nq,n_p,nmodes,nel,kqidx,epc_a,energy,
                        phonon,Args.LHOLE,Args.LSPLIT,Args.LEPCSHM,
                        Args.LPHSHM,Args.BANDDEG
                    )
                endtime = MPI.Wtime()
                if myid_split == 0:
                    print("FSSH time in sample %d: %.6fs"\
                          %(j,endtime-starttime),flush=True)

            # free
            MPI.Win.Free(win2)
            MPI.Win.Free(win3)
            if Args.LEPCSHM:
                MPI.Win.Free(win1)
                MPI.Win.Free(win)
            else:
                kqidx = None
                epc_a = None
            if Args.LPHSHM:
                MPI.Win.Free(win0)
            else:
                phonon = None
