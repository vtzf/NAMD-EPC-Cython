#!/usr/bin/env python
#-*- encoding:utf-8 -*-
import Args
import readh5
import sys
import numpy as np
import mpi4py
from mpi4py import MPI
import os
import CalFunc


def SurfHop():
    # INICON read
    inicon = np.loadtxt('INICON',dtype=np.int32).reshape(-1,3)-1
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
            nk,nk_a,nq,n_p,nmodes,nbands,ekidx,ebidx,kqidx_p,\
            k_proc,k_proc_num,energy,phonon,epc_a\
            = readh5.ReadH5(
                comm_split,
                (Args.EPMDIR+'/'+Args.EPMPREF).encode('utf-8'),
                Args.NPARTS,Args.PHCUT,Args.EMIN,Args.EMAX,
                Args.nqx,Args.nqy,Args.nqz,
                Args.LTRANS.encode('utf-8')
            )
            if color == 0 and myid_split == 0:
                Args.WriteInp(nbands,nk,n_p)
                np.save(Args.namddir+'/bassel.npy',\
                        np.vstack([ekidx[0:nk_a],ebidx[0:nk_a]]).T)
            nk_proc = k_proc_num[myid_split]

            for j in range(i*isample,(i+1)*isample):
                starttime = MPI.Wtime()
                istep_s = step_s[j]
                ikstate_s = inicon[j,1]
                ibstate_s = inicon[j,2]
                istate_s = CalFunc.GetIniIdx(ekidx,ebidx,ikstate_s,ibstate_s,nk_a)
                if istate_s < 0 and myid_split == 0:
                    print('Initial energy out of [EMIN,EMAX]!')
                    sys.exit()
                CalFunc.fssh(
                    comm_split,Args.namddir,j,istep_s,istate_s,
                    Args.NTRAJ,Args.NSW,Args.NELM,Args.KbT,Args.edt,Args.hbar,
                    Args.SIGMA,Args.dt,k_proc,k_proc_num,nk_a,nq,n_p,nmodes,
                    kqidx_p,epc_a,energy,phonon,Args.LHOLE
                )

                endtime = MPI.Wtime()
                if myid_split == 0:
                    print("FSSH time in sample %d: %.6fs"\
                          %(j,endtime-starttime))

