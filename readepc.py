import sys
import numpy as np
import configparser
import readh5
from mpi4py import MPI


def ReadEpcInfo(comm,myid,nqx,nqy,nqz,EMIN,EMAX):
    conf = configparser.ConfigParser()
    conf.read('config.ini',encoding='utf-8')

    inDir = conf['epc']['inDir']+'/'
    bandDir = conf['epc']['bandDir']+'/'
    bassel_name = conf['epc']['basselname']
    basselname = inDir+bandDir+bassel_name

    atom_str = conf['epc']['atom']
    atom_list = atom_str[1:-1].split(',')
    atom = [int(i) for i in atom_list]
    atomnum = sum(atom)
    nmodes = atomnum*3

    nq_str = conf['epc']['nq']
    nq_list = nq_str[1:-1].split(',')
    nq = np.array([int(i) for i in nq_list],dtype=np.int32)

    if nq[0] != nqx or nq[1] != nqy or nq[2] != nqz:
        print('Unmatch of nq and [nqx,nqy,nqz]!\n')
        sys.exit()

    bmin = int(conf['epc']['bmin'])
    bmax = int(conf['epc']['bmax'])
    nbands = bmax-bmin+1
    valname = conf['epc']['valname']
    IsAllVec = True if conf['epc']['IsAllVec']=='True' else False
    if (myid==0):
        if IsAllVec:
            IsAllKlist = True
            energy = np.ascontiguousarray(
                np.load(inDir+bandDir+valname)[:,bmin:bmax+1]
            )
            bassel = np.where((energy>=EMIN)&(energy<=EMAX))
            nk = bassel[0].shape[0]
            kmin_s = bassel[0].min()
            kmax_s = bassel[0].max()
            bmin_s = bassel[1].min()
            bmax_s = bassel[1].max()
        else:
            IsAllKlist = True if conf['epc']['IsAllKlist']=='True' else False
            if IsAllKlist:
                energy = np.load(inDir+bandDir+valname)
                bassel = np.where((energy>=EMIN)&(energy<=EMAX))
                nk = bassel[0].shape[0]
                kmin_s = bassel[0].min()
                kmax_s = bassel[0].max()
                bmin_s = bassel[1].min()
                bmax_s = bassel[1].max()
            else:
                bassel = np.load(basselname)
                nk = bassel.shape[0]
                kmin_s = bassel[:,0].min()
                kmax_s = bassel[:,0].max()
                bmin_s = bassel[:,1].min()
                bmax_s = bassel[:,1].max()
    else:
        nk = 0
        kmin_s = 0
        kmax_s = 0
        bmin_s = 0
        bmax_s = 0
    nk = comm.bcast(nk,root=0)
    kmin_s = comm.bcast(kmin_s,root=0)
    kmax_s = comm.bcast(kmax_s,root=0)
    bmin_s = comm.bcast(bmin_s,root=0)
    bmax_s = comm.bcast(bmax_s,root=0)
    if nk == 0:
        print("No data in energy range [%.5f,%.5f]!"%(EMIN,EMAX))
        sys.exit()
   
    poscar = open(inDir+conf['epc']['poscar_ucell']).readlines()
    abc = np.array([i.split()[0:3] for i in poscar[2:5]],dtype=float)*float(poscar[1].split()[0])

    return nk, nqx*nqy*nqz, nmodes, nbands, kmin_s, kmax_s, bmin_s, bmax_s, abc


def ReadEpc(
    comm,PHCUT,EMIN,EMAX,NM_BLOCK,nk,kqidx,
    energy_a,phonon,epc_a,LTRANS,LEPCSHM,LPHSHM
):
    myid = comm.Get_rank()
    nprocs = comm.Get_size()

    conf = configparser.ConfigParser()
    conf.read('config.ini',encoding='utf-8')
    EpcType = conf['epc']['EpcType']
    if EpcType != 'A':
        print('EpcType not supporting!')
        sys.exit()

    inDir = conf['epc']['inDir']+'/'
    bandDir = conf['epc']['bandDir']+'/'
    phononDir = conf['epc']['phononDir']+'/'
    epcDir = conf['epc']['epcDir']+'/'
    bassel_name = conf['epc']['basselname']
    epc_name = conf['epc']['epcname']

    basselname = inDir+bandDir+bassel_name
    epcname = inDir+epcDir+epc_name

    atom_str = conf['epc']['atom']
    atom_list = atom_str[1:-1].split(',')
    atom = [int(i) for i in atom_list]
    atomnum = sum(atom)
    nmodes = atomnum*3

    nq_str = conf['epc']['nq']
    nq_list = nq_str[1:-1].split(',')
    nq = np.array([int(i) for i in nq_list],dtype=np.int32)

    phvalname = conf['epc']['phvalname']
    phname = inDir+phononDir+phvalname
    #phonon = np.load(inDir+phononDir+phvalname)

    bmin = int(conf['epc']['bmin'])
    bmax = int(conf['epc']['bmax'])
    nbands = bmax-bmin+1
    valname = conf['epc']['valname']
    IsAllVec = True if conf['epc']['IsAllVec']=='True' else False
    if IsAllVec:
        IsAllKlist = True
        if myid == 0:
            energy_a[:] = np.ascontiguousarray(
                np.load(inDir+bandDir+valname)[:,bmin:bmax+1]
            )
    else:
        IsAllKlist = True if conf['epc']['IsAllKlist']=='True' else False
        if IsAllKlist:
            if (myid==0):
                energy_a[:] = np.load(inDir+bandDir+valname)
        else:
            if (myid==0):
                energy_a[:] = np.load(inDir+bandDir+valname)
                bassel = np.load(basselname)
                ekidx = np.ascontiguousarray(bassel[:,0])
                ebidx = np.ascontiguousarray(bassel[:,1])
                valpname = valname.split('.')[0]+'_p.npy'
                energy = np.load(inDir+bandDir+valpname)
            else:
                ekidx = np.zeros((nk),dtype=np.int32)
                ebidx = np.zeros((nk),dtype=np.int32)
                energy = np.zeros((nk),dtype=float)
            comm.Bcast(ekidx,root=0)
            comm.Bcast(ebidx,root=0)
    comm.Bcast(energy,root=0)

    if IsAllVec or IsAllKlist:
        return readh5.ReadNpy(
            comm,nmodes,NM_BLOCK,nbands,nq[0],nq[1],nq[2],nk,
            PHCUT,EMIN,EMAX,kqidx,energy_a,phonon,epc_a,LTRANS.encode('utf-8'),
            phname,epcname.encode('utf-8'),LEPCSHM,LPHSHM
        )
    else:
        return readh5.ReadNpyPart(
            comm,nmodes,NM_BLOCK,nbands,nq[0],nq[1],nq[2],nk,
            ekidx,ebidx,PHCUT,EMIN,EMAX,kqidx,energy_a,energy,
            phonon,epc_a,LTRANS.encode('utf-8'),phname,
            epcname.encode('utf-8'),LEPCSHM,LPHSHM
        )
