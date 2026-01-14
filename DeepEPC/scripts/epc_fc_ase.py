import os
import time
from glob import glob
import numpy as np

from ase.io import read
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii

import configparser


conf = configparser.ConfigParser()
conf.read('config.ini',encoding='utf-8')

phonon_method = conf['epc']['phonon_method']
inDir = conf['epc']['inDir']+'/'
dhamilDir = conf['epc']['dhamilDir']+'/'
phononDir = conf['epc']['phononDir']+'/'
ucellidx_str = conf['epc']['ucellidx']
ucellidx_list = ucellidx_str[1:-1].split(',')
ucellidx = [int(i) for i in ucellidx_list]
poscar_ucell = inDir+conf['epc']['poscar_ucell']
infile = conf['epc']['infile_out']
dQ = float(conf['epc']['dQ'])

atom_str = conf['epc']['atom']
atom_list = atom_str[1:-1].split(',')
atom = [int(i) for i in atom_list]

atomnum = sum(atom)

center = np.array([i//2 for i in ucellidx],dtype=int)
icell = (center[0]*ucellidx[1]+center[1])*ucellidx[2]+center[2]
catom = icell*atomnum+np.arange(atomnum)

R_list = np.array(
    [
        [i,j,k] \
        for i in range(ucellidx[0]) \
        for j in range(ucellidx[1]) \
        for k in range(ucellidx[2]) \
    ],
    dtype=int
)-center
R_num = ucellidx[0]*ucellidx[1]*ucellidx[2]

Hartree2eV = 27.211396641308
Bohr2Ang = 0.529177249


def get(Dir):
    idx = [0]*4
    with open(Dir,'r') as f:
        info = f.readlines()
    for i in range(len(info)):
        if info[i] == "<Atoms.SpeciesAndCoordinates\n":
            idx[0] = i
        if info[i] == "Atoms.SpeciesAndCoordinates>\n":
            idx[1] = i
        if info[i] == "<Atoms.UnitVectors\n":
            idx[2] = i
        if info[i] == "Atoms.UnitVectors>\n":
            idx[3] = i
    
    xyzinfo = np.array([info[i].split() for i in range(idx[0]+1,idx[1])],dtype='U')
    atomlist = xyzinfo[catom,1]
    xyz = np.array(xyzinfo[:,2:5],dtype=float)
    abc = np.array([info[i].split()[0:3] for i in range(idx[2]+1,idx[3])],dtype=float)

    return xyz,abc,\
        [atomic_masses[atomic_numbers[atomlist[i]]] for i in np.arange(atomnum)]


def ReadForce(Dir):
    force = []
    with open(glob(Dir+'*.out')[0],'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line == "<coordinates.forces\n":
                line = f.readline()
                while True:
                    line = f.readline()
                    if line == "coordinates.forces>\n":
                        break
                    else:
                        force.append(line.split()[5:8])
    f.close()

    force = np.array(force,dtype=float)
    return force
            

def FC_Part(inDir,dhamilDir,R_num,catom,delta,mass):
    force = np.zeros((atomnum*3,R_num*atomnum,3))
    if phonon_method == 'F':
        f0 = np.zeros((R_num*atomnum,3))
        f0 = ReadForce(inDir)
        for i in range(atomnum):
            for j in range(3):
                force[i*3+j] = ReadForce(inDir+dhamilDir+'%d+%s/'%(catom[i],delta[j]))
        fc_p = -(force-f0[np.newaxis,np.newaxis])/dQ
    elif phonon_method == 'B':
        f0 = np.zeros((R_num*atomnum,3))
        f0 = ReadForce(inDir)
        for i in range(atomnum):
            for j in range(3):
                force[i*3+j] = ReadForce(inDir+dhamilDir+'%d-%s/'%(catom[i],delta[j]))         
        fc_p = (force-f0[np.newaxis,np.newaxis])/dQ
    else:
        for i in range(atomnum):
            for j in range(3):
                force[i*3+j] = ReadForce(inDir+dhamilDir+'%d+%s/'%(catom[i],delta[j]))
                force[i*3+j] -= ReadForce(inDir+dhamilDir+'%d-%s/'%(catom[i],delta[j]))
        fc_p = -force/(2*dQ)

    return fc_p.reshape(atomnum*3,R_num,atomnum*3).swapaxes(0,1)


def apply_cutoff(fc,Rcut):
    poscar = read(poscar_ucell)
    fc1 = fc.reshape(R_num,atomnum,3,atomnum,3)
    cell = poscar.cell.transpose()
    pos = poscar.get_positions()

    for n in range(R_num):
        # Lattice vector to cell
        R_v = np.dot(cell,R_list[n])
        # Atomic positions in cell
        posn = pos+R_v
        # Loop over atoms and zero elements
        for a in range(atomnum):
            dist_a = np.sqrt(np.sum((pos[a]-posn)**2,axis=-1))
            # Atoms where the distance is larger than the cufoff
            a_p = dist_a>Rcut  # np.where(dist_a > r_c)
            # Zero elements
            fc1[n,a,:,a_p,:] = 0.0


def symmetrize(fc):
    nRx = ucellidx[0]
    nRy = ucellidx[1]
    nRz = ucellidx[2]
    fc1 = fc.reshape(nRx,nRy,nRz,atomnum*3,atomnum*3)
    i, j, k = 1 - np.array(ucellidx) % 2
    fc1[i:,j:,k:] *= 0.5
    fc1[i:,j:,k:] += \
        fc1[i:,j:,k:][::-1,::-1,::-1].transpose(0,1,2,4,3).copy()

    return fc1.reshape(R_num,atomnum*3,atomnum*3)


def acoustic(fc):
    fc1 = fc.copy()
    # Correct atomic diagonals of R_m = (0, 0, 0) matrix
    for C in fc1:
        for a in range(atomnum):
            for b in range(atomnum):
                fc[icell,3*a:3*a+3,3*a:3*a+3] -= C[3*a:3*a+3,3*b:3*b+3]


def ForceConstant(inDir,dhamilDir,phononDir,infile):
    delta = ['x','y','z']
    abc, xyz, mass = get(inDir+infile)

    fc = FC_Part(inDir,dhamilDir,R_num,catom,delta,np.array(mass))

    #apply_cutoff(fc,8)
    for i in range(3):
        fc = symmetrize(fc)
        acoustic(fc)

    #print(np.average(fc))

    for i in range(R_num):
        for j in range(atomnum):
            for l in range(atomnum):
                factor = Hartree2eV/Bohr2Ang/np.sqrt(mass[j]*mass[l])
                fc[i,j*3:j*3+3,l*3:l*3+3] *= factor

    np.save(inDir+phononDir+'fc_avg.npy',fc)


##########################################
start = time.time()
if not os.path.exists(inDir+phononDir):
    os.mkdir(inDir+phononDir)
ForceConstant(inDir,dhamilDir,phononDir,infile)
end = time.time()
print('Running time: %.2fs'%(end-start))
