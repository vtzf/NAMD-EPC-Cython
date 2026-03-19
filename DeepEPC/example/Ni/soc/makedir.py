import numpy as np
import time
import os, sys
import shutil
from ase.io import read
import configparser


conf = configparser.ConfigParser()
conf.read('config.ini',encoding='utf-8')

dhamil_method = conf['epc']['dhamil_method']
inDir = conf['epc']['inDir']+'/'
dhamilDir = conf['epc']['dhamilDir']+'/'
ucellidx_str = conf['epc']['ucellidx']
ucellidx_list = ucellidx_str[1:-1].split(',')
ucellidx = [int(i) for i in ucellidx_list]
dQ = float(conf['epc']['dQ'])
poscar_ucell = inDir+conf['epc']['poscar_ucell']
infile_out = conf['epc']['infile_out']
subfile_s = conf['epc']['subfile_s']

pos = read(poscar_ucell)
atomnum = len(pos)
center = np.array([i//2 for i in ucellidx],dtype=int)
ucellnum = np.array([ucellidx]*3).T
ncell = ucellidx[0]*ucellidx[1]*ucellidx[2]
icell = (center[0]*ucellidx[1]+center[1])*ucellidx[2]+center[2]
catom = icell*atomnum+np.arange(atomnum)


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
    xyz = np.array([info[i].split()[2:5] for i in range(idx[0]+1,idx[1])],dtype=float)
    abc = np.array([info[i].split()[0:3] for i in range(idx[2]+1,idx[3])],dtype=float)
    return info[0:idx[0]+1],info[idx[0]+1:idx[1]],info[idx[1]:],xyz,abc


def write_file(info,info1,info2,xyz,abc,sign):
    if sign == 1:
        dQ0 = dQ
        signstr = '+'
    else:
        dQ0 = -dQ
        signstr = '-'
    for i in catom:
        for j,k in zip(['x','y','z'],[0,1,2]):
            dxyz = np.zeros((xyz.shape[0],3))
            xyz_tmp = xyz.copy()
            xyz_tmp[i,k] += dQ0
            dxyz[i,k] += dQ0

            dirname_h = inDir+dhamilDir+'%d%s%s/'%(i,signstr,j)
            if not os.path.exists(dirname_h):
                os.mkdir(dirname_h)
            shutil.copyfile(inDir+subfile_s,dirname_h+subfile_s)
            f_h =  open(dirname_h+infile_out,'w')
            for l in info:
                f_h.write(l)
            for l in range(xyz.shape[0]):
                xyzw = info1[l].split()
                f_h.write('%3s%4s%18.12f%18.12f%18.12f'\
                         %(xyzw[0],xyzw[1],\
                           xyz_tmp[l,0],xyz_tmp[l,1],xyz_tmp[l,2]))
                for m in range(5,len(xyzw)):
                    f_h.write(' %5s'%(xyzw[m]))
                f_h.write('\n')
            for l in info2:
                f_h.write(l)
            f_h.close()


def write_ucell(info1):
    if not os.path.exists(inDir+'ucell'):
        os.mkdir(inDir+'ucell')
    ucell = pos.cell[:]
    atom = pos.positions

    infile_in = conf['ucell']['infile_in_u']
    infile_split = int(conf['ucell']['infile_split_u'])
    infile_c = open(inDir+infile_in).readlines()
    f = open(inDir+'ucell/'+infile_out,'w')
    for i in range(infile_split):
        f.write(infile_c[i])
    f.write("Atoms.Number  %5d\n"%(atom.shape[0]))
    f.write("Atoms.SpeciesAndCoordinates.Unit   Ang # Ang|AU\n")
    f.write("<Atoms.SpeciesAndCoordinates\n")
    for i in range(atom.shape[0]):
        xyzw = info1[i].split()
        f.write('%3d %2s%18.12f%18.12f%18.12f'\
                %(i+1,xyzw[1],atom[i,0],atom[i,1],atom[i,2]))
        for m in range(5,len(xyzw)):
            f.write(' %5s'%(xyzw[m]))
        f.write('\n')
    f.write("Atoms.SpeciesAndCoordinates>\n")
    f.write("Atoms.UnitVectors.Unit             Ang # Ang|AU\n")
    f.write("<Atoms.UnitVectors\n")
    for i in range(3):
        for j in range(3):
            f.write("%18.12f"%(ucell[i,j]))
        f.write("\n")
    f.write("Atoms.UnitVectors>\n")
    for i in range(infile_split,len(infile_c)):
        f.write(infile_c[i])
    f.close()

    subfile_u = conf['ucell']['subfile_u']
    shutil.copyfile(inDir+subfile_u,inDir+'ucell/'+subfile_u)


def process(inDir,infile_out):
    info, info1, info2, xyz, abc = get(inDir+infile_out)
    if not os.path.exists(inDir+dhamilDir):
        os.mkdir(inDir+dhamilDir)
    if dhamil_method == 'F':
        write_file(info,info1,info2,xyz,abc,1)
    elif dhamil_method == 'B':
        write_file(info,info1,info2,xyz,abc,-1)
    else:
        write_file(info,info1,info2,xyz,abc,1)
        write_file(info,info1,info2,xyz,abc,-1)
    write_ucell(info1)


start = time.time()
process(inDir,infile_out)
end = time.time()
print('Makedir time: %.2fs'%(end-start))
