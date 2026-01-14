import numpy as np
from ase.io import read
import time
import configparser
import json


start = time.time()

conf = configparser.ConfigParser()
conf.read('config.ini',encoding='utf-8')

inDir = conf['epc']['inDir']+'/'
chg_dic = json.loads(conf['epc']['chg_dic'])
ucellidx_str = conf['epc']['ucellidx']
ucellidx_list = ucellidx_str[1:-1].split(',')
ucellidx = [int(i) for i in ucellidx_list]
poscar_ucell = inDir+conf['epc']['poscar_ucell']
infile_in = inDir+conf['epc']['infile_in_s']
infile_split = int(conf['epc']['infile_split_s'])
infile_out = inDir+conf['epc']['infile_out']

pos = read(poscar_ucell)
ucell = pos.cell[:]
atom = pos.positions
atomnum = atom.shape[0]
atomidx = list(pos.symbols)
charge = [chg_dic[i] for i in atomidx]

ncell = ucellidx[0]*ucellidx[1]*ucellidx[2]
atom_s = [j for i in range(ncell) for j in atomidx]
charge_s = [j for i in range(ncell) for j in charge]

center = [i//2 for i in ucellidx]

# generate scell
xyz = np.zeros((ncell*atom.shape[0],3))
i = 0
for j in range(ucellidx[0]):
    for k in range(ucellidx[1]):
        for l in range(ucellidx[2]):
            xyz[i:i+atomnum] = atom+np.sum(ucell*(np.array([[j],[k],[l]])),axis=0)
            i += atomnum

scell = ucell*np.array(ucellidx).reshape(3,1)

# get fixed grid
infile_c = open(infile_in).readlines()
for i in range(len(infile_c)):
    if infile_c[i].find("scf.energycutoff")>=0:
        Grid_Ecut = float(infile_c[i].split()[1])
        break
print(Grid_Ecut)

Bohr2Ang = 0.529177249
abc = scell/Bohr2Ang
tmp = np.cross(abc[1],abc[2])
CellV = np.dot(abc[0],tmp)
Cell_Volume = abs(CellV)

A = np.zeros((3))
tmp = np.cross(abc[1],abc[2])
A[0] = np.dot(tmp,tmp)
tmp = np.cross(abc[2],abc[0])
A[1] = np.dot(tmp,tmp)
tmp = np.cross(abc[0],abc[1])
A[2] = np.dot(tmp,tmp)
DouN = np.sqrt(Grid_Ecut/A)*Cell_Volume/np.pi

Lg2 = np.log(2)
Lg3 = np.log(3)
Lg5 = np.log(5)
Lg7 = np.log(7)

ngrid = np.zeros((3),dtype=int)
for i in range(3):
    LgN = np.log(DouN[i])
    MinD = 10e+10  
    pmax = int(np.ceil(LgN/Lg2))
    for p in range(pmax+1):
        qmax = int(np.ceil((LgN-p*Lg2)/Lg3))
        for q in range(qmax+1):
            rmax = int(np.ceil((LgN-p*Lg2-q*Lg3)/Lg5))
            for r in range(rmax+1):
                smax = int(np.ceil((LgN-p*Lg2-q*Lg3-r*Lg5)/Lg7))
                for s in range(smax+1):
                    LgTN = p*Lg2+q*Lg3+r*Lg5+s*Lg7
                    if (abs(LgTN-LgN)<MinD):
                        MinD = abs(LgTN-LgN)
                        popt = p
                        qopt = q
                        ropt = r
                        sopt = s
  
    k = 1
    for p in range(popt): k *= 2
    for q in range(qopt): k *= 3
    for r in range(ropt): k *= 5
    for s in range(sopt): k *= 7
  
    ngrid[i] = k

xyzc = np.sum(xyz,axis=0)/(atomnum*ncell*Bohr2Ang)
gtv = abc/ngrid.reshape(3,1)
xyzm = np.dot(ngrid//2-((ngrid+1)%2)/2,gtv)
Grid_Origin = xyzc - xyzm

# output inp
f = open(infile_out,'w')
for i in range(infile_split):
    f.write(infile_c[i])
f.write("Atoms.Number  %5d\n"%(xyz.shape[0]))
f.write("Atoms.SpeciesAndCoordinates.Unit   Ang # Ang|AU\n")
f.write("<Atoms.SpeciesAndCoordinates\n")
for i in range(xyz.shape[0]):
    f.write('%3d  %2s'%(i+1,atom_s[i]))
    for j in range(3):
        f.write('%18.12f'%(xyz[i,j]))
    f.write('  %3.1f  %3.1f  off\n'%(charge_s[i],charge_s[i]))
f.write("Atoms.SpeciesAndCoordinates>\n")
f.write("Atoms.UnitVectors.Unit             Ang # Ang|AU\n")
f.write("<Atoms.UnitVectors\n")
for i in range(3):
    for j in range(3):
        f.write("%18.12f"%(scell[i,j]))
    f.write("\n")
f.write("Atoms.UnitVectors>\n")
for i in range(infile_split,len(infile_c)):
    f.write(infile_c[i])
f.write("scf.fixed.grid  %17.12f %17.12f %17.12f\n"\
        %(Grid_Origin[0],Grid_Origin[1],Grid_Origin[2]))
f.close()

end = time.time()
print('Supercell time: %.4fs'%(end-start))
