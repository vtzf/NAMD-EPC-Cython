import numpy as np

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
    atom = np.array([info[i].split()[1] for i in range(idx[0]+1,idx[1])],dtype='U')
    xyz = np.array([info[i].split()[2:5] for i in range(idx[0]+1,idx[1])],dtype=float)
    abc = np.array([info[i].split()[0:3] for i in range(idx[2]+1,idx[3])],dtype=float)

    atom_set,atom_idx,atom_count = np.unique(atom,return_counts=True,return_index=True)
    atom_order = np.argsort(atom_idx)

    return xyz,abc,atom_set[atom_order],atom_count[atom_order]


def writePOS(infile,outfile):
    xyz,abc,atom_set,atom_count = get(infile)
    with open(outfile,'w') as f:
        f.write('test\n1.0\n')
        for i in range(3):
            for j in range(3):
                f.write('%17.12f'%(abc[i,j]))
            f.write('\n')
        for i in range(atom_set.shape[0]):
            f.write('%5s'%atom_set[i])
        f.write('\n')
        for i in range(atom_count.shape[0]):
            f.write('%5s'%atom_count[i])
        f.write('\nCartesian\n')
        for i in range(xyz.shape[0]):
            for j in range(3):
                f.write('%17.12f'%(xyz[i,j]))
            f.write('\n')


writePOS('input.dat','POSCAR_s')
