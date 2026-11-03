import os
import numpy as np
import configparser

conf = configparser.ConfigParser()
conf.read('postprocessing.ini',encoding='utf-8')

nq_str = conf['main']['nq']
qmesh = np.array([int(i) for i in nq_str[1:-1].split(',')],dtype=np.int32)
nk_str = conf['Sigma']['nk']
kmesh = np.array([int(i) for i in nk_str[1:-1].split(',')],dtype=np.int32)
inDir = conf['main']['inDir']
epcDir = conf['main']['epcDir']

k_list = np.genfromtxt('k_all_list')
q_list = np.genfromtxt('q_all_list')
kpath = np.load(os.path.join(inDir,epcDir,'kpath.npy'))
nq = qmesh[0]*qmesh[1]*qmesh[2]
nk = len(kpath)
qidx = []
kidx = []

for j in range(nq):
    nx = q_list[j][0]*kmesh[0]
    ny = q_list[j][1]*kmesh[1]
    nz = q_list[j][2]*kmesh[2]
    idx = round(nz+(nx*kmesh[1]+ny)*kmesh[2])
    qidx.append(idx)

for j in range(nk):
    nx = kpath[j][0]*kmesh[0]
    ny = kpath[j][1]*kmesh[1]
    nz = kpath[j][2]*kmesh[2]
    idx = round(nz+(nx*kmesh[1]+ny)*kmesh[2])
    kidx.append(idx)

with open('k_list','w') as f:
    for i in range(nk):
        if kidx[i]<0 or kidx[i]>=kmesh[0]*kmesh[1]*kmesh[2]:
            raise IndexError(f'k-path point is outside the grid: {kpath[i]}')
        f.write(f"{kpath[i][0]:.8f} {kpath[i][1]:.8f} {kpath[i][2]:.8f} {1} {int(kidx[i])}\n")

with open('q_list','w') as f:
    for i in range(nq):
        f.write(f"{q_list[i][0]:.8f} {q_list[i][1]:.8f} {q_list[i][2]:.8f} {1} {int(qidx[i])}\n")
