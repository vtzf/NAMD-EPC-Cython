import os
import numpy as np
import configparser

conf = configparser.ConfigParser()
conf.read('postprocessing.ini',encoding='utf-8')

nq_str = conf['main']['nq']
qmesh = np.array([int(i) for i in nq_str[1:-1].split(',')],dtype=np.int32)
kmesh = qmesh
inDir = conf['main']['inDir']
epcDir = conf['main']['epcDir']

with open('k_all_list','w') as f:
    for i in range(kmesh[0]):
        for j in range(kmesh[1]):
            for k in range(kmesh[2]):
                f.write(f"{i/kmesh[0]:.8f} {j/kmesh[1]:.8f} {k/kmesh[2]:.8f}\n")

k_list = np.genfromtxt('k_all_list')
if os.path.exists('eph.qpt'):
    q_list = np.genfromtxt('eph.qpt')
    weights = q_list[:,3]
else:
    q_list = np.load(os.path.join(inDir,epcDir,'qpath.npy'))
    weights = np.ones(len(q_list))

nq = len(q_list)
knum = kmesh[0]*kmesh[1]*kmesh[2]
qidx = []
for j in range(nq):
    nx = q_list[j][0]*kmesh[0]
    ny = q_list[j][1]*kmesh[1]
    nz = q_list[j][2]*kmesh[2]
    idx = round(nz+(nx*kmesh[1]+ny)*kmesh[2])
    if idx<0 or idx>=knum:
        raise IndexError(f'q point is outside the grid: {q_list[j]}')
    qidx.append(idx)

with open('q_list','w') as f:
    for i in range(nq):
        f.write(f"{q_list[i][0]:.8f} {q_list[i][1]:.8f} {q_list[i][2]:.8f} {weights[i]:.8f} {int(qidx[i])}\n")
