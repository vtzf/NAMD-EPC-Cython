# generage the whole k list in fine grid
#
#
import numpy as np
import configparser




conf = configparser.ConfigParser()
conf.read('postprocessing.ini',encoding='utf-8')


nk_str = conf['Sigma']['nk']
nk_list = nk_str[1:-1].split(',')
kmesh = np.array([int(i) for i in nk_list],dtype=np.int32)

nq_str  = conf['main']['nq']
nq_list = nq_str[1:-1].split(',')
qmesh   = np.array([int(i) for i in nq_list],dtype=np.int32)



with open("k_all_list", "w") as f:
    for i in range(kmesh[0]):
        for j in range(kmesh[1]):
            for k in range(kmesh[2]):
                x = f"{i / kmesh[0]:.8f}"
                y = f"{j / kmesh[1]:.8f}"
                z = f"{k / kmesh[2]:.8f}"
                f.write(f"{x} {y} {z}\n")

with open("q_all_list", "w") as f:
    for i in range(qmesh[0]):
        for j in range(qmesh[1]):
            for k in range(qmesh[2]):
                x = f"{i / qmesh[0]:.8f}"
                y = f"{j / qmesh[1]:.8f}"
                z = f"{k / qmesh[2]:.8f}"
                f.write(f"{x} {y} {z}\n")

