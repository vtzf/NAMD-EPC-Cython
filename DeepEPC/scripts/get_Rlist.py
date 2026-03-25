import numpy as np
import json
import configparser
from glob import glob
import h5py

#conf = configparser.ConfigParser()
#conf.read('config.ini',encoding='utf-8')
#
#IsH5_u = True if conf['ucell']['IsH5_u']=='True' else False
#inDir = conf['epc']['inDir']+'/'
#
#if IsH5_u:
#    H5HamName_u = conf['ucell']['H5HamName_u']
#    filename = inDir+'ucell/%s.h5'%H5HamName_u
#else:
#    filename = glob(inDir+'ucell/*.scfout')[0]

IsH5_u = True
filename = '../example/Ni/soc/7_so/ucell/out/hamiltonians.h5'

#IsH5_u = False
#filename = '../example/Ni/nosoc/7_no_so/ucell/openmx.scfout'

# get R_num
def ReadRlist(Name):
    if not IsH5_u:
        fp = open(Name,'rb')
        fp.seek(0)
        i_vec = np.fromfile(fp,dtype=np.intc,count=6)
        atomnum = i_vec[0]
        TCpyCell = i_vec[5]
        fp.seek(4+(TCpyCell+1)*4*8,1)
        
        atv_ijk = np.zeros((TCpyCell+1,4),dtype=np.intc)
        for Rn in range(TCpyCell+1):
            atv_ijk[Rn] = np.fromfile(fp,dtype=np.intc,count=4)
        fp.seek(atomnum*4,1)

        FNAN = np.zeros((atomnum+1),dtype=np.intc)
        FNAN[1:] = np.fromfile(fp,dtype=np.intc,count=atomnum)
        natn = [[]]
        n_num = 0
        for ct_AN in range(1,atomnum+1):
            natn.append(np.fromfile(fp,dtype=np.intc,count=FNAN[ct_AN]+1))
            n_num += FNAN[ct_AN]+1
        ncn = np.zeros((n_num),dtype=np.intc)
        n_count = 0
        for ct_AN in range(1,atomnum+1):
            tmp = np.fromfile(fp,dtype=np.intc,count=FNAN[ct_AN]+1)
            ncn[n_count:n_count+FNAN[ct_AN]+1] = tmp
            n_count += FNAN[ct_AN]+1
    
        R_list = np.unique(atv_ijk[ncn,1:],axis=0)
    else:
        f=h5py.File(Name,'r')
        h_key = list(f.keys())
        h_key = np.array([json.loads(x) for x in h_key],dtype=int)
        R_list = np.unique(h_key[:,0:3],axis=0)
    
    print(np.max(np.abs(2*R_list+1),axis=0))


ReadRlist(filename)
