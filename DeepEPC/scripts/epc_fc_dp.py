from ase.io import read
from ase.phonons import Phonons
from deepmd.calculator import DP
import time
import numpy as np
import configparser

starttime = time.time()

conf = configparser.ConfigParser()
conf.read('config.ini',encoding='utf-8')

phonon_method = conf['epc']['phonon_method']
inDir = conf['epc']['inDir']+'/'
phononDir = conf['epc']['phononDir']+'/'
ifcname = conf['epc']['ifcname']
ifcRstr = conf['epc']['ifcRcut']
if ifcRstr == '': ifcRcut = None
else: ifcRcut = float(ifcRstr)
ucellidx_str = conf['epc']['ucellidx']
ucellidx_list = ucellidx_str[1:-1].split(',')
ucellidx = [int(i) for i in ucellidx_list]
poscar_ucell = inDir+conf['epc']['poscar_ucell']
infile = conf['epc']['infile_out']
dQ = float(conf['epc']['dQ'])

pos = read(poscar_ucell)
ph = Phonons(pos,supercell=ucellidx,delta=dQ,center_refcell=True)
ph.calc = DP(model='./graph.pb')
ph.run()

# Read forces and assemble the force constant matrix
ph.read(method='standard',symmetrize=3,acoustic=True,cutoff=ifcRcut) #standard, frederiksen
np.save(inDir+phononDir+ifcname,ph.D_N)

end = time.time()
print('Running time: %.2fs'%(end-start))
