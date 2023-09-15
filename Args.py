import numpy as np
import os
from mpi4py import MPI

# parameter
EPMDIR   = '../namdepc/h5files'
EPMPREF  = 'graphene_ephmat_p'
NPARTS   = 9
nqx      = 90
nqy      = 90
nqz      = 1

namddir  = 'output' # output NAMD output file in namddir
POTIM    = 1.0       # MD time step (fs)
SIGMA    = 0.025

LHOLE    = False     # Hole/electron transfer

TEMP     = 300.0       # temperature in Kelvin
NSAMPLE  = 1         # INICON sample number
NSW      = 50      # time step for NAMD run
NELM     = 100      # electron time step (per fs)
NTRAJ    = 2000     # SH trajectories number

EMIN     = -5.0   # band minimum energy (eV)
EMAX     = 2.0    # band maximum energy (eV)
PHCUT    = 1.0    # phonon minimum energy (meV)
LTRANS   = 'L'    # epc symmetrization:
                  # 'U': use upper triangle
                  # 'L': use lower triangle
                  # 'S': use both by (epc+epc.T.conj())/2


def ReadInp():
    global EPMDIR;  global EPMPREF; global NPARTS
    global nqx;     global nqy;     global nqz
    global namddir; global POTIM;   global SIGMA
    global LHOLE;   global TEMP;    global NSAMPLE
    global NSW;     global NELM;    global NTRAJ
    global EMIN;    global EMAX;    global PHCUT
    global LTRANS

    text = [line.strip() for line in open('inp') if line.strip()]
    inp = {}
    for line in text:
        if (line[0]=='&' or line[0]=='/' or line[0]=='!'):
            continue
        temp = line.split('!')[0].split('=')
        key = temp[0].strip()
        value = temp[1].strip().strip('\'').strip('\"')
        inp[key] = value

    EMIN = float(inp['EMIN']); EMAX = float(inp['EMAX'])
    NBANDS = int(inp['NBANDS']); nqx = int(inp['NQX'])
    nqy = int(inp['NQY']); nqz = int(inp['NQZ'])
    NSW = int(inp['NSW']); POTIM = float(inp['POTIM'])
    TEMP = float(inp['TEMP']); NSAMPLE = int(inp['NSAMPLE'])
    NELM = int(inp['NELM']); NTRAJ = int(inp['NTRAJ'])
    LHOLE = True if inp['LHOLE'] == '.T.' else False
    NPARTS = int(inp['NPARTS']); SIGMA = float(inp['SIGMA'])
    EPMDIR = inp['EPMDIR']
    EPMPREF = inp['EPMPREF']+'_ephmat_p'
    namddir = inp['NAMDDIR']

    
def WriteInp(nbands,nk,n_p):
    with open(namddir+'/inp','w') as f:
        f.write('&NAMDPARA\n')
        f.write("  {:<11}= ".format('BMIN')+"%d\n"%(1))
        f.write("  {:<11}= ".format('BMAX')+"%d\n"%(nbands))
        f.write("  {:<11}= ".format('EMIN')+"%d\n"%(EMIN))
        f.write("  {:<11}= ".format('EMAX')+"%d\n"%(EMAX))
        f.write("  {:<11}= ".format('NBANDS')+"%d\n"%(nbands))
        f.write("  {:<11}= ".format('NKPOINTS')+"%d\n"%(nk))
        f.write("  {:<11}= ".format('Np')+"%d\n\n"%(n_p))
        f.write("  {:<11}= ".format('NSW')+"%d\n"%(NSW))
        f.write("  {:<11}= ".format('POTIM')+"%.1f\n"%(POTIM))
        f.write("  {:<11}= ".format('TEMP')+"%.1f\n\n"%(TEMP))
        f.write("  {:<11}= ".format('NSAMPLE')+"%d\n"%(NSAMPLE))
        f.write("  {:<11}= ".format('NAMDTIME')+"%.1f\n"%(POTIM*NSW))
        f.write("  {:<11}= ".format('NELM')+"%d\n"%(NELM))
        f.write("  {:<11}= ".format('NTRAJ')+"%d\n"%(NTRAJ))
        f.write("  {:<11}= ".format('LHOLE')+".%s.\n\n"%(str(LHOLE)[0]))
        f.write("  {:<11}= ".format('LEPC')+".T.\n")
        f.write("  {:<11}= ".format('LSORT')+".F.\n")
        f.write("  {:<11}= ".format('EPCTYPE')+"1\n")
        f.write("  {:<11}= ".format('NPARTS')+"%d\n"%(NPARTS))
        f.write("  {:<11}= ".format('SIGMA')+"%s\n"%(str(SIGMA)))
        f.write("  {:<11}= ".format('EPMDIR')+"\'../%s\'\n"%(EPMDIR))
        f.write("  {:<11}= ".format('EPMPREF')+"\'%s\'\n/\n"%(EPMPREF[0:-9]))


ReadInp()
# constants
Ry2eV = 13.605698065894
hbar = 0.6582119281559802 # reduced Planck constant eV/fs
eV = 1.60217733E-19
Kb_eV = 8.6173857E-5
Kb = Kb_eV*eV
KbT = kbT = Kb_eV * TEMP

# parameters check
comm = MPI.COMM_WORLD
myid = comm.Get_rank()
if myid == 0:
    if not os.path.exists(namddir):
        os.makedirs(namddir)

# preprocess
dt = POTIM
edt = dt/NELM
