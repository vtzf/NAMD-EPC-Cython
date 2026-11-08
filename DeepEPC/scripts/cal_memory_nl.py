import numpy as np
import configparser
import sys

# ============================================================
# memory estimate for bak1 epc_nl.py (non-linear EPC engine, CURRENT code)
# usage: python cal_memory_nl.py [config.ini] [nnodes] [procs/node] [node_RAM_GB]
#
# models the CURRENT epc_nl.py (olp_inv + epc_t + MPIepcNL all ACTIVE). the
# tested/relevant path is IsAllKlist=True, EpcType=A, Ispin=2 (SOC); that path
# is modeled buffer-accurate. other branches reuse the same epc_t/drSH sizing
# but their compute buffers are approximated (see the per-branch notes).
#
# three allocation peaks are reported (in execution order):
#   PEAK_read : sparse-read/idx SHM only (per node). keyinfo + csr indices +
#               olp/ham/dr coexist; no big per-proc array yet -> rarely binds.
#   PEAK1     : olp_inv phase. drSH[/proc] + ScaLAPACK Cholesky buffers[/proc]
#               are alive WHILE the sparse SHM (olp/ham/dr + keyinfo + csr) is
#               still mapped (freed right after olp_inv). epc_t not yet alloc'd.
#   PEAK2     : compute (MPIepcNL_L then _R, sequential). epc_t[/proc] (dominant,
#               ~kfactor*kproc_num*nmnb2) + drSH + phvecval_p + NB_k batch bufs
#               [/proc], plus bandveck SHM + the drveck window (all-k in _R,
#               LARGE) [/node]. sparse SHM is freed by now.
# ============================================================
cfgfile = sys.argv[1] if len(sys.argv)>1 else 'config.ini'
nnodes   = int(sys.argv[2]) if len(sys.argv)>2 else 8
procs_pn = int(sys.argv[3]) if len(sys.argv)>3 else 48         # ranks per node
node_ram_gb = float(sys.argv[4]) if len(sys.argv)>4 else 187.0  # h3c=187, h3cm=1510
nprocs   = nnodes*procs_pn
# node RAM headroom: absolute reserve, not a flat fraction (shared w/ cal_memory.py).
# Idle top: non-reclaimable (kernel slab/struct-page + daemons) ~3.6 GiB on
# h3c(187), ~8.4 GiB on h3cm(1546). os_reserve=max(4,0.6%*RAM) bounds both;
# writeback_gb = dirty-page spike headroom for the large epc_t/drveck writes.
os_reserve_gb = max(4.0,0.006*node_ram_gb)
writeback_gb  = 4.0
usable_gb     = node_ram_gb-os_reserve_gb-writeback_gb   # shared w/ cal_memory.py
# python+MKL ~0.24 GiB VmHWM/rank; ~half is read-only .so/libmkl/interpreter text,
# ONE mmap-shared physical copy/node -> count once (shared w/ cal_memory.py).
py_shared_gb = 0.12            # shared code, once per node
py_priv_gb   = 0.12            # private anon heap, per rank
NB_k_buf_gb = 0.5              # per-proc byte budget for the NB_k batch buffers

# sparse nonzero counts feed ONLY the read/olp_inv-phase SHM. fill from a run's
# GetSparseNum, or drop key_num_nl.npy beside this script with columns
# [Nkey_h,Nkey_o,Nkey_dr,Nsparse_h,Nsparse_o,Nsparse_dr].
Nsparse_h  = 102092036         # H nonzeros  (WSe2 default; override for others)
Nsparse_o  = 102092036         # S nonzeros
Nsparse_dr = 102092036         # dr nonzeros (3 xyz share one count here)
Nkey_h = Nkey_o = Nkey_dr = 414376
try:
    kn = np.load('key_num_nl.npy')
    Nkey_h,Nkey_o,Nkey_dr,Nsparse_h,Nsparse_o,Nsparse_dr = [int(x) for x in kn[:6]]
    print('sparse counts loaded from key_num_nl.npy')
except (OSError,ValueError):
    print('sparse counts use in-script defaults (WSe2); edit Nsparse_* for other systems')

conf = configparser.ConfigParser()
conf.read(cfgfile,encoding='utf-8')
Ispin = int(conf['epc']['Ispin'])
EpcType = conf['epc']['EpcType']
ucellidx = [int(i) for i in conf['epc']['ucellidx'][1:-1].split(',')]
atom = [int(i) for i in conf['epc']['atom'][1:-1].split(',')]
orbital = [int(i) for i in conf['epc']['orbital'][1:-1].split(',')]
IsAllVec = conf['epc']['IsAllVec']=='True'
IsAllKlist = conf['epc']['IsAllKlist']=='True'
nq = [int(i) for i in conf['epc']['nq'][1:-1].split(',')]

def GenKlist(Kpoint_str,nqa):
    from math import gcd
    k_list = Kpoint_str.replace('[','').replace(']','').split(',')
    k_arr = np.array([eval(i) for i in k_list],dtype=float).reshape(-1,3)
    k_int = np.around(k_arr*nqa).astype(int)
    nkpoint = k_arr.shape[0]
    nklist = np.zeros((nkpoint-1),dtype=int)
    for i in range(nkpoint-1):
        dk_int = np.abs(k_int[i+1]-k_int[i])
        dk_int_p = dk_int[np.where(dk_int>0)[0]]
        if dk_int_p.shape[0]==3:
            nklist[i] = gcd(gcd(dk_int_p[0],dk_int_p[1]),gcd(dk_int_p[1],dk_int_p[2]))
        elif dk_int_p.shape[0]==2:
            nklist[i] = gcd(dk_int_p[0],dk_int_p[1])
        else:
            nklist[i] = dk_int_p[0]
    return int(np.sum(nklist))

cplx = 16
real = 8
s_i = 4
s_l = 8
GB = 1024.0**3
ceil_div = lambda a,b: -(-a//b)
# numroc upper bound: local rows of an n-dim array, block nb, over nproc procs
numroc = lambda n,nb,nproc: ceil_div(ceil_div(n,nb),nproc)*nb

# structural dims (mirror epc_nl.py exactly) --------------------------------
ncell = ucellidx[0]*ucellidx[1]*ucellidx[2]
atomnum_u = sum(atom)
Np = sum(x*y for x,y in zip(atom,orbital))    # unit-cell orbital dim (no spin)
N = Np*ncell                                  # supercell orbital dim
norb = Np if Ispin==0 else 2*Np               # bands dim (spin-doubled for SOC)
nbands = norb if IsAllVec else int(conf['epc']['bmax'])-int(conf['epc']['bmin'])+1
nmodes = atomnum_u*3
knum = nq[0]*nq[1]*nq[2]
if EpcType=='A':
    nkpath = knum
else:
    nkpath = GenKlist(conf['epc']['Kpoint'],np.array(nq))
kfactor = nkpath
nspin = 2 if Ispin==2 else 1
nspin2 = 4 if Ispin==2 else (Ispin+1)
nmnb2 = nmodes*nbands*nbands
norbnb = Np*nbands

# per-proc splits (mirror epc_nl.py) -----------------------------------------
# Np ceil-block split -> Np_num_max (epc_nl.py:414-426, also drSH row split)
Np_num_max = ceil_div(Np,nprocs) if Np%nprocs else Np//nprocs
# knum balanced split -> kproc_num_max (epc_nl.py:512-516 / qproc for AllKlist=F)
kproc_num_max = ceil_div(knum,nprocs)
# N ceil-block split -> Nproc_len_max (olp_inv densify row split)
Nproc_len_max = ceil_div(N,nprocs) if N%nprocs else N//nprocs

# ---- part1 k-batch NB_k -----------------------------------------------------
# NB_k batches k-points so drSH is streamed once per zgemm3m (read /NB_k). the
# amortization benefit SATURATES at ~ncell (drSH read drops below the
# drSHexpikR write floor), so ncell is the OPTIMAL target -> upper limit. a
# per-proc byte budget (now sitting on top of the ~epc_t footprint) caps it
# further down on tight nodes. worst-case per-k bytes = MPIepcNL_R_new:
#   drSHexpikR_B[Np_num*Np] + bandveckp_B[nspin*norbnb] + drveck_p_B[Np_num*nb]
NB_k_byte = (Np_num_max*Np + nspin*norbnb + Np_num_max*nbands)*cplx
NB_k_cap = max(1,int(NB_k_buf_gb*GB/NB_k_byte))
NB_k = max(1,min(ncell,nkpath,NB_k_cap))          # NEW adaptive-optimal formula
NB_k_old = max(ncell,min(4*ncell,nkpath,NB_k_cap)) # OLD formula (for comparison)

# ---- per-process private arrays --------------------------------------------
# drSH[nspin2,3,ncell,Np_num*Np]: alloc'd before olp_inv (epc_nl.py:429/437),
# HELD the whole run. SOC=complex, else real.
if Ispin==2:
    drSH = nspin2*3*ncell*Np_num_max*Np*cplx
else:
    drSH = (Ispin+1)*3*ncell*Np_num_max*Np*real
# olp_inv_new ScaLAPACK buffers (readdrSH_nc.pyx). square grid nprow*npcol.
nprow = 1
for i in range(1,int(nprocs**0.5)+1):
    if nprocs%i==0: nprow = i
npcol = nprocs//nprow
max_grid = max(nprow,npcol)
Mb = max(1,min(128,N//max_grid))
Nb = max(1,min(128,Np//max_grid))
sl_matbuf = N*Nproc_len_max*real                       # densified S column strip
sl_Sbuf   = numroc(N,Mb,nprow)*numroc(N,Mb,npcol)*real # 2D S/L (Cholesky), held
sl_Zbuf   = numroc(N,Mb,nprow)*numroc(Np,Nb,npcol)*real# 2D RHS/solution [N,Np]
sl_drT    = N*Np_num_max*real                          # dr^T / Z^T column strip
sl_Cbuf   = N*Np_num_max*real                          # H*Z^T column strip
# peak inside olp_inv: max(densify: matbuf+Sbuf, solve: Sbuf+Zbuf+drT+Cbuf)
sl_peak = max(sl_matbuf+sl_Sbuf, sl_Sbuf+sl_Zbuf+sl_drT+sl_Cbuf)
# epc_t[kfactor*kproc_num,nmnb2]: DOMINANT per-proc array (epc_nl.py:536)
epc_t = kfactor*kproc_num_max*nmnb2*cplx
# phvecval_p[kproc_num,nmodes*nmodes] (epc_nl.py:535)
phvecval_p = kproc_num_max*nmodes*nmodes*cplx
# MPIepcNL_L_new / _R_new per-proc batch buffers (norb_p=Np_num, norbital=Np)
buf_drSHexpikR_B = NB_k*Np*Np_num_max*cplx
buf_expikR_B = NB_k*ncell*cplx
buf_vdrvexpikR = nmodes*nbands*nbands*cplx
buf_drveck_a = nkpath*3*nspin*Np_num_max*nbands*cplx     # L_new only (calloc)
buf_bandveckp_B = NB_k*nspin*norbnb*cplx                 # R_new only
buf_drveck_p_B = NB_k*Np_num_max*nbands*cplx             # R_new only
buf_vdrv = nbands*nbands*cplx                            # R_new only
buf_L = buf_drSHexpikR_B+buf_expikR_B+buf_vdrvexpikR+buf_drveck_a
buf_R = (buf_drSHexpikR_B+buf_expikR_B+buf_bandveckp_B
         +buf_drveck_p_B+buf_vdrv+buf_vdrvexpikR)
g = lambda x: x/GB
drSH_gb,sl_gb,epc_t_gb,phv_gb = g(drSH),g(sl_peak),g(epc_t),g(phvecval_p)
buf_L_gb,buf_R_gb = g(buf_L),g(buf_R)

# ---- per-node shared (SHM, one copy per node on shm_id==0) ------------------
ham_shm = (2*4*Nsparse_h*real if Ispin==2 else (Ispin+1)*Nsparse_h*real)
olp_shm = Nsparse_o*real
dr_shm  = 3*Nsparse_dr*real
# read/idx peak: keyinfo(coo) + csr(built from it) + olp/ham/dr all coexist
# (epc_nl.py:358-385, keyinfo_h/o/dr freed at 387-389 right after coo2csridx)
keyinfo_coo = (Nsparse_h+Nsparse_o+Nsparse_dr)*2*s_l
csr = (Nsparse_o*2*s_i + Nsparse_h*2*s_i     # olp/ham_keyinfo
       + N*2*s_i + Nsparse_dr*s_i            # dr_csr ridx/cidx
       + N*2*s_i + Nsparse_h*s_i)            # ham_csr ridx/cidx
shm_readidx = ham_shm+olp_shm+dr_shm+keyinfo_coo+csr
# olp_inv-phase SHM: coo keyinfo freed, csr kept (olp_inv_new reads csr H/dr)
shm_olpinv = (ham_shm+olp_shm+dr_shm
              + Nsparse_o*2*s_i + Nsparse_h*2*s_i        # olp/ham_keyinfo
              + Nsparse_dr*s_i + N*2*s_i + Nsparse_h*s_i # dr/ham csr cidx+ridx
              + Np*2*s_i)                                # drp_csr_ridx
# compute-phase SHM: bandveck (always) + drveck window (per-k in _L, all-k _R)
if IsAllKlist:
    shm_bandveck = (knum*norb*nbands*cplx if Ispin==2
                    else (Ispin+1)*knum*Np*nbands*cplx)
else:
    shm_bandveck = (knum*norb*cplx if Ispin==2
                    else (Ispin+1)*knum*Np*cplx)
shm_win_L = 3*nspin*norbnb*cplx
shm_win_R = knum*3*nspin*norbnb*cplx

# physical totals of the DISTRIBUTED arrays (fixed total split across ranks;
# per-node = total/nnodes, NOT procs_pn*per-rank-max which over-counts the ceil
# load-imbalance). drSH splits Np, phvecval_p & epc_t split knum, ScaLAPACK is
# 2D block-cyclic. The NB_k batch bufs + py are REPLICATED (procs_pn*size).
epc_t_tot_b = kfactor*knum*nmnb2*cplx                       # split over knum
drSH_tot_b  = (4*3*ncell*Np*Np*cplx if Ispin==2 else (Ispin+1)*3*ncell*Np*Np*real)  # split Np
phv_tot_b   = knum*nmodes*nmodes*cplx                       # split over knum
sl_tot_b    = sl_peak*nprocs                                # numroc local x nprocs

# each phase: (name, SHM items[(lab,bytes)], DISTRIBUTED[(lab,total bytes)],
#             REPLICATED[(lab,GB/rank)]).  epc_nl.py has 3 peaks in exec order.
phases = [
 ('read/idx (sparse H/S/dr build)',
  [('ham H',ham_shm),('olp S',olp_shm),('dr(3xyz)',dr_shm),
   ('keyinfo coo',keyinfo_coo),('csr idx',csr)],
  [], [('py/mkl',py_priv_gb)]),
 ('olp_inv (ScaLAPACK Cholesky solve)',
  [('ham H',ham_shm),('olp S',olp_shm),('dr(3xyz)',dr_shm),
   ('csr/keyinfo idx',shm_olpinv-ham_shm-olp_shm-dr_shm)],
  [('drSH',drSH_tot_b),('ScaLAPACK',sl_tot_b)], [('py/mkl',py_priv_gb)]),
 ('compute (MPIepcNL_R, all-k drveck)',
  [('bandveck',shm_bandveck),('drveck window(all-k)',shm_win_R)],
  [('epc_t',epc_t_tot_b),('drSH',drSH_tot_b),('phvecval_p',phv_tot_b)],
  [('batch bufs(NB_k=%d)'%NB_k,buf_R_gb),('py/mkl',py_priv_gb)]),
]
def phase_peak(shm_items,dist_items,repl_items,nn,ppn):
    return (py_shared_gb + sum(v for _,v in shm_items)/GB
            + sum(t for _,t in dist_items)/nn/GB
            + ppn*sum(v for _,v in repl_items))
ptot = [phase_peak(s,d,r,nnodes,procs_pn) for _,s,d,r in phases]
pk   = int(np.argmax(ptot))
name,shm_items,dist_items,repl_items = phases[pk]
shm_gb  = sum(v for _,v in shm_items)/GB
dist_node_gb = sum(t for _,t in dist_items)/nnodes/GB
repl_gb = sum(v for _,v in repl_items)
peak_gb = ptot[pk]
shm_node_max = max(sum(v for _,v in s)/GB for _,s,_,_ in phases)

bar = '='*70
print(bar)
print(' epc_nl.py  peak memory  (%s)'%name)
print(bar)
print('config: %s   Ispin=%d  EpcType=%s  IsAllKlist=%s  IsAllVec=%s'
      %(cfgfile,Ispin,EpcType,IsAllKlist,IsAllVec))
print('Np=%d N=%d ncell=%d norb=%d nbands=%d nmodes=%d knum=%d nkpath=%d'
      %(Np,N,ncell,norb,nbands,nmodes,knum,nkpath))
print('layout: nprocs=%d nnodes=%d procs/node=%d   NB_k=%d   ScaLAPACK %dx%d(Mb%d Nb%d)'
      %(nprocs,nnodes,procs_pn,NB_k,nprow,npcol,Mb,Nb))
if not (IsAllKlist and EpcType=='A' and Ispin==2):
    print('NOTE: compute-buffer detail exact only for IsAllKlist/EpcType=A/Ispin=2; epc_t & drSH hold.')
print('phase peaks: read %.1f | olp_inv %.1f | compute %.1f GB  -> binds at %s'
      %(ptot[0],ptot[1],ptot[2],name.split(' ')[0]))
print('-'*70)
print('MPI-SHM  [per node, one copy]:')
for lab,v in sorted(shm_items,key=lambda x:-x[1]):
    if v/GB>=0.0005: print('    %-20s %9.3f GB'%(lab,v/GB))
print('    %-20s %9.3f GB'%('SHM TOTAL',shm_gb))
if shm_node_max>50.0:
    print('    >>> WARNING: per-node MPI-SHM up to %.1f GB > 50 GB (sparse H/S/dr in'%shm_node_max)
    print('        read/olp_inv, or bandveck+all-k drveck in compute). One copy/node.')
if dist_items:
    print('DISTRIBUTED (fixed total split across ranks; per-node = total/nnodes):')
    for lab,t in sorted(dist_items,key=lambda x:-x[1]):
        print('    %-14s total %8.2f GB | /node %7.2f | /rank avg %6.3f'
              %(lab,t/GB,t/nnodes/GB,t/nprocs/GB))
print('REPLICATED  [per rank, x%d ranks/node]:'%procs_pn)
for lab,v in sorted(repl_items,key=lambda x:-x[1]):
    if lab=='py/mkl': continue
    print('    %-14s %8.3f GB/rank'%(lab,v))
print('    %-14s %8.3f GB/rank  (private anon; +%.2f GB/node shared code)'
      %('py/mkl',py_priv_gb,py_shared_gb))
print('    %-14s %8.3f GB/rank  x%d = %8.2f GB/node'%('subtotal',repl_gb,procs_pn,procs_pn*repl_gb))
print('-'*70)
print('PER-NODE PEAK = SHM %.2f + py-shared %.2f + distributed/node %.2f + replicated/node %.2f = %.2f GB'
      %(shm_gb,py_shared_gb,dist_node_gb,procs_pn*repl_gb,peak_gb))
print('              (node RAM %.0f - %.1f os - %.1f writeback = %.1f usable)'
      %(node_ram_gb,os_reserve_gb,writeback_gb,usable_gb))
print('epc_t TOTAL (physical, all ranks)          = %8.2f GB   <-- distributed'%g(epc_t_tot_b))
print('-'*70)
if peak_gb>usable_gb:
    print('>>> OOM: %s = %.1f GB > %.1f GB usable.'%(name.split(' ')[0],peak_gb,usable_gb))
    print('    distributed/node %.1f (epc_t %.1f + drSH %.1f + phv %.1f) dominates.'
          %(dist_node_gb,g(epc_t_tot_b)/nnodes,g(drSH_tot_b)/nnodes,g(phv_tot_b)/nnodes))
    for nn in range(nnodes,nnodes+65):
        for ppn in (procs_pn,48,32,24,16,12,8,6,4,2,1):
            if nn*ppn<nprocs: continue
            pk_nn = max(phase_peak(s,d,r,nn,ppn) for _,s,d,r in phases)
            if pk_nn<=usable_gb:
                print('    -> %2d nodes x %2d ranks (%d procs): peak %.1f GB OK'%(nn,ppn,nn*ppn,pk_nn)); break
        else: continue
        break
    print('    (or h3cm 1510GB nodes.)')
else:
    print('>>> OK: peak %.1f GB fits (margin %.1f GB)'%(peak_gb,usable_gb-peak_gb))
print('adaptive NB_k=%d (optimal=ncell=%d, cap=%d); NB_k bufs %.3f GB/rank are replicated,'
      %(NB_k,ncell,NB_k_cap,buf_R_gb))
print('  dwarfed by epc_t/node %.1f GB, so NB_k is not a memory driver.'%(g(epc_t_tot_b)/nnodes))
print(bar)
