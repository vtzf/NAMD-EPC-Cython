import numpy as np
import configparser
import sys

# ============================================================================
# Peak-memory estimate for bak1 epcsparse.py (sparse EPC engine, final code).
# Covers every branch: Ispin 0/1/2, dhamil_method C vs F/B, IsAllKlist T/F,
# EpcType A/K/Q (P = IsAllKlist False), per-jk vs scatter, NB_b1/NB_b2 batch.
#
# usage: python cal_memory.py [config.ini] [nnodes] [procs/node] [node_RAM_GB]
#
# epcsparse.py has ONE binding memory peak: the compute phase, where the
# per-node MPI-SHM windows, the private epc_t, and the NB_b1/NB_b2 batch
# buffers are all live at once (the earlier alloc phase is a strict subset).
# We report that single peak, itemized, plus the epc_t total on its own line.
# ============================================================================
cfgfile     = sys.argv[1] if len(sys.argv)>1 else 'config.ini'
nnodes      = int(sys.argv[2]) if len(sys.argv)>2 else 8
procs_pn    = int(sys.argv[3]) if len(sys.argv)>3 else 48          # ranks per node
node_ram_gb = float(sys.argv[4]) if len(sys.argv)>4 else 187.0     # h3c=187, h3cm=1510
nprocs      = nnodes*procs_pn
# ---- node RAM headroom: absolute reserve, not a flat fraction (shared w/ _nl)
# Idle `top` on these nodes: total-avail (non-reclaimable kernel slab/struct-page
# + daemons) is ~3.6 GiB on h3c(187) and ~8.4 GiB on h3cm(1546); daemons are
# negligible, buff/cache is reclaimable. os_reserve=max(4,0.6%*RAM) bounds both
# with a small cushion. writeback_gb absorbs dirty-page spikes during the ~TB
# epc_t write (no usable swap, OOM-killer unforgiving); lower once a real run
# confirms it fits.
os_reserve_gb = max(4.0,0.006*node_ram_gb)
writeback_gb  = 4.0
usable_gb     = node_ram_gb-os_reserve_gb-writeback_gb        # shared w/ _nl
# python+MKL was ~0.24 GiB VmHWM/rank (measured), but ~half is read-only
# .so/libmkl/interpreter text: ONE physical copy mmap-shared across all ranks on
# a node -> count once, not procs_pn*.  (shared w/ _nl)
py_shared_gb = 0.12             # shared code, once per node
py_priv_gb   = 0.12             # private anon heap, per rank
GB   = 1024.0**3
cplx = 16
real = 8

# ---- sparse structure sizes (from a prior GetSparseNum) --------------------
#   ns        = key_num[ncell2,3]  total nonzeros (== total_S for R_eff)
#   total_dhl = key_num_s[ncell,1] summed sparse length (key_info_s rows)
#   max_dhl   = max_j key_num_s[j,0] largest single-cell sparse length (matlen)
#   npair     = key_num[ncell2,2]  distinct atom-pair count (pub_key rows)
ns        = 102092036
total_dhl = 102092036
max_dhl   = total_dhl
npair     = 414376
try:
    kn  = np.load('key_num.npy');  ncell2_ld = kn.shape[0]-1
    ns  = int(kn[ncell2_ld,3]);    npair = int(kn[ncell2_ld,2])
    kns = np.load('key_num_s.npy')
    total_dhl = int(kns[kns.shape[0]-1,1])
    max_dhl   = int(kns[:kns.shape[0]-1,0].max())
    print('sparse sizes loaded from key_num.npy / key_num_s.npy')
except (OSError,ValueError):
    print('sparse sizes use in-script defaults; edit ns/total_dhl/max_dhl/npair for other systems')

# ---- config ----------------------------------------------------------------
conf = configparser.ConfigParser(); conf.read(cfgfile,encoding='utf-8')
Ispin         = int(conf['epc']['Ispin'])
dhamil_method = conf['epc']['dhamil_method']
EpcType       = conf['epc']['EpcType']
ucellidx = [int(i) for i in conf['epc']['ucellidx'][1:-1].split(',')]
atom     = [int(i) for i in conf['epc']['atom'][1:-1].split(',')]
orbital  = [int(i) for i in conf['epc']['orbital'][1:-1].split(',')]
IsAllVec   = conf['epc']['IsAllVec']=='True'
IsAllKlist = conf['epc']['IsAllKlist']=='True'
nm_block   = int(conf['mpi']['NMODES_BLOCK'])

def GenKlist(Kpoint_str,nq):
    from math import gcd
    k_list = Kpoint_str.replace('[','').replace(']','').split(',')
    k_arr  = np.array([eval(i) for i in k_list],dtype=float).reshape(-1,3)
    k_int  = np.around(k_arr*nq).astype(int)
    nklist = np.zeros((k_arr.shape[0]-1),dtype=int)
    for i in range(k_arr.shape[0]-1):
        d  = np.abs(k_int[i+1]-k_int[i]); dp = d[np.where(d>0)[0]]
        if   dp.shape[0]==3: nklist[i]=gcd(gcd(dp[0],dp[1]),gcd(dp[1],dp[2]))
        elif dp.shape[0]==2: nklist[i]=gcd(dp[0],dp[1])
        else:                nklist[i]=dp[0]
    return int(np.sum(nklist))

# ---- derived system params -------------------------------------------------
atomnum  = sum(atom)
nmodes   = atomnum*3
norbital = sum(x*y for x,y in zip(atom,orbital))
norb     = norbital if Ispin==0 else 2*norbital
nbands   = norb if IsAllVec else int(conf['epc']['bmax'])-int(conf['epc']['bmin'])+1
ncell    = ucellidx[0]*ucellidx[1]*ucellidx[2]
ncell2   = ncell*ncell
nq       = [int(i) for i in conf['epc']['nq'][1:-1].split(',')]
knum     = nq[0]*nq[1]*nq[2]
nm_buffer= nm_block+1 if nmodes%nm_block else nm_block
norbnb   = norbital*nbands
nbands2  = nbands*nbands
ceil_div = lambda a,b: -(-a//b)

shm_dh_ch   = 4 if Ispin==2 else (Ispin+1)      # dhamil/hamil_buf channels
shm_dh_elem = cplx if Ispin==2 else real
kn_nspin    = 2 if Ispin==2 else 1              # kernel-buffer spin factors
kn_nspin2   = 4 if Ispin==2 else 1
nspin_hv    = 2 if Ispin==2 else 1

def read_l3():
    try:
        s = open('/sys/devices/system/cpu/cpu0/cache/index3/size').read().strip()
        tot = int(s[:-1])*1024 if s.endswith('K') else \
              (int(s[:-1])*1024*1024 if s.endswith('M') else int(s))
        try:
            cl = open('/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list').read().strip()
            ncpu = 0
            for p in cl.split(','):
                a,b = (p.split('-')+[None])[:2]
                ncpu += (int(b)-int(a)+1) if b is not None else 1
            ncpu = max(ncpu,1)
        except OSError:
            ncpu = 1
        return tot, tot//ncpu
    except OSError:
        return 32*1024*1024, 2*1024*1024
socketL3, percpuL3 = read_l3()

# ---- per-jk vs scatter (exact _use_perjk criterion) ------------------------
R_eff = ns/float(total_dhl)
cache_penalty = max(max_dhl*kn_nspin*kn_nspin*16/float(percpuL3), 1.0)
r_thresh = cache_penalty*nbands/(nbands-1) if nbands>1 else cache_penalty
use_perjk = R_eff < r_thresh

# ---- epc_t: per-rank peak (each program's own partition) + physical total --
nmnb2 = nmodes*nbands*nbands
if IsAllKlist:
    row_len = nmnb2
    if EpcType=='Q':
        nkpath   = GenKlist(conf['epc']['Kpoint'],np.array(nq)); nqpath = nkpath
        rows_max = ceil_div(knum,nprocs)*nkpath                  # knum split, x nkpath
        knum2    = knum
    else:
        nkpath   = knum if EpcType=='A' else GenKlist(conf['epc']['Kpoint'],np.array(nq))
        knum2    = nkpath*knum
        rows_max = ceil_div(knum2,nprocs)                        # flat split of knum*nkpath
        nqpath   = 0
    total_rows = nkpath*knum
    knum_bv    = knum
else:                                                            # P path (bassel klist)
    row_len = nmodes
    try:
        bn = conf['epc']['basselname'].split('.')[0]
        knum_p = int(np.load(bn+('_up.npy' if Ispin==1 else '.npy')).shape[0])
    except (OSError,KeyError,ValueError):
        knum_p = knum
        print('IsAllKlist=False: bassel npy not found, knum_p defaults to knum=%d'%knum)
    knum2      = knum_p*knum_p
    rows_max   = ceil_div(knum2,nprocs)
    total_rows = knum_p*knum_p
    knum_bv    = knum_p; nkpath = 0; nqpath = 0

epc_t_b     = rows_max*row_len*cplx                              # per-rank peak
epc_t_tot_b = total_rows*row_len*cplx                            # physical total

# ---- adaptive NB_b1 / NB_b2 (exact epcsparse.py formulas) -------------------
if not IsAllKlist:
    nb1_raw = int(socketL3/(norbital*16*(2*nspin_hv+1)))
    cap     = (knum2//nprocs)//max(knum_p,1) + 2;   nb2_cap = knum
else:
    nb1_raw = int(socketL3/(norbital*nbands*16*(2*nspin_hv+1)))
    if EpcType=='Q':
        cap = ceil_div(knum,nprocs);                nb2_cap = nqpath
    else:
        cap = (knum2//nprocs)//max(knum,1) + 2;     nb2_cap = knum
NB_b1 = max(1,min(nb1_raw,cap))
NB_b2 = min(max(nm_buffer*ncell,32), nb2_cap)

# ---- per-node MPI-SHM (one copy per node), itemized ------------------------
shm_items = [
    ('key_num',    (ncell2+1)*4*4),   ('pub_key',  npair*6*4),
    ('key_info',   ns*2*4),           ('key_info1',ns*2*4),
    ('key_num_s',  (ncell+1)*2*4),    ('key_info_s',total_dhl*4),
    ('col_idx_jk', ns*4),             ('row_ptr_jk',ncell2*(norbital+1)*4),
]
if dhamil_method[0]!='C':
    shm_items.append(('hamil_buf(H0)', shm_dh_ch*ns*shm_dh_elem))
shm_items.append(('dhamil', nm_buffer*shm_dh_ch*ns*shm_dh_elem))
if IsAllKlist:
    bandveck_b = (Ispin+1)*knum*norbital*nbands*cplx if Ispin!=2 else knum*norb*nbands*cplx
else:
    bandveck_b = (Ispin+1)*knum_bv*norbital*cplx if Ispin!=2 else knum_bv*norb*cplx
shm_items.append(('bandveck', bandveck_b))
shm_items.append(('phvecval', knum*nm_buffer*nmodes*cplx))
shm_b  = sum(v for _,v in shm_items)
shm_gb = shm_b/GB

# ---- per-rank private compute buffers (NB_b1 / NB_b2 / fixed split) --------
hv_row = norbnb if IsAllKlist else norbital     # P has no band dim
b_nb1 = nm_buffer*ncell*hv_row*kn_nspin*cplx + ncell*cplx        # hamilveck + expikR_b1
if use_perjk:
    b_nb1 += (kn_nspin*hv_row + hv_row)*cplx                     # bandveck_b1 + dhveck_b1
    b_fix  = ncell2*nm_buffer*kn_nspin2*8                        # csrH_jk pointers
else:
    b_nb1 += kn_nspin2*max_dhl*cplx                              # dhexpikR
    b_fix  = ncell*(norbital+1)*4 + total_dhl*4                  # csr_rowptr + csr_cidx
b_nb2  = kn_nspin*hv_row*cplx + ncell*cplx                       # dhvexpikR_b2 + expikR_b2
b_fix += nbands2*cplx                                            # vdhvexpikR (tiny)
buf_b   = b_fix + b_nb1*NB_b1 + b_nb2*NB_b2
batch_b = b_nb1*NB_b1 + b_nb2*NB_b2

g = lambda x: x/GB
buf_gb = g(buf_b)
# epc_t is DISTRIBUTED (one physical array split across all ranks): its per-node
# footprint is total/nnodes, NOT procs_pn*per-rank-max (which would over-count the
# ceil load-imbalance). The compute buffers + py are REPLICATED (each rank a full
# working copy) -> procs_pn*size. Assumes ranks map ~evenly across nodes.
epc_t_avg_gb  = g(epc_t_tot_b)/nprocs     # per-rank average (== total/nprocs)
epc_t_node_gb = g(epc_t_tot_b)/nnodes     # distributed per-node footprint
repl_gb  = buf_gb + py_priv_gb            # replicated per-rank
peak_gb  = shm_gb + py_shared_gb + epc_t_node_gb + procs_pn*repl_gb

# ============================================================================
bar = '='*70
print(bar)
print(' epcsparse.py  peak memory  (compute phase)')
print(bar)
print('config: %s   Ispin=%d  EpcType=%s  IsAllKlist=%s  dhamil_method=%s'
      %(cfgfile,Ispin,EpcType,IsAllKlist,dhamil_method))
print('norbital=%d nbands=%d ncell=%d knum=%d nmodes=%d ns=%d  R_eff=%.3f->%s'
      %(norbital,nbands,ncell,knum,nmodes,ns,R_eff,'per-jk' if use_perjk else 'scatter'))
print('layout: nprocs=%d nnodes=%d procs/node=%d   NB_b1=%d NB_b2=%d  (socketL3=%.0fMB)'
      %(nprocs,nnodes,procs_pn,NB_b1,NB_b2,socketL3/1024/1024))
print('-'*70)
print('MPI-SHM  [per node, one copy]:')
for name,v in sorted(shm_items,key=lambda x:-x[1]):
    if v/GB>=0.0005: print('    %-16s %9.3f GB'%(name,v/GB))
print('    %-16s %9.3f GB'%('SHM TOTAL',shm_gb))
if shm_gb>50.0:
    print('    >>> WARNING: per-node MPI-SHM %.1f GB > 50 GB (dhamil x%d-ch + phvecval'
          %(shm_gb,shm_dh_ch))
    print('        dominate; lower NMODES_BLOCK to shrink dhamil). One copy/node.')
if not IsAllKlist:   part = 'flat ceil(knum_p^2/nprocs)'
elif EpcType=='Q':   part = 'ceil(knum/nprocs)*nqpath'
else:                part = 'flat ceil(knum*nkpath/nprocs)'
print('DISTRIBUTED (fixed total split across ranks; per-node = total/nnodes):')
print('    %-14s total %8.2f GB | /node %8.2f | /rank avg %7.3f  (%s)'
      %('epc_t',g(epc_t_tot_b),epc_t_node_gb,epc_t_avg_gb,part))
print('REPLICATED  [per rank, x%d ranks/node]:'%procs_pn)
print('    %-14s %8.3f GB/rank  (NB_b1 %.3f + NB_b2 %.3f + fixed %.3f)'
      %('compute bufs',buf_gb,g(b_nb1*NB_b1),g(b_nb2*NB_b2),g(b_fix)))
print('    %-14s %8.3f GB/rank  (private anon; +%.2f GB/node shared code)'
      %('py/mkl',py_priv_gb,py_shared_gb))
print('    %-14s %8.3f GB/rank  x%d = %8.2f GB/node'%('subtotal',repl_gb,procs_pn,procs_pn*repl_gb))
print('-'*70)
print('PER-NODE PEAK = SHM %.2f + py-shared %.2f + epc_t/node %.2f + replicated/node %.2f = %.2f GB'
      %(shm_gb,py_shared_gb,epc_t_node_gb,procs_pn*repl_gb,peak_gb))
print('              (node RAM %.0f - %.1f os - %.1f writeback = %.1f usable)'
      %(node_ram_gb,os_reserve_gb,writeback_gb,usable_gb))
print('-'*70)

# ---- verdict + adaptive-param range hint -----------------------------------
if peak_gb > usable_gb:
    print('>>> OOM: peak %.1f GB > %.1f GB usable.'%(peak_gb,usable_gb))
    # per-rank budget left for the replicated NB batch buffers
    budget = (usable_gb - shm_gb - py_shared_gb - epc_t_node_gb)/procs_pn - py_priv_gb - g(b_fix)
    if budget<=0:
        print('    epc_t/node + SHM alone overflow: raise nnodes (cuts epc_t/node) or fewer ranks/node.')
    else:
        nb1_fit = int((budget - g(b_nb2*NB_b2))/g(b_nb1)) if b_nb1>0 else 0
        nb2_fit = int((budget - g(b_nb1*NB_b1))/g(b_nb2)) if b_nb2>0 else 0
        print('    adaptive range that fits this layout:')
        print('      NB_b1 <= %d  (auto %d; only trades batch width for memory, safe to cap)'
              %(max(nb1_fit,1),NB_b1))
        print('      NB_b2 <= %d  (auto %d)'%(max(nb2_fit,1),NB_b2))
    for nn in range(nnodes,nnodes+25):
        for ppn in (procs_pn,48,24,16,12,8):
            if nn*ppn<nprocs: continue
            p = shm_gb + py_shared_gb + g(epc_t_tot_b)/nn + ppn*repl_gb
            if p<=usable_gb:
                print('    -> %2d nodes x %2d ranks (%d procs): peak %.1f GB OK'%(nn,ppn,nn*ppn,p)); break
        else: continue
        break
else:
    print('>>> OK: peak %.1f GB fits (margin %.1f GB)'%(peak_gb,usable_gb-peak_gb))
print(bar)
