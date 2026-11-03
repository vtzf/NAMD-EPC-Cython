import os
import math
import numpy as np
import configparser

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
conf = configparser.ConfigParser()
conf.read(os.path.join(base,'lambda','postprocessing.ini'),encoding='utf-8')
nq_str = conf['main']['nq']
qmesh = np.array([int(i) for i in nq_str[1:-1].split(',')],dtype=np.int32)
qnum = qmesh[0]*qmesh[1]*qmesh[2]
sigma = float(conf['a2f']['sigma'])
mu = float(conf['a2f']['mu'])
nw = int(conf['a2f']['nw'])
Kelvin2eV = 0.086173427909

data = np.genfromtxt(os.path.join(base,'lambda',f'lambda_qnu_{qmesh[0]}'))
if data.ndim==1:
    data = data[None,:]
lamda = np.sum(data[:,3]*data[:,6]/qnum)
if lamda<=0:
    raise ValueError('Total lambda must be positive')
omega = data[:,5]
wqmax = max(omega)*1.1
if wqmax<=0:
    raise ValueError('No positive phonon frequency found')
wqstep = wqmax/nw

solution = []
ax = []
for i in range(1,nw+1):
    w = i*wqstep
    gauss = np.exp(-np.minimum((w-omega)**2/(2*sigma**2),200))/(sigma*math.sqrt(2*np.pi))
    solution.append(np.sum(data[:,3]*data[:,6]*omega*gauss*0.5/qnum))
    ax.append(w)
solution = np.array(solution)
ax = np.array(ax)
wlog = math.exp(2*wqstep*np.sum(solution*np.log(ax)/ax)/lamda)/Kelvin2eV
denom = lamda*(1-0.62*mu)-mu
if denom==0:
    raise ValueError('Allen-Dynes denominator is zero')
Tc = wlog/1.2*math.exp(-1.04*(1+lamda)/denom)

np.savetxt(
    os.path.join(base,'a2f','a2f.dat'),np.column_stack([ax,solution]),
    header='omega(meV) alpha2F',fmt='%.10e'
)
print(f'# lambda: {lamda}')
print(f'# wlog(meV): {wlog*Kelvin2eV}')
print(f'# Tc(K): {Tc}')
