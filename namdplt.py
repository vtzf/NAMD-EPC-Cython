#!/usr/bin/env python

import math, os
import numpy as np
import postnamd as pn
import matplotlib as mpl; mpl.use('agg')
import matplotlib.pyplot as plt
from glob import glob

def main():

    #                       Basic parameters setting                      #
    #######################################################################

    Eref = 0.0
    which_plt = [1, 2, 4, 5, 6, 11, 12]
    #which_plt = [4,5]
    '''
    Select which figures to plot.
    1: COUPLE.png; 11:COUPLE_EL.png; 12:COUPLE_PH.png; 2: TDEN.png;
    31: TDKPROPxy.png; 32: TDKPROPyz.png; 33: TDKPROPxz.png;
    4: TDBAND.png; 5: TDPH.png; 6:TDPHEN.png
    7: DISTRIBUTION.png; 8: TPROP.png
    '''

    kplabels = 'gmkg'
    kpath = np.array([ # for TDBAND.png
        [0.00000, 0.00000, 0.00000],
        [0.50000, 0.00000, 0.00000],
        [0.33333, 0.33333, 0.00000],
        [0.00000, 0.00000, 0.00000]
        ])
    qpath = kpath # for TDPH.png
    qplabels = kplabels

    #                              Read data                              #
    #######################################################################

    print("\nReading data ...")
    inp = pn.read_inp('inp')
    prefix = inp['EPMPREF']; epmdir = inp['EPMDIR']
    filepm = os.path.join(epmdir, prefix + '_ephmat_p1.h5')
    A = pn.read_ephmath5(filepm, dset='/el_ph_band_info/lattice_vec_angstrom')
    a1, a2, a3 = (A[0], A[1], A[2])
    b1, b2, b3 = pn.calc_rec_vec(a1, a2, a3)

    #filbassel='BASSEL'
    #bassel  = np.loadtxt(filbassel, dtype=int, skiprows=1)
    #b_index = bassel[:, 1] - 1

    bassel = np.load('bassel.npy')+1
    b_index = bassel[:,1] - 1

    en, kpts = pn.ek_selected(bassel=bassel,inp=inp) # en & kpts selected
    Enk = pn.get_Enk(kpath, B=[b1, b2, b3], inp=inp) # total Enk
    kpts_cart = pn.frac2cart(kpts, b1, b2, b3)
    kpath_cart = pn.frac2cart(kpath, b1, b2, b3)
    k_index, kp_index = pn.select_kpts_on_path(kpts, kpath, norm=0.001)
    k_loc, kp_loc = pn.loc_on_kpath(kpts_cart, k_index, kp_index, kpath_cart)
    qpts = pn.read_ephmath5(filepm, dset='/el_ph_band_info/q_list')
    phen = pn.read_ephmath5(filepm, dset='/el_ph_band_info/ph_disp_meV')
    qpts_cart = pn.frac2cart(qpts, b1, b2, b3)
    qpath_cart = pn.frac2cart(qpath, b1, b2, b3)
    q_index, qp_index = pn.select_kpts_on_path(qpts, qpath, norm=0.001)
    q_loc, qp_loc = pn.loc_on_kpath(qpts_cart, q_index, qp_index, qpath_cart)

    #filshps = glob('SHPROP.*')
    #if filshps:
    #    shp = pn.readshp(filshps)
    #    ntsteps = shp.shape[0]
    #    namdtime = shp[-1,0]
    #else:
    #    shp = None ; ntsteps = 0 ; namdtime = 0.0
    #    print('\nERROR: SHPROP files are not found!')

    fileshp_e = glob('fssh_e_sh-*.npy')
    fileshp_pop = glob('fssh_pop_sh-*.npy')
    shp_e = []
    shp_pop = []
    for i in range(len(fileshp_e)):
        shp_e.append(np.load(fileshp_e[i]))
        shp_pop.append(np.load(fileshp_pop[i]))
    shp_e = np.average(shp_e,axis=0)
    shp_pop = np.average(shp_pop,axis=0)
    ntsteps = shp_e.shape[0]
    namdtime = ntsteps
    nbasis = shp_pop.shape[1]
    shp = np.zeros((namdtime,shp_pop.shape[1]+2))
    shp[:,0] = np.arange(1,namdtime+1)
    shp[:,1] = shp_e
    shp[:,2:] = shp_pop


    #                             Plot figures                            #
    #######################################################################

    if ((1 in which_plt) or (11 in which_plt)):
        #coup = pn.read_couple(filcoup='EPECTXT', inp=inp)
        coup = np.load('epcec-0.npy')
        coup = coup[np.newaxis]
        coup = coup * 1000.0 # change unit to meV
        coup_av = np.average(np.abs(coup), axis=0)
        print(coup_av.shape)
    if (1 in which_plt):
        plot_couple(coup_av, figname='COUPLE.png')
    if (11 in which_plt):
        plot_couple_el(coup_av, k_loc, en, kp_loc, kplabels, k_index, Enk,
                       Eref, figname='COUPLE_EL.png')
    if (12 in which_plt):
        if os.path.isfile('epcph-0.npy'):
            #coup_ph = np.loadtxt('EPPHTXT')
            coup_ph = np.load('epcph-0.npy').T
            coup_ph *= 1000.0 # change unit to meV
            plot_coup_ph(coup_ph, q_loc, phen, qp_loc, qplabels, q_index,
                         figname='COUPLE_PH.png')
        else:
            print("\nERROR: EPPHTXT file is not found!")

    if (2 in which_plt):
        plot_tdprop(shp, Eref, lplot=2, ksen=en, figname='TDEN.png')

    if (31 in which_plt):
        plot_kprop(kpts_cart, shp, B=[b1, b2, b3], axis='xy', figname='TDKPROPxy.png')
    if (32 in which_plt):
        plot_kprop(kpts_cart, shp, B=[b1, b2, b3], axis='yz', figname='TDKPROPyz.png')
    if (33 in which_plt):
        plot_kprop(kpts_cart, shp, B=[b1, b2, b3], axis='xz', figname='TDKPROPxz.png')

    if (4 in which_plt):
        times = [80, 250, 500, 750, 1000]
        #times = list( range(0, int(namdtime)+1, int(namdtime/4)) )
        plot_tdband_sns(k_loc, en, kp_loc, kplabels, shp, k_index, times,
                        b_index=b_index, Eref=Eref, figname='TDBAND.png')


    if ((5 in which_plt) or (6 in which_plt)):
        #if not os.path.isfile('PHPROP'):
        #    filphps = glob('PHPROP.*')
        #    php = pn.readphp(filphps)
        #else:
        #    php = np.loadtxt('PHPROP')
        #    nmodes = int( php.shape[0] / ntsteps ) ; nqs = php.shape[1] - 2
        #    php = php.reshape(nmodes, ntsteps, nqs+2)

        filephp_e = glob('fssh_e_ph-*.npy')
        filephp_pop = glob('fssh_pop_ph-*.npy')
        php_e = []
        php_pop = []
        for i in range(len(filephp_e)):
            php_e.append(np.load(filephp_e[i]))
            php_pop.append(np.load(filephp_pop[i]))
        php_e = np.average(php_e,axis=0)
        php_pop = np.average(php_pop,axis=0)
        nmodes = php_pop.shape[1]; nqs = php_pop.shape[2]
        php = np.zeros((nmodes,ntsteps,nqs+2))
        php[:,:,0] = np.arange(1,namdtime+1)
        php[:,:,1] = php_e.T
        php[:,:,2:] = np.swapaxes(php_pop,0,1)

    if (5 in which_plt):
        # times = [0, 50, 100, 200, 500, 1000]
        times = list( range(0, int(namdtime)+1, int(namdtime/4)) )
        plot_tdph_sns(q_loc, phen, qp_loc, qplabels, php, q_index, times, figname='TDPH.png')

    if (6 in which_plt):
        plot_tdphen(php, figname='TDPHEN.png')

    if (7 in which_plt):
        times = list( range(0, int(namdtime)+1, int(namdtime/4)) )
        plot_distrib(en, shp, times, Ef=Eref, figname='DISTRIBUTION.png')

    if (8 in which_plt):
        plot_Tprop(en, shp, Ef=Eref, figname='TPROP.png')

    print("\nDone!\n")


###############################################################################

def plot_couple_el(coup_in, k_loc, en, kp_loc, kplabels, index, Enk,
                   Eref=0.0, figname='COUPLE_EL.png'):

    X = k_loc
    E = en[index] - Eref
    coup = np.sum(coup_in, axis=1)[index]
    nbands = Enk.shape[1] - 1

    xmin = kp_loc[0] ; xmax = kp_loc[-1]
    ymin = E.min() ; ymax = E.max() ; dy = ymax - ymin
    ymin -= dy*0.05 ; ymax += dy*0.05

    # cmap = 'hot_r'
    # cmin = np.min(coup[coup>0.0]); cmax = np.max(coup)
    # norm = mpl.colors.LogNorm(cmin,cmax)
    cmap = 'rainbow'
    cmin = 0.0; cmax = np.max(coup)
    norm = mpl.colors.Normalize(cmin,cmax)

    figsize_x = 4.2
    figsize_y = 4.8 # in inches
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize_x, figsize_y)
    mpl.rcParams['axes.unicode_minus'] = False

    sc = ax.scatter(X, E, s=30, lw=0,
            c=coup, cmap=cmap, norm=norm)

    for ib in range(nbands):
        ax.plot(Enk[:,0], Enk[:,ib+1]-Eref, '#1A5599', lw=0.7)

    nkpath = kp_loc.shape[0]
    for ipath in range(1, nkpath):
        x = kp_loc[ipath]
        ax.plot([x,x], [ymin,ymax], 'k', lw=0.7, ls='--')

    ticks = []
    for s in kplabels:
        s = u'\u0393' if (s=='g' or s=='G') else s.upper()
        ticks.append(s)

    ax.set_xticks(kp_loc)
    ax.set_xticklabels(ticks)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Energy (eV)')

    cbar = plt.colorbar(sc, aspect=30)
    cbar.set_label('Coupling (meV)')
    plt.tight_layout()
    plt.savefig(figname, dpi=400)
    print("\n%s has been saved."%figname)


def plot_couple(coup, figname='COUPLE.png'):
    '''
    This function plots average couplings.

    Parameters:
    coup   : ndarray, average coupling data in forms of coup[nb, nb]
    figname: string, output figure file name.
    '''

    fig = plt.figure()
    figsize_x = 4.8
    figsize_y = 3.6 # in inches
    fig.set_size_inches(figsize_x, figsize_y)

    cmap = 'bwr'
    n = coup.shape[0]
    Bmin = 0.5; Bmax = n + 0.5
    cmin = 0.0; cmax = np.max(coup)
    norm = mpl.colors.Normalize(cmin,cmax)
    plt.imshow(coup, cmap=cmap, origin='lower', norm=norm,
        extent=(Bmin,Bmax,Bmin,Bmax), interpolation='none')

    cbar = plt.colorbar()
    # cbar.ax.set_title('   meV')
    cbar.set_label('Coupling (meV)')
    plt.tight_layout()
    plt.savefig(figname, dpi=400)
    print("\n%s has been saved."%figname)


def plot_coup_ph(coup_ph, q_loc, phen, qp_loc, qplabels, index,
        figname='COUPLE_PH.png'):

    X = q_loc
    E = phen[index, :]
    coup = coup_ph[index, :]
    nmodes = phen.shape[1]

    xmin = qp_loc[0] ; xmax = qp_loc[-1]
    ymin = E.min() ; ymax = E.max() ; dy = ymax - ymin
    ymin -= dy*0.05 ; ymax += dy*0.05

    figsize_x = 4.2
    figsize_y = 4.8 # in inches
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize_x, figsize_y)
    mpl.rcParams['axes.unicode_minus'] = False

    cmap = 'rainbow'
    cmin = coup.min() ; cmax = coup.max()
    norm = mpl.colors.Normalize(cmin, cmax)
    # cmap = 'plasma'
    # cmin = np.min(coup[coup>0.0]); cmax = np.max(coup)
    # norm = mpl.colors.LogNorm(cmin,cmax)

    sort = np.argsort(X)
    for im in range(nmodes):
        ax.plot(X[sort], E[sort, im], '#1A5599', lw=0.7)

    nqpath = qp_loc.shape[0]
    for ipath in range(1, nqpath):
        x = qp_loc[ipath]
        ax.plot([x,x], [ymin,ymax], 'gray', lw=0.7, ls='--')

    for im in range(nmodes):
        sort = np.argsort(coup[:,im])
        sc = ax.scatter(X[sort], E[sort,im], s=30, lw=0,
                c=coup[sort,im], cmap=cmap, norm=norm)

    ticks = []
    for s in qplabels:
        s = u'\u0393' if (s=='g' or s=='G') else s.upper()
        ticks.append(s)

    ax.set_xticks(qp_loc)
    ax.set_xticklabels(ticks)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel('Phonon energy (meV)')

    cbar = plt.colorbar(sc, aspect=30)
    cbar.set_label('Coupling (meV)')

    plt.tight_layout()
    plt.savefig(figname, dpi=400)
    print("\n%s has been saved."%figname)


def plot_tdprop(shp, Eref=0.0, lplot=1, ksen=None, figname='tdshp.png'):
    '''
    This function loads data from SHPROP.xxx files,
    and plot average evolution of energy & surface hopping proportion of
    electronic states.

    Parameters:
    shp    : ndarray, average data of SHPROP.xxx files, in forms of
             shp[ntsteps, nb+2].
    Eref   : float, energy reference. Make sure shp & ksen have same Eref!!!
    lplot  : integer, fig type to plot. 1: plot average proportions; 2: plot
             average energy evolution with proportions.
    ksen   : ndarray, KS energies in forms of ksen[nbands]. Here we suppose
             ksen do not change by time.
    figname: string, file name of output figure.
    '''

    figsize_x = 4.8
    figsize_y = 3.2 # in inches
    namdtime = shp[-1,0]

    fig, ax = plt.subplots()
    fig.set_size_inches(figsize_x, figsize_y)
    mpl.rcParams['axes.unicode_minus'] = False

    if lplot==1:
        ylabel = 'SHPROP'
        ax.plot(shp[:,0], shp[:,2:])
        ax.set_ylim(0.0,1.0)
        ax.set_ylabel(ylabel)
    else:
        cmap = 'hot_r'
        dotsize = 30
        ylabel = 'Energy (eV)'
        ntsteps = shp.shape[0]
        nbands = shp.shape[1] -2

        # cmin = 0.0; cmax = np.max(shp[:,2:])
        # cmax = math.ceil(cmax*10)/10
        # norm = mpl.colors.Normalize(cmin,cmax)
        pop = shp[:,2:]
        cmin = np.min(pop[pop>0.0]); cmax = np.max(pop)
        norm = mpl.colors.LogNorm(cmin,cmax)

        if (ksen.shape[0]!=nbands):
            print('\nNumber of ksen states doesn\'t match with SHPROP data!\n')
        sort = np.argsort(np.sum(pop, axis=0))
        E = np.tile(ksen-Eref, ntsteps).reshape(ntsteps, nbands)
        T = np.tile(shp[:,0], nbands).reshape(nbands,ntsteps).T
        sc = ax.scatter(T, E[:,sort], s=dotsize, c=pop[:, sort], lw=0,
                        norm=norm, cmap=cmap)
        ax.plot(shp[:,0], shp[:,1]-Eref, 'b', lw=1, label='Average Energy')
        plt.colorbar(sc)

        # x1 = 0.05 * namdtime; x2 = 0.1 * namdtime
        # for ib in range(nbands):
        #     y = ksen[ib] - Eref
        #     ax.plot([x1, x2], [y, y], color='r', lw=0.7)

        ax.set_ylabel(ylabel)

    ax.set_xlim(0,namdtime)
    ax.set_ylim(int(shp[:,1].min())-1,int(shp[:,1].max())+1)
    ax.set_xlabel('Time (fs)')

    plt.tight_layout()
    plt.savefig(figname, dpi=400)
    print("\n%s has been saved."%figname)


def plot_kprop(kpts, shp, B, axis='xy', figname='TDKPROP.png'):

    axdict = {'x':0, 'y':1, 'z':2}
    # axis must be set as 'xy', 'yz' or 'xz'!
    kax = [axdict[s] for s in axis]

    gamma = [0, 0, 0]
    Bpath = np.array([gamma, B[kax[0]], B[kax[0]]+B[kax[1]], B[kax[1]], gamma])
    xmin = Bpath[:,kax[0]].min(); xmax = Bpath[:,kax[0]].max()
    ymin = Bpath[:,kax[1]].min(); ymax = Bpath[:,kax[1]].max()

    namdtime = shp[-1,0]
    ntsteps = shp.shape[0]
    potim = namdtime / ntsteps
    nbasis = shp.shape[1] - 2

    scale = np.sqrt( 12.0 / ((ymax - ymin) * (xmax - xmin)) )
    figsize_x = (xmax - xmin) * scale * 1.3
    figsize_y = (ymax - ymin) * scale # in inches

    fig, ax = plt.subplots()
    fig.set_size_inches(figsize_x, figsize_y)
    mpl.rcParams['axes.unicode_minus'] = False

    cmap = 'rainbow'
    norm = mpl.colors.Normalize(0,namdtime)
    color_t = np.tile(np.arange(ntsteps), nbasis).reshape(nbasis,ntsteps).T * potim
    s_avg = np.average(shp[:,2:][shp[:,2:]>0])
    dotsize = shp[:,2:] / s_avg * 0.5

    X = np.tile(kpts[:,kax[0]], ntsteps).reshape(ntsteps,nbasis)
    Y = np.tile(kpts[:,kax[1]], ntsteps).reshape(ntsteps,nbasis)

    ax.plot(Bpath[:, kax[0]], Bpath[:, kax[1]], color='k', lw=1.0)

    sc = ax.scatter(X, Y, s=dotsize, lw=0, c=color_t, norm=norm, cmap=cmap)
    cbar = plt.colorbar(sc, fraction=0.05)
    cbar.set_label('Time (fs)')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(figname, dpi=600)
    print("\n%s has been saved."%figname)


def plot_tdband(k_loc, en, kp_loc, kplabels, shp, index,
        X_bg=None, E_bg=None, Eref=0.0, figname='TDBAND.png'):

    nbasis = index.shape[0]
    ntsteps = shp.shape[0]
    namdtime = shp[-1, 0]
    potim = namdtime / ntsteps
    pop = shp[:, index+2]

    X = np.tile(k_loc, ntsteps).reshape(ntsteps, nbasis)
    E = np.tile(en[index], ntsteps).reshape(ntsteps, nbasis) - Eref

    xmin = kp_loc[0] ; xmax = kp_loc[-1]
    ymin = E.min() ; ymax = E.max() ; dy = ymax - ymin
    ymin -= dy*0.05 ; ymax += dy*0.05

    figsize_x = 4.8
    figsize_y = 3.6 # in inches
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize_x, figsize_y)
    mpl.rcParams['axes.unicode_minus'] = False

    cmap = 'rainbow'
    norm = mpl.colors.Normalize(0,namdtime)
    color_t = np.tile(np.arange(ntsteps), nbasis).reshape(nbasis,ntsteps).T * potim
    s_avg = np.average(pop[pop>0])
    dotsize = pop / s_avg * 5

    nkpath = kp_loc.shape[0]
    for ipath in range(1, nkpath):
        x = kp_loc[ipath]
        ax.plot([x,x], [ymin,ymax], 'k', lw=0.7, ls='--')

    sc = ax.scatter(X, E, s=dotsize, lw=0, c=color_t, cmap=cmap, norm=norm)
    cbar = plt.colorbar(sc, fraction=0.05)
    cbar.set_label('Time (fs)')

    ticks = []
    for s in kplabels:
        s = u'\u0393' if (s=='g' or s=='G') else s.upper()
        ticks.append(s)

    ax.set_xticks(kp_loc)
    ax.set_xticklabels(ticks)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel('Energy (eV)')

    plt.tight_layout()
    plt.savefig(figname, dpi=400)
    print("\n%s has been saved."%figname)


def plot_tdband_sns(k_loc, en, kp_loc, kplabels, shp, index, times,
        b_index=None, X_bg=None, E_bg=None, Eref=0.0, figname='TDBAND.png'):

    nbasis = index.shape[0]
    ntsteps = shp.shape[0]
    namdtime = shp[-1, 0]
    potim = namdtime / ntsteps

    tindex = []
    for time in times:
        if time==0: time = potim
        tindex.append(int(time/potim)-1)
    pop = shp[:,index+2][tindex,:]
    nts = len(times)

    X = k_loc
    E = en[index] - Eref

    xmin = kp_loc[0] ; xmax = kp_loc[-1]
    ymin = E.min() ; ymax = E.max() ; dy = ymax - ymin
    ymin -= dy*0.05 ; ymax += dy*0.05

    figsize_x = 2.4 * nts + 1.0
    figsize_y = 3.2 # in inches
    fig, axes = plt.subplots(1, nts)
    fig.set_size_inches(figsize_x, figsize_y)
    mpl.rcParams['axes.unicode_minus'] = False

    cmap = 'hot_r'
    cmin = np.min(pop[pop>0.0]); cmax = np.max(pop)
    norm = mpl.colors.LogNorm(cmin,cmax)

    for it in range(nts):

        ax = axes[it]
        ax.set_title('%.0f fs'%times[it])

        # plot background E-k lines
        if b_index is not None:
            if (it==0):
                sort_index = np.argsort(k_loc)
                X_bg = k_loc[sort_index]
                E_bg = en[index][sort_index] - Eref
                b_ind_bg = b_index[index][sort_index]

            for ib in np.unique(b_ind_bg):
                ax.plot(X_bg[b_ind_bg==ib], E_bg[b_ind_bg==ib], '#1A5599', lw=0.7)

        sc = ax.scatter(X, E, s=50, lw=0,
                c=pop[it,:], cmap=cmap, norm=norm)

        nkpath = kp_loc.shape[0]
        for ipath in range(1, nkpath):
            x = kp_loc[ipath]
            ax.plot([x,x], [ymin,ymax], 'k', lw=0.7, ls='--')

        ticks = []
        for s in kplabels:
            s = u'\u0393' if (s=='g' or s=='G') else s.upper()
            ticks.append(s)

        ax.set_xticks(kp_loc)
        ax.set_xticklabels(ticks)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if (it==0):
            ax.set_ylabel('Energy (eV)')

    ll = 0.8/figsize_x ; rr = 1.0 - 1.0 / figsize_x
    ll_cb = 1.0 - 0.9 / figsize_x ; ww_cb = 0.12 / figsize_x
    fig.subplots_adjust(bottom=0.1, top=0.9, left=ll, right=rr, wspace=0.35)
    cb_ax = fig.add_axes([ll_cb, 0.1, ww_cb, 0.8])
    cbar = plt.colorbar(sc, cax=cb_ax)

    plt.savefig(figname, dpi=400)
    print("\n%s has been saved."%figname)


def plot_tdph_sns(q_loc, phen, qp_loc, qplabels, php, index, times,
                  X_bg=None, E_bg=None, figname='TDPH.png'):

    nmodes = php.shape[0]
    ntsteps = php.shape[1]
    nbasis = index.shape[0]
    namdtime = php[0, -1, 0]
    potim = namdtime / ntsteps
    php = np.cumsum(php, axis=1)

    tindex = []
    for time in times:
        if time==0: time = potim
        tindex.append(int(time/potim)-1)
    pop = php[:,:,index+2][:,tindex,:]
    nts = len(times)

    # calculate ph number variation between it-1 and it.
    for ii in range(nts-1, 1, -1):
        pop[:,ii,:] = pop[:,ii,:] - pop[:,ii-1,:]

    X = q_loc
    E = phen[index, :]

    xmin = qp_loc[0] ; xmax = qp_loc[-1]
    ymin = E.min() ; ymax = E.max() ; dy = ymax - ymin
    ymin -= dy*0.05 ; ymax += dy*0.05

    figsize_x = 2.4 * nts + 1.0
    figsize_y = 3.2 # in inches
    fig, axes = plt.subplots(1, nts)
    fig.set_size_inches(figsize_x, figsize_y)
    mpl.rcParams['axes.unicode_minus'] = False

    cmin = pop.min() ; cmax = pop.max()
    norm = mpl.colors.Normalize(cmin, cmax)
    cmap = genrcmap(cmin, cmax)
    # cmap = 'hot_r'
    # cmax = np.max(pop) * 1.2 ; cmin = cmax / 1000
    # norm = mpl.colors.LogNorm(cmin,cmax)
    size = np.sqrt(np.abs(pop))
    s_avg = np.average(size[size>0])
    size = size / size.max()  * 50



    for it in range(nts):

        ax = axes[it]
        ax.set_title('%.0f fs'%times[it])

        sort_index = np.argsort(q_loc)
        for im in range(nmodes):
            # ax.plot(q_loc[sort_index], E[sort_index, im], '#1A5599', lw=0.7)
            ax.plot(q_loc[sort_index], E[sort_index, im], 'gray', lw=0.7)
        nqpath = qp_loc.shape[0]
        for ipath in range(1, nqpath):
            x = qp_loc[ipath]
            ax.plot([x,x], [ymin,ymax], 'gray', lw=0.7, ls='--')

        for im in range(nmodes):
            sort = np.argsort(pop[im,it,:])
            sc = ax.scatter(X[sort], E[sort,im], s=size[im,it,sort], lw=0,
            # sc = ax.scatter(X[sort], E[sort,im], s=50, lw=0,
                            c=pop[im,it,sort], cmap=cmap, norm=norm)

        ticks = []
        for s in qplabels:
            s = u'\u0393' if (s=='g' or s=='G') else s.upper()
            ticks.append(s)

        ax.set_xticks(qp_loc)
        ax.set_xticklabels(ticks)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        if(it==0): ax.set_ylabel('Phonon energy (meV)')

    ll = 0.8/figsize_x ; rr = 1.0 - 1.0 / figsize_x
    ll_cb = 1.0 - 0.9 / figsize_x ; ww_cb = 0.12 / figsize_x
    fig.subplots_adjust(bottom=0.1, top=0.9, left=ll, right=rr, wspace=0.35)
    cb_ax = fig.add_axes([ll_cb, 0.1, ww_cb, 0.8])
    cbar = plt.colorbar(sc, cax=cb_ax)

    plt.savefig(figname, dpi=400)
    print("\n%s has been saved."%figname)


def genrcmap(cmin=-1.0, cmax=1.0):

    num_pos = 100
    num_neg = int( np.abs(cmin) / cmax * num_pos )
    index = np.hstack((np.linspace(0.1, 0.5, num_neg),
                       np.linspace(0.5, 0.9, num_pos)))
    if (cmin>=0): index = np.linspace(0.5, 0.9, num_pos)

    cmap = plt.cm.jet
    clist = [cmap(i) for i in index]

    cmap_new = mpl.colors.LinearSegmentedColormap.from_list(
                   name='mycmap', colors=clist)

    return cmap_new


def plot_tdphen(php, figname='TDPHEN.png'):

    phen = php[:, :, 1].T
    phen = np.cumsum(np.cumsum(phen, axis=0), axis=1)
    nmodes = phen.shape[1]
    X = php[0, :, 0]
    namdtime = X[-1]

    figsize_x = 4.8
    figsize_y = 3.2 # in inches

    fig, ax = plt.subplots()
    fig.set_size_inches(figsize_x, figsize_y)
    mpl.rcParams['axes.unicode_minus'] = False

    ax.fill_between(X, phen[:, 0])
    for im in range(1, nmodes):
        ax.fill_between(X, phen[:, im], phen[:, im-1])

    ax.set_xlim(0,namdtime)
    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Energy (eV)')

    plt.tight_layout()
    plt.savefig(figname, dpi=400)
    print("\n%s has been saved."%figname)


def plot_distrib(en, shp, times, Ef=0.0, figname='DISTRIBUTION.png'):

    sort = np.argsort(en)
    en = en[sort] - Ef
    enfit = np.arange(en.min(), en.max(), 0.01)

    ntsteps = shp.shape[0]
    namdtime = shp[-1, 0]
    potim = namdtime / ntsteps
    tindex = []
    for time in times:
        if time==0: time = potim
        tindex.append(int(time/potim)-1)
    pop = shp[tindex, 2:][:,sort]

    nsns = len(times)
    ntsteps = shp.shape[0]
    namdtime = shp[-1, 0]
    potim = namdtime / ntsteps

    figsize_x = 4.8
    figsize_y = 1.2 * nsns # in inches

    fig, axes = plt.subplots(nsns, 1)
    fig.set_size_inches(figsize_x, figsize_y)
    fig.subplots_adjust(hspace=0.1)
    mpl.rcParams['axes.unicode_minus'] = False

    for  ii, time in enumerate(times):

        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(pn.func_fd, en, pop[ii], p0=[1000])
        fdfit = pn.func_fd(enfit, *popt)

        ax = axes if (nsns==1) else axes[ii]

        ax.plot([0,0], [-0.1,1.1], 'k', ls='--', lw=0.5)
        ax.plot(enfit, fdfit, lw=1.0, color='b', alpha=0.8)
        ax.scatter(en, pop[ii], s=8.0, lw=0.0, color='r', alpha=0.5)

        s = "%d fs, %d K"%(time, popt[0])
        ax.text(0.97, 0.90, s, ha='right', va='top',
                bbox=dict(boxstyle='round', ec='k', alpha=0.1),
                transform=ax.transAxes
                )

        ax.set_ylim(-0.1, 1.1)
        ax.set_ylabel('Distribution')

        if ii<nsns-1:
            ax.set_xticks([])
        else:
            ax.set_xlabel('$E-E_f$ (eV)')

    plt.tight_layout()
    plt.savefig(figname, dpi=400)
    print("\n%s has been saved."%figname)


def plot_Tprop(en, shp, Ef=0.0, figname='TPROP.png'):

    en -= Ef
    ntsteps = shp.shape[0]
    namdtime = shp[-1, 0]
    potim = namdtime / ntsteps

    from scipy.optimize import curve_fit
    itime = np.arange(0, ntsteps, 20)
    time = shp[itime, 0]
    pop = shp[:, 2:]

    temp = []
    for it in itime:
        popt, pcov = curve_fit(pn.func_fd, en, pop[it], p0=[1000])
        temp.append(popt[0])
    temp = np.array(temp)

    figsize_x = 4.8
    figsize_y = 3.2 # in inches

    fig, ax = plt.subplots()
    fig.set_size_inches(figsize_x, figsize_y)
    mpl.rcParams['axes.unicode_minus'] = False

    ax.plot(time, temp, lw=1.0)

    ax.set_xlabel('Time (fs)')
    ax.set_ylabel('Temperature (K)')

    plt.tight_layout()
    plt.savefig(figname, dpi=400)
    print("\n%s has been saved."%figname)



if __name__=='__main__':
    main()
