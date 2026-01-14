import re
import os
from glob import glob
import configparser
import numpy as np
import time

conf = configparser.ConfigParser()
conf.read('config.ini',encoding='utf-8')

SUB_NUM_HAMIL = int(conf['sub']['SUB_NUM_HAMIL'])
TIME_MAX_HAMIL = float(conf['sub']['TIME_MAX_HAMIL'])

inDir = conf['epc']['inDir']+'/'
dhamilDir = conf['epc']['dhamilDir']+'/'
subfile_s = conf['epc']['subfile_s']

hamildir = glob(inDir+dhamilDir+'*+*')
hamildir.sort()


def calc_nline(name):
   with open(name, 'rb') as f:
    count = 0
    last_data = '\n'
    while True:
        data = f.read(0x400000)
        if not data:
            break
        count += data.count(b'\n')
        last_data = data
    if last_data[-1:] != b'\n':
        count += 1

    return count


def sub_one():
    while True:
        outfile = glob('*.out')
        if len(outfile) > 0:
            nline = 0
            while True:
                nline_t = calc_nline(outfile[0])
                if nline < nline_t:
                    nline = nline_t
                else:
                    break
            os.system('rm -rf *_rst')
            break
        else:
            time.sleep(2)


def sub_para_hamil(subfile,hamildir,SUB_NUM,TIME_MAX):
    nhamil = len(hamildir)
    if nhamil < SUB_NUM:
        SUB_NUM = nhamil
    time_tag = np.zeros((nhamil),dtype=np.float64)
    none_tag = np.zeros((nhamil),dtype=np.int32)
    undone_tag = np.zeros((nhamil),dtype=np.int32)
    done_tag = np.zeros((nhamil),dtype=np.int32)
    while True:
        none_num = 0
        undone_num = 0
        done_num = 0
        for i in range(nhamil):
            if not os.path.exists(hamildir[i]+'/start_tag'):
                none_tag[none_num] = i
                none_num += 1
            else:
                outfile = glob(hamildir[i]+'/*.out')
                if len(outfile) > 0:
                    nline = 0
                    while True:
                        nline_t = calc_nline(outfile[0])
                        if nline < nline_t:
                            nline = nline_t
                        else:
                            break
                    scfoutfile = glob(hamildir[i]+'/*.scfout')
                    if len(scfoutfile) == 0:
                        os.chdir(hamildir[i])
                        os.system('rm -rf start_tag *.log *.err *_rst')
                        none_tag[none_num] = i
                        none_num += 1
                        os.chdir('../../../')
                    else:
                        done_tag[done_num] = i
                        done_num += 1

                    os.system('rm -rf %s/*_rst'%(hamildir[i]))
                else:
                    time_t = time.time()
                    if time_t - time_tag[i] > TIME_MAX:
                        os.chdir(hamildir[i])
                        logfile = glob('*.log')
                        if len(logfile) > 0:
                            kill = re.findall(r'\d+',logfile[0])[0]
                            os.system('scancel %s'%(kill))
                        os.system('rm -rf start_tag *.log *.err *_rst')
                        none_tag[none_num] = i
                        none_num += 1
                        os.chdir('../../../')
                    else:
                        undone_tag[undone_num] = i
                        undone_num += 1
                    
        if done_num == nhamil:
            break
    
        sub_extra = SUB_NUM - undone_num
        if sub_extra >= none_num:
            sub_extra = none_num
        for i in range(sub_extra):
            os.chdir(hamildir[none_tag[i]])
            os.system('sbatch %s'%(subfile))
            os.system('touch start_tag')
            os.chdir('../../../')
            time_tag[none_tag[i]] = time.time()


starttime = time.time()
sub_para_hamil(subfile_s,hamildir,SUB_NUM_HAMIL,TIME_MAX_HAMIL)
os.chdir(inDir)

os.system('sbatch %s'%(subfile_s))
sub_one()
os.chdir('../')

subfile_u = conf['ucell']['subfile_u']
os.chdir(inDir+'ucell')
os.system('sbatch %s'%(subfile_u))
sub_one()
os.chdir('../../')
endtime = time.time()
print('Submitting time: %fs'%(endtime-starttime))
