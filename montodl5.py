"""
Executable for processing the R1 waveforms, and storing the reduced parameters
into a HDF5 file, openable as a `pandas.DataFrame`.
"""
import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import h5py
import glob
import os
import matplotlib.pyplot as plt
import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
import pandas as pd
from tqdm import tqdm
import json
from CHECLabPy.core.io import ReaderR1, DL1Writer
from inspect import isabstract
from CHECLabPy.core.base_reducer import WaveformReducer
from CHECLabPy.core.spectrum_fitter import SpectrumFitter
from CHECLabPy.core.base_reducer import WaveformReducer
from scipy import interpolate
from scipy.ndimage import correlate1d
import operator
import re
path =r'/Users/chec/Desktop/Monitor to DL5'
fileprefix='Run43461t'
output_path = path + '/' + fileprefix + "_dl1.h5"
loadexistingpd=0
lepddir=r'/Users/chec/Desktop/Monitor to DL5/Run43461t_dl1.h5'

def main():
    if loadexistingpd==1:
        df=pd.read_hdf(lepddir)
    else:
        df = pd.DataFrame()
    for filename in glob.glob(os.path.join(path,fileprefix + '*.mon')):
        file = open(path + '/' + fileprefix + '.mon')
        for row in file:
            hasht=[x.start() for x in re.finditer(' ',row)]
            lenhash=len(hasht)
            if lenhash>0:
                #print(hasht)
                #print(len(hasht))
                date=row[:hasht[0]]
                time=row[hasht[0]:hasht[1]]
                component=row[hasht[1]:hasht[2]]
                object=row[hasht[2]:hasht[3]]
                if lenhash<5:
                    value=row[hasht[3]:]
                    TMno=100
                elif object == ' T_PRI' or object == ' T_AUX' or object == ' T_PSU' or object == ' T_SIPM' or object == ' SiPM_I' or object == ' SiPM_V':
                    value=row[hasht[4]:]
                    TMno=row[hasht[3]:hasht[4]]
                elif lenhash==5 and len([x.start() for x in re.finditer('\t', row[hasht[4]:])])==16 and len([x.start() for x in re.finditer(' ', row[hasht[4]:])])==1:
                    TMno = row[hasht[3]:hasht[4]]
                    a=row[hasht[4]:]
                    asht = [x.start() for x in re.finditer('\t', a)]
                    value=np.zeros(16,)
                    value[0]=a[:asht[0]]
                    for i in range (0,14):
                        value[i+1] = a[asht[i]:asht[i+1]]
                    value[15]=a[asht[14]:]
                elif lenhash==5 and len([x.start() for x in re.finditer('\t', row[hasht[4]:])])==12 and len([x.start() for x in re.finditer(' ', row[hasht[4]:])])==1:
                    #print(object)
                    TMno = row[hasht[3]:hasht[4]]
                    a=row[hasht[4]:]
                    #print(a)
                    asht = [x.start() for x in re.finditer('\t', a)]
                    value=np.zeros(12,)
                    value[0]=a[:asht[0]]
                    for i in range (0,10):
                        value[i+1] = a[asht[i]:asht[i+1]]
                    #print(value)
                    value[11]=a[asht[10]:]
                else:
                    value=1e6
                    TMno=1e6
                df_ev = pd.DataFrame(dict(
                    fileprefix=fileprefix,
                    date=date,
                    time=time,
                    component=component,
                    object=object,
                    value=str(value),
                    TMno=TMno
                ), index=[0])
                df = df.append(df_ev, ignore_index=True)
    #print(df)
    df.to_hdf(output_path,key='df',mode='w')

if __name__ == '__main__':
    main()
    reader=pd.read_hdf(output_path)
    #print(reader)
    object=reader['object']
    value=reader['value']
    component=reader['component']
    BP_TM_I=[]
    IMON=[]
    time=reader['time']
    tbptmi=[]
    for i in range (0, len(object)):
        row=object[i]
        tim=time[i]
        if row=='TM_CURRENT' or row==' TM_CURRENT ' or row==' TM_CURRENT' or row=='TM_CURRENT ':
            print(i)
            a=value[i]
            bptmcur=[x.start() for x in re.finditer('\t',a)]
            BPTMI=np.zeros(32,)
            BPTMI[0] = float(a[:bptmcur[0]])
            for i in range(0, 30):
                BPTMI[i + 1] = float(a[bptmcur[i]:bptmcur[i + 1]])
            BPTMI[31] = float(a[bptmcur[30]:])
            BP_TM_I.append(BPTMI)
            print(tim)
            tbptmi.append(tim)
        if row=='IMON_ETH_12V' or row==' IMON_ETH_12V' or row=='IMON_ETH_12V ' or row==' IMON_ETH_12V ':
            a=value[i]
            IMON.append(float(a[:(len(a)-4)]))
    #plt.plot(tbptmi,BP_TM_I)
    #plt.show()
    plt.plot(IMON,BP_TM_I)
    #plt.show()
    #plt.plot(tbptmi,IMON)
    plt.show()

    #print(object)
    #plt.hist(object)
    #TMno=reader['TMno']
    #print(reader['TMno'])
    #time=reader['time']
    #plt.plot(time,TMno)
    #plt.show()
