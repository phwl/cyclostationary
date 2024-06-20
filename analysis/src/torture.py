# Import functions and libraries
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# import pyaudio
import threading,time
import cyclostationary
import datetime

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import integrate

import threading,time
import multiprocessing

from numpy import mean
from numpy import power
from numpy.fft import fft
from numpy.fft import fftshift
from numpy.fft import ifft
from numpy.fft import ifftshift
import bitarray
from  scipy.io.wavfile import read as wavread


def plotfam(x, t, verbose=255):
    Np = 64
    L = 1
    B = 32 # how far around the center we look
    xs = x
    nx = len(xs) // 2 # center of X axis
    # print("nx=", nx)
    
    # do analysis
    s = cyclostationary.scd_fam(xs, Np, L)
    f = np.absolute(s)
    alpha = cyclostationary.alphaprofile(s)
    (my, mx) = f.shape 
    # print("my, mx", my, mx)
    f = f[(my//2-B):(my//2+B), (mx//2-B):(mx//2 + B)]
    
    if verbose & 1:
        # plot SCD graph
        plt.matshow(f, cmap='hot')
        plt.suptitle(t)
        plt.colorbar()
        plt.show()

    if verbose & 2:
        # plot histogram
        plt.hist(f)
        plt.suptitle("Histogram " + t)
        plt.show()
        hthresh = 1
        hbelow = sum(f.flatten() <= hthresh)
        hn = f.size
        print("Zero/Total={}/{} (thesh={}, {}% sparse)".format(hbelow, hn, hthresh, 100.0 * hbelow / hn))

    if verbose & 4:
        # plot alphaprofile graph
        plt.plot(alpha)
        plt.suptitle(t + "-alpha")
        plt.show()
    return alpha, s

def gen_bpsk(Nbits, snrdB=100, sigdB=0, verbose=0):
    fs = 44100  # sampling rate
    baud = 300  # symbol rate
    Ns = fs//baud
    N = Nbits * Ns
    f0 = 1800
    bits = np.random.randn(Nbits,1) > 0
    imp = zeros(N)
    imp[::Ns] = bits.ravel()*2-1

    h = signal.firwin(Ns*4,1.0/Ns)
    imp_sinc = signal.fftconvolve(imp,h,mode='full')
    t = r_[0.0:len(imp_sinc)]/fs
    
    noisedB = sigdB - snrdB
    nscale = pow(10.0, (noisedB / 20))
    sscale = pow(10.0, (sigdB / 20))
    BPSK = imp_sinc*sin(2*pi*f0*t)
    BPSK = BPSK / max(BPSK)
    BPSK_s = sscale * BPSK + nscale * np.random.normal(0,1,len(imp_sinc))
    BPSK_s = BPSK_s / max(BPSK_s)
    if verbose & 1:
        plt.plot(t,BPSK_s)
    return t, BPSK_s

# sweeps the noise level and returns the largest values of the alpha profile
def bpsk_sweepsnr(Nbits, startSNR=10, verbose=0):
    # returns the set containing the peaks largest indices for the alpha profile for BPSK there should be 2 peaks
    def genandplot(snrdB, peaks=2):
        t, BPSK_s = gen_bpsk(Nbits, snrdB, verbose=verbose)
        alpha, scd = plotfam(BPSK_s, f"SCF of pulse shaped BPSK SNR={snrdB}", verbose=verbose)
        return set(np.argpartition(alpha, len(alpha)-peaks)[-peaks:]), scd
    
    resultl = []
    testdB = startSNR
    mindex, origscd = genandplot(testdB)
    if verbose > 0:
        print(f'scd.shape={origscd.shape}')
    resultl.append((testdB, mindex))
    while True:
        testdB -= 1
        nindex, scd = genandplot(testdB)
        resultl.append((testdB, nindex))
        if nindex != mindex:
            break
    genandplot(testdB)
    return testdB, resultl

def runtest(start, stop, startSNR=10, verbose=0):
    scores = []
    for i in range(start, stop):
        bits = 1 << i
        start = datetime.datetime.now()
        try:
            testdB, resultl = bpsk_sweepsnr(bits, startSNR=testdB+3, verbose=verbose)
        except:
            testdB, resultl = bpsk_sweepsnr(bits, startSNR=startSNR, verbose=verbose)        
        print(f'bits=2^{i}: Score_SNR={testdB} ({datetime.datetime.now() - start})', flush=True)
        scores.append([testdB, resultl])
    return scores

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print(f'scores = runtest(1,16,5)')
        scores = runtest(1,16,startSNR=5)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("start", type=int, help="start nbits e.g. 1")
    parser.add_argument("stop", type=int, help="stop nbits e.g. 10")
    parser.add_argument("startSNR", type=int, help="startSNR e.g. 0")
    args = parser.parse_args()
                
    print(f'scores = runtest({args.start},{args.stop},startSNR={args.startSNR})')
    scores = runtest(args.start, args.stop, 
                    startSNR=args.startSNR, verbose=args.verbose)
