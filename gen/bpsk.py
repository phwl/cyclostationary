import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib
import scipy.signal as signal
import math
 
# 
size = 10
sampling_t = 0.01
t = np.arange(0, size, sampling_t)
 
# Randomly generated signal sequence
a = np.random.randint(0, 2, size)
m = np.zeros(len(t), dtype=np.float32)
for i in range(len(t)):
    m[i] = a[math.floor(t[i])]
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
 
ax1.set_title('Generate random n-bit binary signal', fontsize = 20)
plt.axis([0, size, -0.5, 1.5])
plt.plot(t, m, 'b')
 
 
fc = 4000
fs = 20 * fc # sampling frequency
ts = np.arange(0, (100 * size) / fs, 1 / fs)
coherent_carrier = np.cos(np.dot(2 * pi * fc, ts))
bpsk = np.cos(np.dot(2 * pi * fc, ts) + pi * (m - 1) + pi / 4)
 
# BPSK modulated signal waveform
ax2 = fig.add_subplot(3, 1, 2)
ax2.set_title('BPSK modulation signal', fontsize=20)
plt.axis([0,size,-1.5, 1.5])
plt.plot(t, bpsk, 'r')
 
# Define additive white Gaussian noise
 
def awgn(y, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(y ** 2) / len(y)
    npower = xpower / snr
    return np.random.randn(len(y)) * np.sqrt(npower) + y
 
# AWGN noise
noise_bpsk = awgn(bpsk, 5)
print('noise_bpsk', noise_bpsk.shape)
 
# BPSK modulation signal superimposed noise waveform
ax3 = fig.add_subplot(3, 1, 3)
ax3.set_title('BPSK modulated signal superimposed noise waveform', fontsize = 20)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, noise_bpsk, 'r')
 
# ,passband is [2000,6000]
[b11,a11] = signal.ellip(5, 0.5, 60, [2000 * 2 / 80000, 6000 * 2 / 80000], btype = 'bandpass', analog = False, output = 'ba')
 
# Low pass filter design, passband cutoff frequency is 2000Hz
[b12,a12] = signal.ellip(5, 0.5, 60, (2000 * 2 / 80000), btype = 'lowpass', analog = False, output = 'ba')
 
# Filter out-of-band noise by bandpass filter
bandpass_out = signal.filtfilt(b11, a11, noise_bpsk)
 
#Coherent demodulation, multiplied by coherent carrier in phase with the same frequency
coherent_demod = bandpass_out * (coherent_carrier * 2)
 
# Pass low pass filter
lowpass_out = signal.filtfilt(b12, a12, coherent_demod)
fig2 = plt.figure()
bx1 = fig2.add_subplot(3, 1, 1)
bx1.set_title('local carrier downconversion, after low pass filter', fontsize=20)
plt.axis([0, size, -1.5, 1.5])
plt.plot(t, lowpass_out, 'r')
 
#sample judgment
detection_bpsk = np.zeros(len(t), dtype=np.float32)
flag = np.zeros(size, dtype=np.float32)
 
for i in range(10):
    tempF = 0
    for j in range(100):
        tempF = tempF + lowpass_out[i * 100 + j]
    if tempF > 0:
        flag[i] = 1
    else:
        flag[i] = 0
for i in range(size):
    if flag[i] == 0:
        for j in range(100):
            detection_bpsk[i * 100 + j] = 0
    else:
        for j in range(100):
            detection_bpsk[i * 100 + j] = 1
 
bx2 = fig2.add_subplot(3, 1, 2)
bx2.set_title('signal after BPSK signal sampling decision', fontsize=20)
plt.axis([0, size, -0.5, 1.5])
plt.plot(t, detection_bpsk, 'r')
plt.show()
 
