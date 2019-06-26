from scipy.fftpack import rfft
from scipy.signal import firwin
from scipy.signal import freqz
from scipy.signal import lfilter
from pylab import *
import pylab as plt
import numpy as np
import struct
import wave

# create a file we will use to store the data to perform the fft on
file = "ied_signal.wav"

# Create a fake signal to process using the fft
# This fake signal will be replaced with a variable input from a microcontroller pin input
# create a linear space, 0 to 1 seconds with n_samp samples
# The sampling rate of the analog to digital convert
sampling_rate = 100000.0 # 100kHz
amplitude = 16000
num_samples = 100000
time_length = 1
time = arange(0, time_length, 1.0/num_samples)
# time = np.linspace(0,1,1.0/1000)
# frequency is the number of times a wave repeats a second
f1 = 15000 # 15khz
f2 = 25000 # 20kHz
f3 = 30000 # 30kHz
f4 = 35000 # 35kHz
f5 = 40000 # 40kHz

# Create a noisy signal
#signal = np.cos(f1*2*pi*time)
#signal += np.cos(f2*2*pi*time)
#signal += np.sin(f3*2*pi*time)
#signal += np.cos(f4*2*pi*time)
#signal += np.sin(f5*2*pi*time)
#noise_amp = 2.0
#signal += noise_amp * randn(len(time))

signal = [np.sin(2 * np.pi * f3 * x/sampling_rate) + np.sin(2 * np.pi * f2 * x/sampling_rate) for x in range(num_samples)]

# number of frames = number of samples
nframes = num_samples

# the following signify that the data isn't compressed to python wave functions
comptype = "NONE"
compname = "not compressed"
# number of channels
nchannels = 1
# sampling width in bytes, wave files are usually 16 bits or 2 bytes per sample
sampwidth = 2

# open the file and set the parameters
wav_file = wave.open(file, 'w')
wav_file.setparams((nchannels, sampwidth, int(sampling_rate), nframes, comptype, compname))
# s is the single sample being written to the file, multiplying by amplitude to convert to fixed point
# struct takes data and packs it as binary data, 'h' means it is a 16 bit number
# this will take our sine wave samples and write it to our file, ied_signal.wav, packed as 16 bit audio.
for s in signal:
   wav_file.writeframes(struct.pack('h', int(s*amplitude)))

# frame rate same as number of samples and sampling rate of analog to digital converter (100kHz)
frame_rate = 100000
infile = "ied_signal.wav"
num_samples = 100000
wav_file = wave.open(infile, 'r')
# wave readframes() function reads all the signal frames from a wave file
data = wav_file.readframes(num_samples)
wav_file.close()
# telling the unpacker to unpack num_samples 16 bit words
data = struct.unpack('{n}h'.format(n=num_samples), data)
# convert the data to a numpy array.
data = np.array(data)
# take the fft of the data
data_fft = np.fft.fft(data)
# tke absolute value of fft data or else the data is useless (complex)
frequencies = np.abs(data_fft)

# return the frequency array element with the highest value
print("The frequency is {} Hz".format(np.argmax(frequencies)))

plt.subplot(2, 1, 1)
plt.plot(data[:300])
plt.title("Original audio wave")
plt.subplot(2, 1, 2)
plt.plot(frequencies)
plt.title("Frequencies found")
plt.xlim(0, 50000)
plt.show()


## OLD
# W = fftfreq(signal.size, d=time[1]-time[0])

# FFT fnct def
# def fft_function(signal):
#    fft_signal = rfft(signal)/len(signal)
#    return fft_signal

# Create a Bandpass filter
numtaps = 100
low_cutoff = 28000 # 28kHz
high_cutoff = 32000 # 32kHz
nyq_rate = sampling_rate/2.0
bpf = firwin(numtaps, [low_cutoff/nyq_rate, high_cutoff/nyq_rate], pass_zero=False)
w, h = freqz(bpf)
# Plot bandpass filter response
plot(w/(2*pi), 20*log10(abs(h)))
plt.title('FIR Bandpass Filter Response')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency Coefficient')
plt.show()

# Filter the signal using the bandpass filter
signal_filtered = lfilter(bpf, 1, signal)

# Perform an fft of the original input signal
fft_signal = fft_function(signal)
# Perform an fft of the pass-band filtered input signal
fft_signal_filtered = fft_function(signal_filtered)

plt.subplot(221)
plt.plot(time,signal)
#plt.xlim(0,1)
plt.title('Noisy Input Signal')
plt.ylabel('Magnitude (W)')
plt.xlabel('Time (s)')
plt.subplot(222)
plt.plot(20*log10(abs(fft_signal)))
plt.title('Noisy Input Signal FFT')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
#plt.xlim(0,500)
plt.subplot(223)
plt.plot(time,signal_filtered)
plt.title('Filtered Input Signal')
plt.ylabel('Magnitude (W)')
plt.xlabel('Time (s)')
#plt.xlim(0,1)
plt.subplot(224)
plt.plot(20*log10(abs(fft_signal_filtered)))
plt.title('Filtered Input Signal FFT')
plt.ylabel('Magnitude (dB)')
plt.xlabel('Frequency (Hz)')
#plt.xlim(0,500)
plt.show()