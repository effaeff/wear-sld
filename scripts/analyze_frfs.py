import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_peaks(filepath, min_peak_dist=0.4, offset_before=0.05, offset_after=0.5):
    data = pd.read_csv(filepath, skiprows=16, delimiter=';', decimal=',', names=['Time', 'Hammer', 'ACC'])
    data = np.column_stack((data['Time'], data['Hammer'], data['ACC']))

    dt = data[1,0] - data[0,0]
    min_peak_dist_indices = int(min_peak_dist / dt)
    offset_before_indices = int(offset_before / dt)
    offset_after_indices = int(offset_after / dt)

    peaks = []
    for peak in np.where(data[:,1] > 50)[0]:
        if len(peaks) == 0:
            peaks.append(peak)
        elif (peak - peaks[-1]) >= min_peak_dist_indices:
            peaks.append(peak)

    # plt.plot(data[:,1])
    # plt.vlines(peaks, ymin=np.min(data[:,1]), ymax=np.max(data[:,1]), colors='k')
    # plt.show()

    peaks = [data[peak - offset_before_indices : peak + offset_after_indices, :] for peak in peaks]
    for peak in peaks:
        peak[:,2] -= np.mean(peak[:200,2])
    return peaks

def analyze(peaks):
    frfs = []
    for peak in peaks:
        N = peak.shape[0]
        dt = peak[1,0] - peak[0,0]
        fft_freq  = np.fft.fftfreq(N, dt)
        fft_force = np.fft.fft(peak[:,1])
        fft_defl  = np.fft.fft(peak[:,2])
        fft_defl /= 1j * (fft_freq * 2 * np.pi)
        fft_defl /= 1j * (fft_freq * 2 * np.pi)
        fft_defl *= 1e6 # m -> µm
        frf = fft_defl / fft_force

        highpass = 400.0
        highpass_index = np.where(fft_freq >= highpass)[0][0]

        frfs.append(np.column_stack((fft_freq, frf))[highpass_index:N//2])
    return frfs

filenames = [
   './WZ1/FRF/WZ1_XX.csv', 
   # './WZ1/FRF/WZ1_YY.csv', 
   './WZ3/FRF/WZ3_XX.csv', 
   './WZ4/FRF/WZ4_XX_V255.csv'
    ]

plt.figure()

for filename in filenames:
    XX = analyze(get_peaks(filename))
    np.savetxt(filename[:-4] + '.evo.txt', np.column_stack((XX[0][:,0], np.abs(XX[0][:,1]), np.angle(XX[0][:,1]) / np.pi * 180.0)).astype(np.float64), fmt='%.8f')
    plt.plot(XX[0][:, 0], np.abs(XX[0][:, 1]), label=f'$H_{{{filename[10:16]}}}$')


plt.xlim(0, 10000)
plt.legend()
plt.grid()
plt.show()
