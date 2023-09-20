"""Data processing methods"""

import os
import re
import numpy as np
import pandas as pd
from glob import glob

from plot_utils import hist

import pycwt
import pywt
from tqdm import tqdm

from config import (
    PROCESSED_DIR,
    DATA_FNAME,
    TEST_SIZE,
    RANDOM_SEED,
    INPUT_SIZE,
    RESULTS_DIR,
    DATA_DIR,
    DATA_RANGES,
    FZ,
    N_EDGES
)
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataProcessing:
    def __init__(self):
        # data = np.load(f'{PROCESSED_DIR}/{DATA_FNAME}')
        # self.process()
        data = self.load_processed() # spsp, ae, wear, energy

        self.scaler = MinMaxScaler()
        self.train, self.test = train_test_split(data, test_size=TEST_SIZE)

        self.train[:, :INPUT_SIZE] = self.scaler.fit_transform(self.train[:, :INPUT_SIZE])
        self.test[:, :INPUT_SIZE] = self.scaler.transform(self.test[:, :INPUT_SIZE])

        np.random.shuffle(self.train)

    def get_train_test(self):
        return self.train, self.test

    def get_scaler(self):
        return self.scaler

    def load_processed(self):
        folders = sorted(
            [fname for fname in os.listdir(DATA_DIR) if os.path.isdir(f'{DATA_DIR}/{fname}')],
            key=lambda x: int(re.search('\d+', x).group())
        )
        results = np.concatenate([np.load(f'{RESULTS_DIR}/{folder}.npy') for folder in folders])
        results = results[np.where(results[:, -1] < 0.00001)]
        # plt.scatter(results[:, 0], results[:, 1], c=results[:, -1])
        # plt.show()
        # quit()
        return results

    def process(self):
        np.set_printoptions(suppress=True)
        folders = sorted(
            [fname for fname in os.listdir(DATA_DIR) if os.path.isdir(f'{DATA_DIR}/{fname}')],
            key=lambda x: int(re.search('\d+', x).group())
        )

        nb_scales = 50
        freq_div = 20
        wavelet = 'mexh'

        for folder_idx, folder in enumerate(folders):
            folder_idx = 2
            folder = "WZ6"
            result = np.load(f'{RESULTS_DIR}/{folder}.npy')

            result = result[np.where(result[:, -1] < 0.00001)]
            hist(result[:, -1], nb_bins=20)

            plt.scatter(result[:, 0], result[:, 1], c=result[:, -1])
            plt.show()
            quit()

            # fig = plt.figure()
            # ax3d = fig.add_subplot(projection='3d')
            # ax3d.scatter(
                # result[:, 0],
                # result[:, 2],
                # result[:, 1],
                # c=result[:, -1],
                # s=2
                # # vmin=0,
                # # vmax=0.0001
            # )
            # plt.show()
            # quit()
            # folder_idx = 0
            # folder = 'WZ4'
            result = []
            print(f"Processing tool: {folder}")
            data_range = DATA_RANGES[folder_idx]
            doe = pd.read_excel(f'{DATA_DIR}/Versuchsplan_{folder}.xlsx')

            for sample_idx in tqdm(data_range):
                config = doe[doe["V"]==sample_idx]
                spsps = [
                    config.iloc[0]["n-GL"],
                    config.iloc[0]["n-GGL"],
                    config.iloc[1]["n-GL"],
                    config.iloc[1]["n-GGL"],
                    config.iloc[2]["n-GL"],
                    config.iloc[2]["n-GGL"]
                ]
                wears = [
                    config.iloc[0]["Standweg"],
                    config.iloc[0]["Standweg"],
                    config.iloc[1]["Standweg"],
                    config.iloc[1]["Standweg"],
                    config.iloc[2]["Standweg"],
                    config.iloc[2]["Standweg"]
                ]
                ae = doe.iloc[0]["ae_end"] - doe.iloc[0]["ae_start"]

                fname = glob(f'{DATA_DIR}/{folder}/Rampen/{folder}_V{sample_idx}-*.npz')[0]
                data = np.load(fname)['data']
                # data = data[:len(data)//4]

                dt = data[1, 0] - data[0, 0]
                fs = 1.0 / dt
                time = data[:, 0]
                defl = data[:, 1]
                audio = data[:, 2]

                # for spsp in spsps:

                    # fz_freq = spsp / 60.0 * N_EDGES

                    # print(f"spsp: {spsp}, fz freq: {fz_freq}")

                    # freqs = [
                        # fz_freq / freq_div + idx * fz_freq / freq_div for idx in range(nb_scales)
                    # ]
                    # # freqs = np.arange(1500, 8000, 16)
                    # central_freq = pywt.central_frequency(wavelet)
                    # scale = [central_freq / (dt * value) for value in freqs]

                    # cwtmatr, __ = pywt.cwt(audio, scale, wavelet, sampling_period=dt)
                    # cmap = plt.contourf(time, freqs, cwtmatr, extend='both', cmap='inferno')
                    # plt.colorbar(cmap)
                    # plt.hlines([fz_freq, fz_freq / N_EDGES], time[0], time[-1])
                    # plt.show()
                    # plt.close()

                # quit()

                nfft = 2 * 4096
                window = plt.mlab.window_hanning(np.ones(nfft))

                # freqs, time_spec, specs = signal.spectrogram(defl, fs, detrend=False, nfft=nfft, window=window)

                specs, freqs, time_spec, _ = plt.specgram(
                    defl,
                    # audio,
                    NFFT=2*4096,
                    noverlap=1*4096,
                    Fs=fs,
                    cmap='inferno'
                )

                specs_audio, freqs_audio, time_spec_audio, _ = plt.specgram(
                    audio,
                    NFFT=2*4096,
                    noverlap=1*4096,
                    Fs=fs,
                    cmap='inferno'
                )

                dt_spec = time_spec[1] - time_spec[0]

                specs_ = specs[5:np.where(freqs > 1000)[0][0],:].copy()

                clip = np.median(np.max(specs_, axis=0))
                specs_[specs_ > clip] = clip
                specs_ /= np.max(specs_)
                specs_[specs_ < 0.2] = .0
                specs_ *= 255.0

                envelope = [0]
                for j in range(1, specs_.shape[1]-1):
                    envelope.append(np.max(specs_[:2,j-1:j+1]))
                envelope = np.array(envelope)
                spindle_on = np.zeros_like(envelope)
                spindle_on[envelope > 180] = 1

                # plt.figure()
                # plt.imshow(specs_, origin="lower", cmap="gray")
                # plt.plot(spindle_on * 50)
                # plt.show()
                # quit()

                indices_up = np.where(np.diff(spindle_on) > 0)[0][::2]
                indices_down = np.where(np.diff(spindle_on) < 0)[0][1::2]
                if len(indices_down) == 5: # In case a measurement is too short
                    indices_down = list(indices_down)
                    indices_down.append(len(spindle_on))
                # print(indices_up)
                # print(indices_down)
                i_processes = indices_up
                i_processes += indices_down
                i_processes = i_processes // 2
                # print(f'Detected processes: {i_processes}')

                ##
                ##  Evaluate stability limits
                ##
                # fig2, axs2 = plt.subplots(6)
                # fig, ax = plt.subplots(2, 1, sharex=True)
                # ax[0].plot(time, defl)
                # ax[1].plot(time, audio)

                for i, (i_process, spsp) in enumerate(zip(i_processes, spsps)):
                    # print(i)
                    # print(i_process)
                    # print(spsp)
                    vf = N_EDGES * FZ * spsp # mm/min
                    # print(f'Detecting process {i+1} at spindle speed {spsp} and vf {vf}...', flush=True)

                    i_process -= 10
                    i_freq = np.argmax(specs_[:,i_process])
                    while specs_[i_freq+1,i_process] > 140:
                        i_freq += 1
                    row = specs_[i_freq,:]
                    while row[i_process] > 50:
                        i_process -= 1
                    i_process += 1

                    # ax[0].axvline(time_spec[i_process], 0, 1, c='r')
                    # ax[1].axvline(time_spec[i_process], 0, 1, c='r')

                    process_dy = 35 + 10 + 6
                    process_dx = process_dy * (ae / 35.0)
                    process_length = np.sqrt(process_dx**2 + process_dy**2) if i%2 == 0 else process_dy
                    process_duration = process_length / (vf / 60)
                    # print(f'Process duration: {process_duration} s')

                    process_start_offset = 1.0
                    if i % 2 == 0: # GL (Rampe)
                        process_start_offset += process_duration * 10 / process_dy
                    else:
                        process_start_offset += process_duration * 6 / process_dy


                    process_duration = process_duration * 35 / process_dy

                    valid_process_portion = 1 - (12 / (vf / 60)) / process_duration

                    i_process_start = i_process + int(process_start_offset / dt_spec)
                    i_process_end = i_process_start + int((process_duration * valid_process_portion) / dt_spec)

                    # ax[0].axvline(time_spec[i_process_start], 0, 1, c='k')
                    # ax[0].axvline(time_spec[i_process_end], 0, 1, c='k')
                    # ax[1].axvline(time_spec[i_process_start], 0, 1, c='k')
                    # ax[1].axvline(time_spec[i_process_end], 0, 1, c='k')

                    i_f0 = np.where(freqs > 2000)[0][0]
                    i_f1 = np.where(freqs > 6000)[0][0]
                    energy = np.sum(specs[i_f0:i_f1, i_process_start:i_process_end]**1.3, axis=0)
                    energy_audio = np.sum(specs_audio[i_f0:i_f1, i_process_start:i_process_end]**1.3, axis=0)
                    aes = np.linspace(0, ae * valid_process_portion, len(energy))

                    ae_lim = None

                    i1 = np.where(energy > 0.001)[0]
                    if len(i1) > 0:
                        i1 = i1[0]
                        i0 = i1 - 1
                        alpha = (0.001 - energy[i0]) / (energy[i1] - energy[i0])
                        ae0 = aes[i0]
                        ae1 = aes[i1]
                        ae_lim = ae0 + alpha * (ae1 - ae0)


                    # print(sample_idx)

                    # plt.figure()
                    # plt.plot(aes, energy_audio)
                    # plt.show()
                    # plt.close()

                    # result.append([spsp, i % 2, ae, wears[i], ae_lim])
                    # axs2[i].plot(aes, energy)
                    # axs2[i].plot(aes, energy_audio)
                    # axs2[i].axhline(0.001,0,1,c='k')
                    # if ae_lim:
                        # axs2[i].axvline(ae_lim,0,1,c='k')

                    if i % 2 == 0:
                        for idx, ae in enumerate(aes):
                            result.append([spsp, ae, wears[i], energy_audio[idx]])
                # plt.show()
                # plt.close()
                # quit()

            result = np.array(result)
            print(result.shape)
            np.save(f'{RESULTS_DIR}/{folder}.npy', result)
            quit()
