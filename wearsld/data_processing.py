"""Data processing methods"""

import os
import re
import numpy as np
import pandas as pd
from glob import glob

from plot_utils import hist
import h5py

import pycwt
import pywt
from tqdm import tqdm

from time import time as time_lib

from config import (
    PROCESSED_DIR,
    PLOT_DIR,
    TEST_SIZE,
    RANDOM_SEED,
    INPUT_SIZE,
    RESULTS_DIR,
    DATA_DIR,
    DATA_RANGES,
    FZ,
    N_EDGES,
    SENSOR,
    TARGET,
    MACHINE_TOOL,
    TRANSFER,
    BATCH_SIZE,
    FONTSIZE
)
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from misc import gen_dirs

from plot_utils import modify_axis, hist

class DataProcessing:
    def __init__(self):
        self.n_wz1 = 0
        self.n_wz2 = 0
        # data = np.load(f'{PROCESSED_DIR}/{DATA_FNAME}')
        self.data = self.load_energy(TARGET, MACHINE_TOOL)

        # print(self.data.shape)
        # self.data = self.load_stability()
        # self.process_hdf()
        # quit()
        # self.data = self.load_processed() # spsp, ae, wear, energy

        self.scaler = MinMaxScaler()
        # self.scaler = StandardScaler()
        if TRANSFER:
            # WZ3
            # self.train = self.data[self.n_wz1+self.n_wz2:]
            # self.test = self.data[:self.n_wz1+self.n_wz2]
            # WZ1
            # self.train = self.data[:self.n_wz1]
            # self.test = self.data[self.n_wz1:]
            # WZ2
            self.train = self.data[self.n_wz1:self.n_wz1+self.n_wz2]
            self.test = np.concatenate((self.data[:self.n_wz1], self.data[self.n_wz1+self.n_wz2:]))
        else:
            self.train, self.test = train_test_split(self.data, test_size=TEST_SIZE)
            np.random.shuffle(self.train)

        self.train[:, :INPUT_SIZE] = self.scaler.fit_transform(self.train[:, :INPUT_SIZE])
        self.test[:, :INPUT_SIZE] = self.scaler.transform(self.test[:, :INPUT_SIZE])

    def get_train_test(self):
        return self.train, self.test

    def get_data(self, target, mt):
        return self.load_energy(target, mt)
        # return self.load_stability()

    def get_batches(self):
        train = np.random.shuffle(self.train)
        cutoff = (len(self.train) // BATCH_SIZE) * BATCH_SIZE
        return np.reshape(self.train[:cutoff, :], (-1, BATCH_SIZE, self.train.shape[-1]))

    def get_scaler(self):
        return self.scaler

    def validate(self, evaluate, epoch_idx, save_eval, verbose, save_suffix):
        """Validation method"""
        epoch_dir = f'{RESULTS_DIR}/train_epoch{epoch_idx:04}'
        gen_dirs([epoch_dir])
        total_errors = np.empty(len(self.test))

        for idx, __ in enumerate(self.test):
            inp = self.test[idx, :INPUT_SIZE]
            out = self.test[idx, INPUT_SIZE]

            pred_out, error = evaluate(inp, out)
            total_errors[idx] = error


        if verbose:
            fig, axs = plt.subplots(1, 1, figsize=(6, 6))
            fig.suptitle(f'MLP, Error: {np.mean(total_errors):.2f} +- {np.std(total_errors):.2f}')

            spsp = np.arange(4000, 8001, 1)
            test_wears = [2500 * idx for idx in range(0, 11)]

            for test_wear in test_wears:
                wear_data = np.array([test_wear for __ in spsp])
                test_data = np.transpose([spsp, wear_data])

                test_data = self.scaler.transform(test_data)
                pred_out = evaluate(test_data, None)
                # pred_out = np.array(
                    # [
                        # evaluate(
                            # self.scaler.transform(
                                # np.reshape([spsp_value, test_wear], (1, INPUT_SIZE))
                            # ).flatten(),
                            # None
                        # )
                        # for spsp_value in spsp
                    # ]
                # )

                axs.plot(spsp, pred_out, label=f'{test_wear}')

            sca = axs.scatter(self.data[:, 0], self.data[:, -1], c=self.data[:, 1], s=1.5)
            plt.colorbar(sca)

            axs.legend(
                bbox_to_anchor=(0., 1.02, 1., .102),
                loc='lower left',
                ncol=3,
                mode="expand",
                fontsize=FONTSIZE,
                borderaxespad=0.,
                frameon=False
            )

            axs.set_ylabel(r'a$_e$ limit', fontsize=FONTSIZE)
            axs.set_xlabel('Spindle speed', fontsize=FONTSIZE)
            axs.set_xticks(np.arange(4000, 8001, 1000))
            # axs.set_yticks(np.arange(0, 3.6, 0.5))
            axs.set_yticks(np.arange(0, 7, 1))

            fig.canvas.draw()

            axs = modify_axis(axs, 'rpm', 'mm', -2, -2, FONTSIZE)

            axs.set_xlim(4000, 8000)
            # axs.set_ylim(0, 3.5)
            axs.set_ylim(0, 6)
            plt.tight_layout()

            if save_eval:
                plt.savefig(f'{PLOT_DIR}/mlp_epoch{epoch_idx}.png', dpi=600)
            else:
                plt.show()
            plt.close()
            # print(np.mean(total_errors))

        return np.mean(total_errors), np.std(total_errors)


    def load_processed(self):
        # folders = sorted(
            # [fname for fname in os.listdir(DATA_DIR) if os.path.isdir(f'{DATA_DIR}/{fname}')],
            # key=lambda x: int(re.search('\d+', x).group())
        # )
        # results = np.concatenate([np.load(f'{RESULTS_DIR}/{folder}.npy') for folder in folders])
        # results = results[np.where(results[:, -1] < 0.00001)]
        results = np.load(f'{PROCESSED_DIR}/energy_wave_{MACHINE_TOOL}dmu.npy') # spsp, ae, wear, energy
        scaler = MinMaxScaler()
        results = results[np.where(results[:, 1] <= 4)]
        results[:, -1] = scaler.fit_transform(results[:, -1].reshape((-1, 1))).flatten()

        print(results.shape)

        results = results[np.where(results[:, 3] <= 0.3)]
        scaler = MinMaxScaler()
        results[:, -1] = scaler.fit_transform(results[:, -1].reshape((-1, 1))).flatten()
        print(results.shape)
        hist(results[:, 2])
        plt.scatter(results[:, 0], results[:, 1], c=results[:, -1])
        # fig = plt.figure()
        # ax3d = fig.add_subplot(projection='3d')
        # ax3d.scatter(
            # results[:, 0],
            # results[:, 2],
            # results[:, 1],
            # c=results[:, -1],
            # s=2
            # # vmin=0,
            # # vmax=0.0001
        # )
        plt.show()
        quit()
        return results

    def calc_limit(self, series, aes):
        lim = None

        i1 = np.where(series > np.std(series))[0]
        if len(i1) > 0:
            i1 = i1[0]
            i0 = i1 - 1
            alpha = (np.std(series) - series[i0]) / (series[i1] - series[i0])
            ae0 = aes[i0]
            ae1 = aes[i1]
            lim = ae0 + alpha * (ae1 - ae0)
        return lim

    def load_stability(self):
        with h5py.File(f'{DATA_DIR}/neuedmu_stability.hdf5', 'r') as fhandle:
            results = np.concatenate([
                fhandle[f'{tool}/stability_{SENSOR}'][()] for tool in fhandle
            ])
            return results

    def load_energy(self, target, mt):
        fname = 'altedmu_energy_2' if mt=='old' else 'neuedmu_energy_2'
        with h5py.File(f'data/01_raw/{mt}_dmu/{fname}.hdf5', 'r') as fhandle:
            results = []
            for i_tool, tool in enumerate(fhandle):
                exps = fhandle[tool]
                for exp in exps:
                    data = fhandle[f'{tool}/{exp}/energy_{SENSOR}'][()]
                    # wear = fhandle[f'{tool}/{exp}/wear'][:, 1]
                    start_wear = fhandle[f'{tool}/{exp}'].attrs['wear']
                    energy = data[:, 1]
                    aes = data[:, 0]
                    spsp = fhandle[f'{tool}/{exp}'].attrs['spsp']
                    wear_end = fhandle[f'{tool}/{exp}'].attrs['wear_end']
                    ae_max = fhandle[f'{tool}/{exp}'].attrs['ae_max']

                    wear = np.linspace(start_wear, wear_end, len(aes))

                    lim = fhandle[f'{tool}/{exp}/energy_{SENSOR}'].attrs['ae_lim']
                    # if lim==-1: lim = ae_max

                    if target=='energy':
                        for idx, ae in enumerate(aes):
                            results.append([spsp, ae, wear[idx], energy[idx]])
                            if i_tool==0:
                                self.n_wz1 += 1
                            elif i_tool==1:
                                self.n_wz2 += 1

                    elif target=='limit' and lim!=-1:
                        results.append([spsp, start_wear, lim])
                        if i_tool==0:
                            self.n_wz1 += 1
                        elif i_tool==1:
                            self.n_wz2 += 1

            results = np.array(results)
            # if target=='limit':
                # results = results[np.where(results[:, -1]!=-1)]
            # np.save(f'{RESULTS_DIR}/energy_acc.npy', results)

            # hist(results[:, -1], nb_bins=200)


            # plt.scatter(results[:, 0], results[:, 1], c=results[:, -1])
            # plt.show()

            # fig = plt.figure()
            # ax3d = fig.add_subplot(projection='3d')
            # ax3d.scatter(
                # results[:, 0],
                # results[:, 2],
                # results[:, 1],
                # c=results[:, -1],
                # s=2
                # # vmin=0,
                # # vmax=0.0001
            # )
            # plt.show()
            return results

    def process_hdf(self):
        results = []
        # with h5py.File(f'{DATA_DIR}/altedmu.hdf5', 'r') as fhandle:
        # results = np.load(f'{PROCESSED_DIR}/energy_wave.npy')
        audiofile = 'altedmu_audio_filtered.hdf5' if MACHINE_TOOL=='old' else 'neuedmu_audio_filtered.hdf5'
        with h5py.File(f'{DATA_DIR}/{audiofile}', 'r') as fhandle:
            res_i = 0
            for tool in fhandle:
                # tool = 'WZ9'
                exps = fhandle[tool]
                for exp in tqdm(exps):
                    energyfile = 'altedmu_energy_2.hdf5' if MACHINE_TOOL=='old' else 'neuedmu_energy_2.hdf5'
                    energy = h5py.File(f'{DATA_DIR}/{energyfile}', 'r')

                    data = fhandle[f'{tool}/{exp}/audio_filtered'][()]
                    fs = fhandle[f'{tool}/{exp}/audio_filtered'].attrs['sampling_rate']
                    # data = fhandle[f'{tool}/{exp}/audio'][()]
                    # fs = fhandle[f'{tool}/{exp}/audio'].attrs['sampling_rate']
                    spsp = fhandle[f'{tool}/{exp}'].attrs['spsp']
                    wear = energy[f'{tool}/{exp}/wear'][()]
                    ae_wear = wear[:, 0]
                    wear = wear[:, 1]
                    # wear_start = energy_file[f'{tool}/{exp}'].attrs['wear']
                    # wear_end = energy_file[f'{tool}/{exp}'].attrs['wear_end']
                    # istart = fhandle[f'{tool}/{exp}'].attrs['i_rampe_start']
                    # iend = fhandle[f'{tool}/{exp}'].attrs['i_rampe_end']
                    vf = fhandle[f'{tool}/{exp}'].attrs['vf']
                    ae = fhandle[f'{tool}/{exp}'].attrs['ae_valid_max']
                    # ae = fhandle[f'{tool}/{exp}'].attrs['ae_max']

                    nperseg = int(fs * 0.01)

                    ##############################
                    ### Correct ae calculation ###
                    ##############################
                    # res = results[res_i]
                    # if res[2] != wear_start:
                        # print(f'res wear: {res[2]}, data wear: {wear_start}')
                        # quit()

                    # cutoff = (len(data) // (nperseg+nperseg//2)) * (nperseg+nperseg//2)
                    # data = data[:cutoff]
                    # window = np.hamming(nperseg)
                    # data_downsampled = []
                    # for w_start in range(0, len(data)-nperseg, nperseg//2):
                        # data_downsampled.append(np.mean(
                            # data[w_start:w_start + nperseg] * window
                        # ))

                    # aes_wave = np.linspace(0, ae, len(data_downsampled))
                    # # print(len(data_downsampled))

                    # for ae_i, ae in enumerate(aes_wave):
                        # results[res_i + ae_i, 1] = ae

                    # res_i += len(data_downsampled)
                    # continue
                    ##############################

                    dt = 1 / fs

                    data = (data - np.mean(data)) / np.std(data)
                    # sampling frequency is insane for new dmu
                    if MACHINE_TOOL=='new':
                        data = signal.decimate(data, 3)

                    # valid_portion = lambda ae: (
                        # (6 * ae + 1225) - np.sqrt(-ae * (1189*ae - 14700))
                    # ) / (ae**2 + 1225)
                    # i_valid_end = istart + int(valid_portion(ae) * (iend - istart))

                    # data = data[istart:i_valid_end]

                    time = np.array([1 / fs * idx for idx, __ in enumerate(data)])

                    # Use wavelets for high frequency resolution
                    wavelet = 'morl' # Known to be beneficial for audio data
                    freqs = np.arange(1000, 6000, 1)
                    # fz_freq = spsp / 60.0 * N_EDGES
                    scale = pywt.frequency2scale(wavelet, freqs / fs)

                    cwtmatr, __ = pywt.cwt(data, scale, wavelet, sampling_period=dt)

                    # power = np.power(np.abs(cwtmatr), 2)
                    power = np.abs(cwtmatr) # Only use abs for now to avoid high fluctuations

                    # Scale-dependent normalization of power
                    scale_avg = np.repeat(scale, len(data)).reshape(power.shape)
                    power = power / scale_avg

                    # Downsampling in time direction using hamming window with 50% overlap
                    window = np.hamming(nperseg)
                    power_downsampled = []
                    cutoff = (power.shape[1] // (nperseg+nperseg//2)) * (nperseg+nperseg//2)
                    power = power[:, :cutoff]
                    for w_start in range(0, power.shape[1]-nperseg, nperseg//2):
                        power_downsampled.append(np.mean(
                            np.einsum(
                                'ij, j -> ij',
                                power[:, w_start:w_start + nperseg],
                                window
                            ),
                            axis=1
                        ))
                    power_downsampled = np.array(power_downsampled).T
                    time_wave = np.linspace(0, time[-1], power_downsampled.shape[1])
                    # aes = np.linspace(0, ae * valid_portion(ae), len(time))
                    aes = np.linspace(0, ae, len(time))

                    # Only sum over frequencies in the three dominant peaks in the FRF
                    freq_cond = np.where(
                        ((freqs > 1950) & (freqs < 2150)) |
                        ((freqs > 2400) & (freqs < 3299)) |
                        ((freqs > 3550) & (freqs < 4200))
                    )

                    energy = np.sum(power_downsampled, axis=0)
                    energy_filtered = np.sum(power_downsampled[freq_cond], axis=0)

                    energy = signal.savgol_filter(energy, len(energy)//10, 3)
                    energy_filtered = signal.savgol_filter(energy_filtered, len(energy_filtered)//10, 3)

                    aes_wave = np.linspace(0, ae, len(energy_filtered))
                    # wear_wave = np.linspace(wear_start, wear_end, len(energy))
                    f_interp = interp1d(ae_wear, wear, fill_value='extrapolate')
                    wear_wave = f_interp(aes_wave)
                    # aes_wave = np.linspace(0, ae * valid_portion(ae), len(energy))
                    for e_idx, __ in enumerate(energy_filtered):
                        results.append([spsp, aes_wave[e_idx], wear_wave[e_idx], energy_filtered[e_idx]])

                ########################################################
                ### The following is debugging and spectrogram stuff ###
                ########################################################

                    # wave_fig, wave_axs = plt.subplots(3, 1, sharex=True)
                    # wave_axs[0].plot(aes, data)
                    # wave_axs[1].plot(aes_wave, energy)
                    # wave_axs[1].plot(aes_wave, energy_filtered)
                    # cmap = wave_axs[2].pcolormesh(
                        # aes_wave,
                        # freqs,
                        # power_downsampled
                    # )#, extend='both', cmap='inferno')
                    # plt.show()
                    # quit()

                    # for t_idx, time_slice in enumerate(time):
                        # __, slice_ax = plt.subplots(1, 1)
                        # slice_ax.plot(freqs, power[:, t_idx])
                        # for fz_harmonic in [fz_freq * idx for idx in range(1, int(freqs[-1]//fz_freq))]:
                            # slice_ax.axvline(fz_harmonic, 0, 1, c='r')
                        # plt.show()
                        # plt.close()

                    # print(power_downsampled.shape)
                    # print(power.shape)
                    # print(data.shape)
                    # print(freqs.shape)
                    # print(len(energy))

                    # nperseg = 1024
                    # freqs_spec, time_spec, sxx = signal.spectrogram(
                        # data,
                        # fs,
                        # detrend=False,
                        # window=window,
                        # scaling='spectrum',
                        # mode='magnitude',
                        # nperseg=nperseg,
                        # noverlap=nperseg//2
                    # )
                    # freq_slice = np.where((freqs_spec >= freqs[0]) & (freqs_spec <= freqs[-1]))

                    # print(sxx.shape)
                    # print(time_spec.shape)
                    # print(freqs_spec.shape)
                    # for s_idx, slice in enumerate(time_spec):
                        # f_slice = freqs_spec[freq_slice]
                        # amp_slice = sxx[freq_slice, s_idx][0]
                        # print(f_slice.shape)

                        # peaks = signal.find_peaks(amp_slice)[0]

                        # plt.plot(f_slice, amp_slice)
                        # plt.scatter(f_slice[peaks], amp_slice[peaks], c='g')

                        # for fz_harmonic in [fz_freq * idx for idx in range(1, int(freqs[-1]//fz_freq))]:
                            # plt.axvline(fz_harmonic, 0, 1, c='r')

                        # plt.show()
                        # plt.close()

                    # fontsize = 10
                    # fig, axs = plt.subplots(4, 1, sharex=True)
                    # fig.suptitle(f'{tool} {exp}')

                    # axs[0].plot(np.linspace(0, ae, len(data)), data)
                    # # cmap = axs[1].pcolormesh(time, freqs, power)#, extend='both', cmap='inferno')
                    # # cmap = axs[1].imshow(cwtmatr, extent=[0, len(cwtmatr[0]), scale[0], scale[-1]])

                    # energy = np.sum(sxx, axis=0)
                    # aes = np.linspace(0, ae, len(energy))

                    # axs[1].plot(aes, energy)
                    # axs[1].axhline(np.std(energy), 0, 1, c='b')
                    # energy_lim = self.calc_limit(energy, aes)
                    # axs[1].axvline(energy_lim, 0, 1, c='r')

                    # slope = [z - x for x, z in zip(energy[:-1], energy[1:])]
                    # slope_lim = self.calc_limit(slope, aes)
                    # axs[2].plot(aes[1:], slope)
                    # axs[2].axhline(np.std(slope), 0, 1, c='b')
                    # axs[2].axvline(slope_lim, 0, 1, c='r')

                    # ae_lim = slope_lim if slope_lim > energy_lim else energy_lim

                    # cmap = axs[3].pcolormesh(aes, freqs_spec[freq_slice], sxx[freq_slice, :][0])

                    # axs[0].set_ylabel("Audio", fontsize=fontsize)
                    # axs[1].set_ylabel("Energy", fontsize=fontsize)
                    # axs[2].set_ylabel("Energy slope", fontsize=fontsize)
                    # axs[3].set_ylabel("Frequency", fontsize=fontsize)
                    # # axs[2].set_xlabel("Width of cut", fontsize=fontsize)

                    # left, bottom, width, height = axs[2].get_position().bounds
                    # cax = fig.add_axes([left, 0.03, width, height * 0.1])
                    # plt.colorbar(cmap, orientation='horizontal', cax=cax)

                    # fig.canvas.draw()
                    # axs[0] = modify_axis(axs[0], '', '', -2, 2, fontsize)
                    # axs[1] = modify_axis(axs[1], '', '', -2, 2, fontsize)
                    # axs[2] = modify_axis(axs[1], '', '', -2, 2, fontsize, grid=False)
                    # plt.setp(axs[0].get_xticklabels(), visible=False)
                    # plt.setp(axs[1].get_xticklabels(), visible=False)

                    # fig.align_ylabels()

                    # plt.show()
                    # plt.close()
                    # results.append([spsp, wear, ae_lim])

                ########################################################

        # __, res_ax = plt.subplots(1, 1)
        # results = np.array(results)
        # res_ax.scatter(results[:, 0], results[:, 2], c=results[:, 1])
        # plt.show()
        # plt.close()
        np.save(f'{PROCESSED_DIR}/energy_wave_{MACHINE_TOOL}dmu.npy', results)

    def process_raw(self):
        np.set_printoptions(suppress=True)
        folders = sorted(
            [fname for fname in os.listdir(DATA_DIR) if os.path.isdir(f'{DATA_DIR}/{fname}')],
            key=lambda x: int(re.search('\d+', x).group())
        )

        nb_scales = 50
        freq_div = 20
        wavelet = 'mexh'

        for folder_idx, folder in enumerate(folders):
            folder_idx = 1
            folder = "WZ4"
            # result = np.load(f'{RESULTS_DIR}/{folder}.npy')

            # result = result[np.where(result[:, -1] < 0.00001)]
            # hist(result[:, -1], nb_bins=20)

            # plt.scatter(result[:, 0], result[:, 1], c=result[:, -1])
            # plt.show()
            # quit()

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

                nfft = 4096
                window = plt.mlab.window_hanning(np.ones(nfft))

                # freqs, time_spec, specs = signal.spectrogram(defl, fs, detrend=False, nfft=nfft, window=window)

                specs, freqs, time_spec, _ = plt.specgram(
                    defl,
                    # audio,
                    NFFT=2*nfft,
                    noverlap=1*nfft,
                    Fs=fs,
                    cmap='inferno'
                )

                specs_audio, freqs_audio, time_spec_audio, _ = plt.specgram(
                    audio,
                    NFFT=2*nfft,
                    noverlap=1*nfft,
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
                fig2, axs2 = plt.subplots(6)
                fig, ax = plt.subplots(2, 1, sharex=True)
                ax[0].plot(time, defl)
                ax[1].plot(time, audio)

                for i, (i_process, spsp) in enumerate(zip(i_processes, spsps)):
                    print(i)
                    print(i_process)
                    print(spsp)
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

                    ax[0].axvline(time_spec[i_process], 0, 1, c='r')
                    ax[1].axvline(time_spec[i_process], 0, 1, c='r')

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
                    # i_process_end = i_process_start + int((process_duration * valid_process_portion) / dt_spec)
                    i_process_end = i_process_start + int((process_duration) / dt_spec)

                    ax[0].axvline(time_spec[i_process_start], 0, 1, c='k')
                    ax[0].axvline(time_spec[i_process_end], 0, 1, c='k')
                    ax[1].axvline(time_spec[i_process_start], 0, 1, c='k')
                    ax[1].axvline(time_spec[i_process_end], 0, 1, c='k')

                    i_f0 = np.where(freqs > 2000)[0][0]
                    i_f1 = np.where(freqs > 6000)[0][0]
                    energy = np.sum(specs[i_f0:i_f1, i_process_start:i_process_end]**1.3, axis=0)
                    energy_audio = np.sum(specs_audio[i_f0:i_f1, i_process_start:i_process_end]**1.3, axis=0)
                    print(f"len: {len(energy_audio)}")
                    aes = np.linspace(0, ae, len(energy))
                    # aes = np.linspace(0, ae * valid_process_portion, len(energy))

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
                    axs2[i].plot(aes, energy)
                    axs2[i].plot(aes, energy_audio)
                    axs2[i].axhline(0.001,0,1,c='k')
                    if ae_lim:
                        axs2[i].axvline(ae_lim,0,1,c='k')

                    if i % 2 == 0:
                        for idx, ae in enumerate(aes):
                            result.append([spsp, ae, wears[i], energy_audio[idx]])


                    fz_freq = spsp / 60.0 * N_EDGES

                    # freqs_cwt = [
                        # fz_freq / freq_div + idx * fz_freq / freq_div for idx in range(nb_scales)
                    # ]
                    freqs_cwt = [fz_freq - nb_scales//4 + idx for idx in range(nb_scales//2)]
                    freqs_cwt += [fz_freq*N_EDGES - nb_scales//4 + idx for idx in range(nb_scales//2)]

                    print(fz_freq)
                    print(freqs_cwt[nb_scales//4])


                    central_freq = pywt.central_frequency(wavelet)
                    scale = [central_freq / (dt * value) for value in freqs_cwt]

                    cwtmatr, __ = pywt.cwt(audio, scale, wavelet, sampling_period=dt)

                    power_fz = cwtmatr[nb_scales//4]

                    time_power_greater_zero = time[np.where(power_fz > 1)]
                    print(time_power_greater_zero[0])

                    __, ax3 = plt.subplots(1, 1)
                    cmap = ax3.contourf(time, freqs_cwt, cwtmatr, extend='both', cmap='inferno')
                    plt.colorbar(cmap)
                    ax3.axhline(fz_freq, 0, 1)
                    ax3.axhline(fz_freq*N_EDGES, 0, 1)

                    ax3.axvline(time_power_greater_zero[0], 0, 1)
                    ax[0].axvline(time_power_greater_zero[0], 0, 1)
                    ax[1].axvline(time_power_greater_zero[0], 0, 1)


                    plt.show()


                plt.show()
                plt.close()
                quit()

            result = np.array(result)
            print(result.shape)
            np.save(f'{RESULTS_DIR}/{folder}.npy', result)
            quit()
