import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle

from glob import glob
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line



def process(V, plan, fileprefix):
    print(f'Processing V{V}...', flush=True)

    ##
    ##  Read data
    ##
    plan_ = plan[plan["V"] == V]
    n_teeth = 4
    fz = 0.08

    spsps = [plan_.iloc[0]["n-GL"], plan_.iloc[0]["n-GGL"], plan_.iloc[1]["n-GL"], plan_.iloc[1]["n-GGL"], plan_.iloc[2]["n-GL"], plan_.iloc[2]["n-GGL"]]
    ae = plan_.iloc[0]["ae_end"] - plan_.iloc[0]["ae_start"]
    wears = [plan_.iloc[0]["Standweg"], plan_.iloc[0]["Standweg"], plan_.iloc[1]["Standweg"], plan_.iloc[1]["Standweg"], plan_.iloc[2]["Standweg"], plan_.iloc[2]["Standweg"]]

    print(f'Spindle speeds: {spsps}')
    print(f'Rampe ae: {ae} mm')

    datafile = glob(f'{fileprefix}V{V}-*.npz')[0]
    print(f'Datafile: {datafile}')

    data = np.load(datafile)['data']
    dt = data[1,0] - data[0,0]
    fs = 1.0/dt
    duration = data.shape[0] * dt
    print(f'Sampling frequency: {fs} Hz')
    print(f'Duration: {duration} s')

    ##
    ##  Calculate spectrogram
    ##
    fig, ax = plt.subplots()
    specs, freqs, times, _ = plt.specgram(data[:,1], NFFT=2*4096, noverlap=1*4096, Fs=fs)
    plt.ylim(0, 10000)

    dt_spec = times[1] - times[0]


    ##
    ##  Detect six processes
    ##
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
    # plt.imshow(specs_, origin="lower", cmap=cm.gray)
    # plt.plot(spindle_on * 50)
    # plt.show()

    indices_up = np.where(np.diff(spindle_on) > 0)[0][::2]
    indices_down = np.where(np.diff(spindle_on) < 0)[0][1::2]
    if len(indices_down) == 5: # In case a measurement is too short
        indices_down = list(indices_down)
        indices_down.append(len(spindle_on))
    print(indices_up)
    print(indices_down)
    i_processes = indices_up
    i_processes += indices_down
    i_processes = i_processes // 2
    print(f'Detected processes: {i_processes}')

    result = []

    ##
    ##  Evaluate stability limits
    ##
    # fig2, axs2 = plt.subplots(6, sharey=True)
    for i, (i_process, spsp) in enumerate(zip(i_processes, spsps)):
        vf = n_teeth * fz * spsp # mm/min
        print(f'Detecting process {i+1} at spindle speed {spsp} and vf {vf}...', flush=True)

        i_process -= 10
        i_freq = np.argmax(specs_[:,i_process])
        while specs_[i_freq+1,i_process] > 140:
            i_freq += 1
        row = specs_[i_freq,:]
        while row[i_process] > 50:
            i_process -= 1
        i_process += 1

        ax.axvline(times[i_process], 0, 1, c='r')

        process_dy = 35 + 10 + 6
        process_dx = process_dy * (ae / 35.0)
        process_length = np.sqrt(process_dx**2 + process_dy**2) if i%2 == 0 else process_dy
        process_duration = process_length / (vf / 60)
        print(f'Process duration: {process_duration} s')

        process_start_offset = 1.0
        if i % 2 == 0: # GL (Rampe)
            process_start_offset += process_duration * 10 / process_dy
        else:
            process_start_offset += process_duration * 6 / process_dy

        process_duration = process_duration * 35 / process_dy


        i_process_start = i_process + int(process_start_offset / dt_spec)
        i_process_end = i_process_start + int(process_duration / dt_spec)

        ax.axvline(times[i_process_start], 0, 1, c='k')
        ax.axvline(times[i_process_end], 0, 1, c='k')

        i_f0 = np.where(freqs > 2000)[0][0]
        i_f1 = np.where(freqs > 6000)[0][0]
        energy = np.sum(specs[i_f0:i_f1, i_process_start:i_process_end]**1.3, axis=0)
        aes = np.linspace(0, ae, len(energy))

        ae_lim = None

        i1 = np.where(energy > 0.001)[0]
        if len(i1) > 0:
            i1 = i1[0]
            i0 = i1 - 1
            alpha = (0.001 - energy[i0]) / (energy[i1] - energy[i0])
            ae0 = aes[i0]
            ae1 = aes[i1]
            ae_lim = ae0 + alpha * (ae1 - ae0)

        result.append([spsp, i % 2, ae, wears[i], ae_lim])
        # axs2[i].plot(aes, energy)
        # axs2[i].axhline(0.001,0,1,c='k')
        # if ae_lim:
        #     axs2[i].axvline(ae_lim,0,1,c='k')




    # plt.show()
    # plt.close()
    return result


if __name__ == '__main__':
    # plan = pd.read_excel('./Versuchsplan_WZ4.xlsx')
    # process(119, plan, './WZ4/Rampen/WZ4_')
    # exit(0)

    # data = []
    # plan = pd.read_excel('./Versuchsplan_WZ1.xlsx')
    # for i in range(101, 128):
    #     data.extend(process(i, plan, './WZ1/Rampen/WZ1_'))
    # for i in range(201, 211):
    #     data.extend(process(i, plan, './WZ1/Rampen/WZ1_'))
    # np.save('stabilitymap_WZ1.npy', np.array(data, dtype=np.float32))


    # data = []
    # plan = pd.read_excel('./Versuchsplan_WZ2.xlsx')
    # for i in range(101, 113):
    #     data.extend(process(i, plan, './WZ2/Rampen/WZ2_'))
    # np.save('stabilitymap_WZ2.npy', np.array(data, dtype=np.float32))

    # data = []
    # plan = pd.read_excel('./Versuchsplan_WZ3.xlsx')
    # for i in range(101, 137):
    #     data.extend(process(i, plan, './WZ3/Rampen/WZ3_'))
    # for i in range(138, 167):
    #     data.extend(process(i, plan, './WZ3/Rampen/WZ3_'))
    # np.save('stabilitymap_WZ3.npy', np.array(data, dtype=np.float32))

    # plan = pd.read_excel('./Versuchsplan_WZ4.xlsx')
    # data = []
    # for i in range(101, 132):
    #     data.extend(process(i, plan, './WZ4/Rampen/WZ4_'))
    # np.save('stabilitymap_WZ4-100.npy', np.array(data, dtype=np.float32))
    # data = []
    # for i in range(201, 256):
    #     data.extend(process(i, plan, './WZ4/Rampen/WZ4_'))
    # np.save('stabilitymap_WZ4-200.npy', np.array(data, dtype=np.float32))
    # data = []
    # for i in range(301, 337):
    #     data.extend(process(i, plan, './WZ4/Rampen/WZ4_'))
    # np.save('stabilitymap_WZ4-300.npy', np.array(data, dtype=np.float32))


    # plan = pd.read_excel('./Versuchsplan_WZ5.xlsx')
    # data = []
    # for i in range(101, 169):
    #     data.extend(process(i, plan, './WZ5/Rampen/WZ5_'))
    # np.save('stabilitymap_WZ5-100.npy', np.array(data, dtype=np.float32))
    # data = []
    # for i in range(201, 235):
    #     data.extend(process(i, plan, './WZ5/Rampen/WZ5_'))
    # np.save('stabilitymap_WZ5-200.npy', np.array(data, dtype=np.float32))


    plan = pd.read_excel('./Versuchsplan_WZ6.xlsx')
    # data = []
    # for i in range(101, 156):
    #     data.extend(process(i, plan, './WZ6/Rampen/WZ6_'))
    # np.save('stabilitymap_WZ6-100.npy', np.array(data, dtype=np.float32))
    data = []
    for i in range(202, 229):
        data.extend(process(i, plan, './WZ6/Rampen/WZ6_'))
    np.save('stabilitymap_WZ6-200.npy', np.array(data, dtype=np.float32))
    data = []
    for i in range(301, 339):
        data.extend(process(i, plan, './WZ6/Rampen/WZ6_'))
    np.save('stabilitymap_WZ6-300.npy', np.array(data, dtype=np.float32))
