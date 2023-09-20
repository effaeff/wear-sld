import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from glob import glob
import math
from config import DATA_DIR, PROCESSED_DIR, PLOT_DIR

# spsp, GL/GGL, ae, wear, ae_lim

def plot1():
    spsps = np.arange(4000, 8050, 50)

    fig, axs = plt.subplots(2, sharex=True, sharey=True)

    for i, WZ in enumerate(['WZ1', 'WZ2', 'WZ4']):
        data = np.load(f'./stabilitymap_{WZ}.npy')
        stabilitymap_gl = data[data[:,1] == 0]
        stabilitymap_ggl = data[data[:,1] == 1]

        def get_ae_lim(stabilitymap):
            ae_lim = [None]*len(spsps)
            for j, spsp in enumerate(spsps):
                stability = stabilitymap[stabilitymap[:,0] == spsp]
                for ae in stability[:,4]:
                    if ae:
                        ae_lim[j] = ae
            return ae_lim

        axs[0].plot(spsps, get_ae_lim(stabilitymap_gl), 'o-', label=f"{WZ} GL")
        if WZ != 'WZ4':
            axs[1].plot(spsps, get_ae_lim(stabilitymap_ggl), 'o-', label=f"{WZ} GGL")

    for ax in axs:
        ax.set_xlim(4000,8000)
        ax.set_ylim(0,6)
        ax.axvline(4500, 0, 1, c='k')
        ax.axvline(6800, 0, 1, c='k')
        ax.grid()
        ax.legend()

    plt.show()


def plot2():
    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    data = np.load(f'./stabilitymap_WZ3.npy')
    stabilitymap = data[data[:,0] == 6800]
    stabilitymap_gl = stabilitymap[stabilitymap[:,1] == 0]
    stabilitymap_ggl = stabilitymap[stabilitymap[:,1] == 1]
    axs[0].plot(stabilitymap_gl[:,3], stabilitymap_gl[:,4], label=f"WZ3 n=6800 GL")
    axs[0].plot(stabilitymap_ggl[:,3], stabilitymap_ggl[:,4], label=f"WZ3 n=6800 GGL")

    stabilitymap = data[data[:,0] == 4500]
    print(stabilitymap)
    stabilitymap_gl = stabilitymap[stabilitymap[:,1] == 0]
    stabilitymap_ggl = stabilitymap[stabilitymap[:,1] == 1]
    axs[1].plot(stabilitymap_gl[:,3], stabilitymap_gl[:,4], 'o-', label=f"WZ3 n=4500 GL")
    axs[1].plot(stabilitymap_ggl[:,3], stabilitymap_ggl[:,4], 'o-', label=f"WZ3 n=4500 GGL")

    for ax in axs:
        ax.grid()
        ax.legend()

    plt.show()


def plot3():
    spsps = np.arange(4000, 8050, 50)

    fig, axs = plt.subplots(1, sharex=True, sharey=True)

    points_all = []

    # for i, WZ in enumerate(['WZ4-100', 'WZ4-200', 'WZ4-300']):
    # for i, WZ in enumerate(['WZ5-100', 'WZ5-200']):
    # for i, WZ in enumerate(['WZ6-100', 'WZ6-200', 'WZ6-300']):
    for i, WZ in enumerate(['WZ4-100', 'WZ4-200', 'WZ4-300', 'WZ5-100', 'WZ5-200', 'WZ6-100', 'WZ6-200', 'WZ6-300']):
        data = np.load(f'{DATA_DIR}/stabilitymap_{WZ}.npy')
        stabilitymap_gl = data[data[:,1] == 0]

        def get_ae_lim(stabilitymap):
            ae_lim = [None]*len(spsps)
            points = []
            for j, spsp in enumerate(spsps):
                count = 0
                stability = stabilitymap[stabilitymap[:,0] == spsp]
                for i_ae, ae in enumerate(stability[:,4]):
                    wear = stability[i_ae,3]
                    if not math.isnan(ae):
                        points.append([spsp, ae, wear])
                        if ae_lim[j] is None:
                            ae_lim[j] = ae
                        else:
                            ae_lim[j] += ae
                        count += 1
                if ae_lim[j] is not None:
                    ae_lim[j] /= count
                # else:
                    # print(spsp)
            return ae_lim, points

        ae_lim, points = get_ae_lim(stabilitymap_gl)
        points_all.extend(points)
        axs.plot(spsps, ae_lim, '-', label=f"{WZ} GL")

    points_all = np.array(points_all)
    axs.scatter(points_all[:,0], points_all[:,1], c=points_all[:,2], label=f"Wear")

    # Move target to last index
    points_all[:, [0, 1, 2]] = points_all[:, [0, 2, 1]] # 0: spsp, 1: wear, 2: ae limit
    fname = 'wz4-wz6'
    np.save(f'{PROCESSED_DIR}/{fname}.npy', points_all)


    # for WZ in ["WZ1", "WZ3", "WZ4"]:
    #     data = np.loadtxt(f'./Simulation/sld_{WZ}.csv')
    #     axs.plot(data[:,0], data[:,1], label=f'Sim. {WZ}')

    for ax in [axs]:
        ax.set_xlim(4000,8000)
        ax.set_ylim(0,6)
        ax.grid()
        ax.legend()

    # plt.show()
    plt.savefig(f'{PLOT_DIR}/{fname}.png', dpi=600)

if __name__ == '__main__':
    plot3()


