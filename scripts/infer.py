"""Inference of trained regressors"""

from plot_utils import modify_axis

import numpy as np
from matplotlib import pyplot as plt
import itertools

from wearsld.data_processing import DataProcessing
from wearsld.utils import load_estimators

from config import MODEL_DIR, PLOT_DIR, FONTSIZE, TARGET, MACHINE_TOOL
import misc
from plot_utils import hist

def infer_stability():
    """Inference method"""
    processing = DataProcessing()
    scaler = processing.get_scaler()

    data = processing.get_data(TARGET, 'new')

    hyperopts = load_estimators(MODEL_DIR)
    for hyper_idx, hyperopt in enumerate(hyperopts):
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        fig.suptitle(hyperopt[0].best_estimator_.__class__.__name__)

        spsp = np.arange(4000, 8001, 1)

        wear_data = np.array([0 for __ in spsp])
        test_data = np.transpose([spsp, wear_data])
        test_data = scaler.transform(test_data)
        pred_0 = hyperopt[0].predict(test_data)

        # test_wears = [17500 * idx for idx in range(0, 11)]
        test_wears = [2500 * idx for idx in range(0, 11)]

        for test_wear in test_wears:
            wear_data = np.array([test_wear for __ in spsp])
            test_data = np.transpose([spsp, wear_data])

            test_data = scaler.transform(test_data)

            pred = hyperopt[0].predict(test_data) #- pred_0

            axs.plot(spsp, pred, label=f'{test_wear}')

        sca = axs.scatter(data[:, 0], data[:, -1], c=data[:, 1], s=1.5)
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

        # plt.show()
        plt.savefig(f'{PLOT_DIR}/{hyperopt[0].best_estimator_.__class__.__name__}.png', dpi=600)
        plt.close()

def infer_energy():
    """Inference method"""
    processing = DataProcessing()
    scaler = processing.get_scaler()
    np.set_printoptions(suppress=True)
    data = processing.get_data('limit', 'new')

    hyperopts = load_estimators(MODEL_DIR)
    for hyper_idx, hyperopt in enumerate(hyperopts):
        # test_wears = [17500 * idx for idx in range(0, 11)]
        spsp = np.arange(4000, 8001, 1)
        ae = np.arange(0, 6.1, 0.1)
        # for test_wear in test_wears:
        # for test_wear in range(0, 175000, 1000):
        for test_wear in range(0, 25000, 100):
            fig, axs = plt.subplots(1, 1, figsize=(6, 6))
            fig.suptitle(f'{hyperopt[0].best_estimator_.__class__.__name__} Wear: {test_wear}')

            wear_data = np.array([test_wear for __ in spsp])
            test_data = np.array(list(itertools.product(spsp, ae)))
            test_data_wear = np.c_[test_data, [test_wear for __ in range(len(test_data))]]
            # print(np.shape(test_data))
            # print(test_data[:100])

            test_data_wear = scaler.transform(test_data_wear)

            pred = hyperopt[0].predict(test_data_wear)


            pred = pred.reshape((len(spsp), len(ae)))
            # ae_limits = np.zeros_like(spsp)
            # for i_row, row in enumerate(pred):

                # limit = 10

                # energy = np.zeros_like(row)
                # for i in range(1, len(energy)):
                    # slope = row[i]
                    # energy[i] = 0
                    # for j in range(i+1):
                        # energy[i] += (row[j] - j * slope)**2
                    # energy[i] /= i

                # try:
                    # # i_lim = np.where(energy > limit)[0][0]
                    # i_lim = np.where(row > limit)[0][0]
                    # ae_lim = ae[i_lim]
                # except:
                    # ae_lim = ae[-1]

                # ae_limits[i_row] = ae_lim

                # __, energy_ax = plt.subplots(1, 1)
                # energy_ax.plot(ae, row)
                # energy_ax.axhline(np.std(row), 0, 1, c='r')
                # plt.show()
                # plt.close()

            axs.imshow(
                pred.T,
                cmap='inferno',
                extent=[spsp[0], spsp[-1], ae[0], ae[-1]],
                # interpolation='antialiased',
                origin='lower',
                aspect='auto'
            )
            sca = axs.scatter(data[:, 0], data[:, -1], c=data[:, 1], s=1.5)
            # axs.plot(spsp, ae_limits)

            axs.set_ylabel(r'a$_e$', fontsize=FONTSIZE)
            axs.set_xlabel('Spindle speed', fontsize=FONTSIZE)
            axs.set_xticks(np.arange(4000, 8001, 1000))
            # axs.set_yticks(np.arange(0, 3.6, 0.5))
            axs.set_yticks(np.arange(1, 7, 1))

            fig.canvas.draw()

            axs = modify_axis(axs, 'rpm', 'mm', -2, -2, FONTSIZE, grid=False)

            axs.set_xlim(4000, 8000)
            # axs.set_ylim(0, 3.5)
            axs.set_ylim(0, 6)
            plt.tight_layout()

            # plt.show()
            plt.savefig(
                f'{PLOT_DIR}/{hyperopt[0].best_estimator_.__class__.__name__}_{test_wear:06d}.png',
                dpi=600
            )
            plt.close()

if __name__ == '__main__':
    misc.gen_dirs([PLOT_DIR, MODEL_DIR])
    infer_stability() if TARGET=='limit' else infer_energy()
