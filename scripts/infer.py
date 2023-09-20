"""Inference of trained regressors"""

from plot_utils import modify_axis

import numpy as np
from matplotlib import pyplot as plt

from wearsld.data_processing import DataProcessing
from wearsld.utils import load_estimators

from config import MODEL_DIR, PLOT_DIR, FONTSIZE

def infer():
    """Inference method"""
    processing = DataProcessing()
    scaler = processing.get_scaler()

    hyperopts = load_estimators(MODEL_DIR)
    for hyper_idx, hyperopt in enumerate(hyperopts):
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        fig.suptitle(hyperopt[0].best_estimator_.__class__.__name__)

        spsp = np.arange(4000, 8001, 1)

        wear_data = np.array([0 for __ in spsp])
        test_data = np.transpose([spsp, wear_data])
        test_data = scaler.transform(test_data)
        pred_0 = hyperopt[0].predict(test_data)

        test_wears = [2500 * idx for idx in range(0, 11)]

        for test_wear in test_wears:
            wear_data = np.array([test_wear for __ in spsp])
            test_data = np.transpose([spsp, wear_data])

            test_data = scaler.transform(test_data)

            pred = hyperopt[0].predict(test_data) #- pred_0

            axs.plot(spsp, pred, label=f'{test_wear}')

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
        axs.set_yticks(np.arange(1, 6, 1))

        fig.canvas.draw()

        axs = modify_axis(axs, 'rpm', 'mm', -2, -2, FONTSIZE)

        axs.set_xlim(4000, 8000)
        # axs.set_ylim(0, 3.5)
        axs.set_ylim(1, 5)
        plt.tight_layout()

        # plt.show()
        plt.savefig(f'{PLOT_DIR}/{hyperopt[0].best_estimator_.__class__.__name__}.png', dpi=600)
        plt.close()

if __name__ == '__main__':
    infer()

