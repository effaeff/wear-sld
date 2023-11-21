"""Script for transfer-learning dynamics based on spindle speeds, depth of cuts and tool wear"""

import sys
import os
import warnings
import misc
from tqdm import tqdm
from plot_utils import modify_axis
import numpy as np
import math
from joblib import dump
from scipy.stats import uniform, randint

from wearsld.data_processing import DataProcessing
from wearsld.train import train
from wearsld.test import test

from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.base import clone

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from wearsld.utils import write_results, load_estimators

from config import (
    INPUT_SIZE,
    OUTPUT_SIZE,
    MODEL_DIR,
    RESULTS_DIR,
    REGRESSORS,
    PLOT_DIR,
    TARGET,
    TEST_SIZE,
    RANDOM_SEED,
    FONTSIZE,
    CV_FOLDS,
    N_ITER_SEARCH
)

def main():
    """Main method"""
    misc.gen_dirs([MODEL_DIR, RESULTS_DIR, PLOT_DIR])

    processing = DataProcessing()

    data_old = processing.get_data(TARGET, 'old')
    scaler_old = processing.get_scaler()
    train_old, test_old = processing.get_train_test()

    data_new = processing.get_data(TARGET, 'new')
    train_new, test_new = train_test_split(data_new, test_size=TEST_SIZE)
    np.random.shuffle(train_new)
    scaler_new = MinMaxScaler()
    train_new[:, :INPUT_SIZE] = scaler_new.fit_transform(train_new[:, :INPUT_SIZE])
    test_new[:, :INPUT_SIZE] = scaler_new.fit_transform(test_new[:, :INPUT_SIZE])

    hyperopts = load_estimators(f'models/new_dmu_limit_audio')
    for hyper_idx, hyperopt in enumerate(hyperopts):

        # Train estimator using all data of new dmu with hyperparameters of best estimator
        # print(hyperopt[0].best_estimator_)
        estimator = getattr(ensemble, hyperopt[0].best_estimator_.__class__.__name__)()
        params = hyperopt[0].best_estimator_.get_params()
        # Enable re-using model coefficients for transfer-learning
        params['warm_start'] = True
        estimator.set_params(**params)

        estimator.fit(train_new[:, :INPUT_SIZE], train_new[:, INPUT_SIZE])

        # Iteratively post-train estimator using more and more experiments of old dmu
        for i_exp in tqdm(range(10, len(train_old))):
            data_old_avail = train_old[:i_exp, :]

            trans_est = clone(estimator)
            grid = RandomizedSearchCV(
                estimator=trans_est,
                param_distributions={'n_estimators': randint(700, 1000)},
                cv=5,
                random_state=RANDOM_SEED,
                n_iter=50
            )
            if not sys.warnoptions:
                warnings.simplefilter("ignore")
                os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
                grid.fit(data_old_avail[:, :INPUT_SIZE], data_old_avail[:, INPUT_SIZE])
                # trans_est.fit(data_old_avail[:, :INPUT_SIZE], data_old_avail[:, INPUT_SIZE])
            dump(
                # trans_est,
                grid,
                f'{MODEL_DIR}/{estimator.__class__.__name__}_trans{i_exp}.joblib'
            )
            pred = grid.predict(test_old[:, :INPUT_SIZE])
            # pred = trans_est.predict(test_old[:, :INPUT_SIZE])
            target = test_old[:, INPUT_SIZE]
            err = math.sqrt(mean_squared_error(target, pred)) / np.ptp(target) * 100.0
            var = np.std(
                [
                    abs(target[idx] - pred[idx]) / np.ptp(target)
                    for idx in range(len(target))
                ]
            ) * 100.0

            fig, axs = plt.subplots(1, 1, figsize=(6, 6))
            fig.suptitle(f'{hyperopt[0].best_estimator_.__class__.__name__}: {err:.2f} +- {var:.2f}')

            spsp = np.arange(4000, 8001, 1)

            test_wears = [2500 * idx for idx in range(0, 11)]

            for test_wear in test_wears:
                wear_data = np.array([test_wear for __ in spsp])
                test_data = np.transpose([spsp, wear_data])

                test_data = scaler_old.transform(test_data)

                pred = grid.predict(test_data) #- pred_0
                # pred = trans_est.predict(test_data) #- pred_0

                axs.plot(spsp, pred, label=f'{test_wear}')

            sca = axs.scatter(data_old[:, 0], data_old[:, -1], c=data_old[:, 1], s=1.5)
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
            axs.set_yticks(np.arange(0, 6, 1))

            fig.canvas.draw()

            axs = modify_axis(axs, 'rpm', 'mm', -2, -2, FONTSIZE)

            axs.set_xlim(4000, 8000)
            axs.set_ylim(0, 5)
            plt.tight_layout()

            # plt.show()
            plt.savefig(
                f'{PLOT_DIR}/{hyperopt[0].best_estimator_.__class__.__name__}_trans{i_exp:03d}.png',
                dpi=600
            )
            plt.close()

if __name__ == '__main__':
    main()
