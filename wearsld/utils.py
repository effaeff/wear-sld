"""Various helper functions"""

import numpy as np
from joblib import load
from config import REGRESSORS, OUTPUT_SIZE, RESULTS_DIR

def write_results(hyperopts, errors, variances):
    """Write results to file"""
    with open(f'{RESULTS_DIR}/results.txt', 'w', encoding='utf-8') as res_file:
        res_file.write(
            f"{'Regressor':<40} NRMSE ae-limit\n"
        )

        for hyper_idx, hyperopt in enumerate(hyperopts):
            res_file.write(
                f"{hyperopt[0].best_estimator_.__class__.__name__:<40} "
                f"{f'{errors[hyper_idx, 0]:.2f} +/- {variances[hyper_idx, 0]:.2f}'}\n"
            )

def load_estimators(directory):
    """Load already trained hyperopt objects"""
    hyperopts = np.empty((len(REGRESSORS), OUTPUT_SIZE), dtype=object)
    for idx, __ in enumerate(hyperopts):
        hyperopts[idx] = load(
            f'{directory}/hyperopt_{REGRESSORS[idx][0].__class__.__name__}.joblib'
        )
    return hyperopts
