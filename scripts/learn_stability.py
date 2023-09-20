"""Script for learning forces based on chip thickness and vibration velocity"""

import misc
import numpy as np
from joblib import dump

from wearsld.data_processing import DataProcessing
from wearsld.train import train
from wearsld.test import test

from wearsld.utils import write_results, load_estimators

from config import OUTPUT_SIZE, MODEL_DIR, RESULTS_DIR, REGRESSORS, PLOT_DIR

def main():
    """Main method"""
    misc.gen_dirs([MODEL_DIR, RESULTS_DIR, PLOT_DIR])

    processing = DataProcessing()
    train_data, test_data = processing.get_train_test()

    hyperopts = train(train_data)
    # hyperopts = load_estimators(MODEL_DIR)
    total_errors = np.empty((len(hyperopts), OUTPUT_SIZE))
    total_variances = np.empty((len(hyperopts), OUTPUT_SIZE))
    for hyper_idx, hyperopt in enumerate(hyperopts):
        dump(
            hyperopt,
            f'{MODEL_DIR}/hyperopt_{hyperopt[0].best_estimator_.__class__.__name__}.joblib'
        )
        errors, variances = test(hyperopt, test_data)
        total_errors[hyper_idx] = errors
        total_variances[hyper_idx] = variances
    write_results(hyperopts, total_errors, total_variances)

if __name__ == '__main__':
    main()
