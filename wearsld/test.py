"""Testing routine using trained regressor"""

import re
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from plot_utils import modify_axis
import matplotlib.ticker as mticker
from matplotlib import rc
# rc('font', family='Arial')

from config import (
    INPUT_SIZE,
    OUTPUT_SIZE,
    FONTSIZE,
    LINEWIDTH,
    PLOT_DIR,
    RESULTS_DIR,
    TARGET_LBLS
)

def test(hyperopt, test_data):
    errors = np.zeros(OUTPUT_SIZE)
    variances = np.zeros(OUTPUT_SIZE)

    for out_idx in range(OUTPUT_SIZE):
        pred = hyperopt[out_idx].predict(test_data[:, :INPUT_SIZE])
        target = test_data[:, INPUT_SIZE + out_idx]

        errors[out_idx] = math.sqrt(
            mean_squared_error(target, pred)
        ) / np.ptp(target) * 100.0
        variances[out_idx] = np.std(
            [
                abs(target[idx] - pred[idx]) / np.ptp(target)
                for idx in range(len(target))
            ]
        ) * 100.0

    return errors, variances
