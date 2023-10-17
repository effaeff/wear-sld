"""Learning routine"""

import numpy as np
import sys
import os
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV

import warnings

from config import (
    REGRESSORS,
    PARAM_DICTS,
    INPUT_SIZE,
    OUTPUT_SIZE,
    CV_FOLDS,
    N_ITER_SEARCH
)

def train(train_data):
    """Learning method"""
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
        hyperopts = np.empty((len(REGRESSORS), OUTPUT_SIZE), dtype=object)
        for reg_idx in tqdm(range(len(REGRESSORS))):
            for out_idx in range(OUTPUT_SIZE):
                inp = train_data[:, :INPUT_SIZE]
                target = train_data[:, INPUT_SIZE + out_idx]
                rand_search = RandomizedSearchCV(
                    REGRESSORS[reg_idx][out_idx],
                    param_distributions=PARAM_DICTS[reg_idx],
                    n_iter=N_ITER_SEARCH,
                    cv=CV_FOLDS,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
                rand_search.fit(
                    inp,
                    target
                )
                hyperopts[reg_idx, out_idx] = rand_search

    return hyperopts
