from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)
import xgboost as xgb
from scipy.stats import uniform, randint
import numpy as np

# TARGET = 'limit'
TARGET = 'energy'
SENSOR = 'audio'
# SENSOR = 'acc'

INPUT_SIZE = 3 if TARGET=='energy' else 2
OUTPUT_SIZE = 1
FZ = 0.08
N_EDGES = 4

TEST_SIZE = 0.2

# MACHINE_TOOL = 'old'
MACHINE_TOOL = 'new'
DATA_DIR = f'data/01_raw/{MACHINE_TOOL}_dmu'
PROCESSED_DIR = 'data/02_processed'

MODEL_DIR = f'models/{MACHINE_TOOL}_dmu_{TARGET}_{SENSOR}'
PLOT_DIR = f'plots/{MACHINE_TOOL}_dmu_{TARGET}_{SENSOR}'
RESULTS_DIR = f'results/{MACHINE_TOOL}_dmu_{TARGET}_{SENSOR}'


DATA_RANGES = [
    np.concatenate((np.arange(101, 132), np.arange(201, 256), np.arange(301, 337))), # WZ4
    np.concatenate((np.arange(101, 169), np.arange(201, 235))), # WZ5
    np.concatenate((np.arange(101, 156), np.arange(202, 229), np.arange(301, 339))) # WZ6
]

RANDOM_SEED = 1234

CV_FOLDS = 20
N_ITER_SEARCH = 300

LINEWIDTH = 1
FONTSIZE = 14
TARGET_LBLS = ['ae_limit']

PARAM_DICTS = [
    {
        'learning_rate': uniform(0.0001, 0.1),
        'max_depth': randint(2, 32),
        'subsample': uniform(0.5, 0.5),
        'n_estimators': randint(100, 1000),
        'colsample_bytree': uniform(0.4, 0.6),
        'lambda': randint(1, 100),
        'gamma': uniform()
    },
    {
        'learning_rate': uniform(0.0001, 0.1),
        'n_estimators': randint(100, 1000)
    },
    {
        'learning_rate': uniform(0.0001, 0.1),
        'n_estimators': randint(100, 1000),
        'max_depth': randint(2, 32),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(2, 11),
        'max_features': randint(1, INPUT_SIZE)
    },
    {
        'n_estimators': randint(100, 1000),
        'max_depth': randint(2, 32),
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(2, 11),
        'max_features': randint(1, INPUT_SIZE)
    }
]
REGRESSORS = [
    [xgb.XGBRegressor(objective='reg:squarederror') for __ in range(OUTPUT_SIZE)],
    [AdaBoostRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)],
    [GradientBoostingRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)],
    [RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1) for __ in range(OUTPUT_SIZE)]
]
