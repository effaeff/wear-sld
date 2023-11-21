from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.neural_network import MLPRegressor
import numpy as np

NN = False
OPT = False

# TARGET = 'limit'
TARGET = 'energy'
SENSOR = 'audio'
# SENSOR = 'acc'

INPUT_SIZE = 3 if TARGET=='energy' else 2
OUTPUT_SIZE = 1
FZ = 0.08
N_EDGES = 4

TEST_SIZE = 0.2
TRANSFER = False

# MACHINE_TOOL = 'old'
MACHINE_TOOL = 'new'
DATA_DIR = f'data/01_raw/{MACHINE_TOOL}_dmu'
PROCESSED_DIR = 'data/02_processed'

# MODEL_DIR = f'models/{MACHINE_TOOL}_dmu_{TARGET}_{SENSOR}_transfer-wz2'
# PLOT_DIR = f'plots/{MACHINE_TOOL}_dmu_{TARGET}_{SENSOR}_transfer-wz2'
# RESULTS_DIR = f'results/{MACHINE_TOOL}_dmu_{TARGET}_{SENSOR}_transfer-wz2'

MODEL_DIR = f'models/{MACHINE_TOOL}_dmu_{TARGET}_{SENSOR}_oldwear-intpl'
PLOT_DIR = f'plots/{MACHINE_TOOL}_dmu_{TARGET}_{SENSOR}_oldwear-intpl'
RESULTS_DIR = f'results/{MACHINE_TOOL}_dmu_{TARGET}_{SENSOR}_oldwear-intpl'

DATA_RANGES = [
    np.concatenate((np.arange(101, 132), np.arange(201, 256), np.arange(301, 337))), # WZ4
    np.concatenate((np.arange(101, 169), np.arange(201, 235))), # WZ5
    np.concatenate((np.arange(101, 156), np.arange(202, 229), np.arange(301, 339))) # WZ6
]

RANDOM_SEED = 1234

CV_FOLDS = 10
N_ITER_SEARCH = 300

BATCH_SIZE = 4

LINEWIDTH = 1
FONTSIZE = 14
TARGET_LBLS = ['ae_limit']

PARAM_DICTS = [
    # {
        # 'learning_rate': uniform(0.0001, 0.1),
        # 'max_depth': randint(2, 32),
        # 'subsample': uniform(0.5, 0.5),
        # 'n_estimators': randint(100, 1000),
        # 'colsample_bytree': uniform(0.4, 0.6),
        # 'lambda': randint(1, 100),
        # 'gamma': uniform()
    # },
    # {
        # 'learning_rate': uniform(0.0001, 0.1),
        # 'n_estimators': randint(100, 1000)
    # },
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
    },
    # {
        # 'learning_rate_init': uniform(0.0001, 0.01),
        # 'alpha': uniform(0.0001, 0.05),
        # 'learning_rate': ['constant','adaptive'],
        # 'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)]
        # # 'hidden_layer_sizes': [[randint(10, 200) for __ in range(randint(2, 20))]]
    # }
]
REGRESSORS = [
    # [xgb.XGBRegressor(objective='reg:squarederror') for __ in range(OUTPUT_SIZE)],
    # [AdaBoostRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)],
    [GradientBoostingRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)],
    [RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1) for __ in range(OUTPUT_SIZE)]
    # [MLPRegressor(random_state=RANDOM_SEED) for __ in range(OUTPUT_SIZE)]
]

NN_CONFIG = {
    'models_dir': MODEL_DIR,
    'nb_layers': 10,
    'nb_units': 10,
    'input_size': INPUT_SIZE,
    'output_size': OUTPUT_SIZE,
    'activation': 'ReLU',
    'optimizer': 'Adam',
    'loss': 'MSELoss',
    'max_iter': 300,
    'learning_rate': 1e-3,
    'early_stopping': False,
    'max_problem': False,
    'drop_rate': 0.2
}

