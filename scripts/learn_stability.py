"""Script for learning dynamics based on spindle speeds, depth of cuts and tool wear"""

import misc
import numpy as np
from joblib import dump

from wearsld.data_processing import DataProcessing
from wearsld.train import train
from wearsld.nn_trainer import Trainer
from wearsld.test import test

from wearsld.utils import write_results, load_estimators
from pytorchutils.mlp import MLPModel

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from config import OUTPUT_SIZE, MODEL_DIR, RESULTS_DIR, REGRESSORS, PLOT_DIR, NN, NN_CONFIG, OPT

def objective(config):
    # print(f"Trying config: {config}")

    processing = DataProcessing()

    NN_CONFIG['nb_layers'] = int(config['nb_layers'])
    NN_CONFIG['nb_units'] = int(config['nb_units'])
    NN_CONFIG['lr'] = int(config['lr'])
    model = MLPModel(NN_CONFIG)
    trainer = Trainer(NN_CONFIG, model, processing)
    trainer.get_batches_fn = processing.get_batches
    if OPT:
        error = trainer.train(validate_every=1, save_every=0, save_eval=False, verbose=False)
        return {'loss': error, 'params': config, 'status': STATUS_OK}
    else:
        trainer.train(validate_every=10, save_every=10, save_eval=True, verbose=True)
    # print(f"Error: {error:.2f}")


def main():
    """Main method"""
    misc.gen_dirs([MODEL_DIR, RESULTS_DIR, PLOT_DIR])


    if NN:
        print("Start hyperparameter optimization")
        config = {
            'nb_layers': 24,
            'nb_units': 52,
            'lr': 0.0009582994911837965
        }
        if not OPT:
            objective(config)
            quit()
        space = {
            'nb_layers': hp.randint('nb_layers', 3, 50),
            'nb_units': hp.randint('nb_units', 5, 100),
            'lr': hp.uniform('lr', 0.00001, 0.001)
        }
        trials = Trials()
        best = fmin(objective, space=space, algo=tpe.suggest, max_evals=75, trials=trials)
        np.save(f"{RESULTS_DIR}/hyperopt_best.npy", best)
        print(f"Finished hyperparameter tuning. Best config:\n{best}")

    else:
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
