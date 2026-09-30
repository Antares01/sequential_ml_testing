import argparse
import os
import warnings
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AntisymmetricBet import AntisymmetricBet
from ML_e_process import ML_e_process
import pickle
from utils import g_family_cb, generate_dataset_safety_experiment, return_model, prepare_lambda_parameters,  update_g_func_static
from Sampler import GaussianSampler

class TrueModel:
    def __init__(self, W):
        self.W = np.asarray(W)

    def predict(self, Z):
        Z = Z[:, 1:]  # Exclude the first column (X) from Z
        return (Z @ self.W) ** 2


def parse_args():
    parser = argparse.ArgumentParser(description="Safety and robustness")
    parser.add_argument("--seeds", type=int, nargs="+", help="List of seeds")
    parser.add_argument("--n", type = int, default = 8000, help = "for how many timesteps to run the experiment")
    parser.add_argument("--pretraining_percentage", type = int, default = 10, help = "percentage (as an integer) of the data to use for pretraining")
    parser.add_argument("--batches", type=list, default = [5, 10, 20], help = "list of batch sizes to test")
    parser.add_argument("--resamplings", type=int, default = 10, help = "number of resamplings for the martingale")
    parser.add_argument("--model", type=str)
    return parser.parse_args()


def main(args):
    regressor_name= args.model
    n = args.n
    pretraining_fraction = args.pretraining_percentage / 100.0
    n_init = int(n * pretraining_fraction)
    batches = args.batches
    b_resamplings = args.resamplings
    js_list = [0]
    test_set_size = 300
    for s in args.seeds:
        XZ,Y, U, W, stdev = generate_dataset_safety_experiment(n=(n+test_set_size), beta=0, d=19, seed=s)
        XZ_tests = XZ[-test_set_size:]
        Y_tests = Y[-test_set_size:]
        XZ = XZ[:n]
        Y = Y[:n]
        true_model = TrueModel(W)
        sampler = GaussianSampler(j=0)
        model = return_model(regressor_name=regressor_name, seed=s)
        params_gs = prepare_lambda_parameters()
        strategy_cb = AntisymmetricBet(
            g_family= g_family_cb, 
            update_g_func = update_g_func_static, 
            parameters=params_gs,
        )

        betting_strategies = {"coin betting": strategy_cb}

        e_process = ML_e_process(batch_list=batches, 
                        n_init=n_init, 
                        b_resamplings=b_resamplings, 
                        study_j=js_list,
                        betting_strategies=betting_strategies, 
                        model=model,
                        samplers=[sampler], 
                        learn_conditional_distribution=True,
                        optional_stopping=True, 
                        true_sampler=(U, stdev),
                        true_model=true_model,
                        test_data=(XZ_tests, Y_tests)   
                        )

        martingales = e_process.martingales(XZ, Y)
        lambdas = e_process.get_best_parameters()
        distances = e_process.wasserstein_distances
        errors = e_process.estimation_errors

        results_dir = Path("../results/csv/safety_simulations")
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            results_dir
            / f"simulations_safety_model{regressor_name}_{s}_seed_n_{n}_martingales.pkl"
        )

        with open(filename, "wb") as f:
            pickle.dump((martingales, lambdas, distances, errors), f)


if __name__ == "__main__":
    main(parse_args())
            

