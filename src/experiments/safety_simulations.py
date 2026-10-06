import argparse
import os
import warnings
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AntisymmetricBet import AntisymmetricBet
from ML_e_process import ML_e_process
import pickle
from utils import g_family_cb, g_family_sign, generate_dataset_safety_experiment, return_model, prepare_lambda_parameters,  update_g_func_static
from Sampler import GaussianSampler, Sampler

class TrueModel:
    def __init__(self, W):
        self.W = np.asarray(W)

    def predict(self, Z):
        Z = Z[:, 1:]  # Exclude the first column (X) from Z
        return (Z @ self.W) ** 2

    def fit(self, Z, Y):
        return self

class TrueSampler(Sampler):
    def __init__(self, j, U, stdev):
        super().__init__(j)
        self.U = U
        self.stdev = stdev

    def fit(self, X):
        # No fitting needed for the true sampler
        return self

    def sample(self, X):
        feature_j = self.j
        mask = np.arange(X.shape[1]) != feature_j
        X_minus = X[:, mask]
        mu_true = (X_minus) @ self.U

        return np.random.normal(mu_true, self.stdev)

def get_strategy(name):
    if name == "coin_betting":
        params_gs = prepare_lambda_parameters()
        return AntisymmetricBet(
            g_family=g_family_cb,
            update_g_func=update_g_func_static,
            parameters=params_gs,
        )
    elif name == "sign":
        params_gs = prepare_lambda_parameters()
        return AntisymmetricBet(
            g_family= g_family_sign,
            update_g_func = update_g_func_static,
            parameters=params_gs,
        )
    else:
        raise ValueError(f"Unknown strategy: {name}")

def parse_args():
    parser = argparse.ArgumentParser(description="Safety and robustness")
    parser.add_argument("--seeds", type=int, nargs="+", help="List of seeds")
    parser.add_argument("--n", type = int, default = 3999, help = "for how many timesteps to run the experiment")
    parser.add_argument("--pretraining_percentage", type = int, default = 10, help = "percentage (as an integer) of the data to use for pretraining")
    parser.add_argument("--batches", type=list, default = [5, 10, 20], help = "list of batch sizes to test")
    parser.add_argument("--resamplings", type=int, default = 10, help = "number of resamplings for the martingale")
    parser.add_argument("--model", type=str, help="nn for neural network, tm for true model")
    parser.add_argument("--strategy", type=str, default = "coin_betting", help = "betting strategy to use")
    parser.add_argument("--n_jobs", type=int, default = 1, help = "number of seeds to run in parallel (one process per seed)")
    return parser.parse_args()


def run_seed(s, args):
    regressor_name= args.model
    n = args.n
    pretraining_fraction = args.pretraining_percentage / 100.0
    n_init = int(n * pretraining_fraction)
    batches = args.batches
    b_resamplings = args.resamplings
    js_list = [0]
    test_set_size = 300
    wasserstein_resamplings = 1000
    # The samplers draw from the global numpy RNG: seed it so that runs are reproducible
    # and parallel workers do not share the same random stream
    np.random.seed(s)
    XZ,Y, U, W, stdev = generate_dataset_safety_experiment(n=(n+test_set_size), beta=0, d=19, seed=s)
    XZ_tests = XZ[-test_set_size:]
    Y_tests = Y[-test_set_size:]
    XZ = XZ[:n]
    Y = Y[:n]
    true_model = TrueModel(W)
    sampler = GaussianSampler(j=0)
    if(regressor_name == "tm"):
        model = true_model
    else:
        model = return_model(regressor_name=regressor_name, seed=s)

    strat = get_strategy(args.strategy)

    betting_strategies = {args.strategy: strat}

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
                    test_data=(XZ_tests, Y_tests),   
                    wasserstein_resamplings = wasserstein_resamplings
                    )

    martingales = e_process.martingales(XZ, Y)
    lambdas = e_process.get_best_parameters()
    distances = e_process.wasserstein_distances
    errors = e_process.estimation_errors

    results_dir = Path("../results/csv/safety_simulations")
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        results_dir
        / f"simulations_safety_model_{regressor_name}_{s}_seed_n_{n}_strategy_{args.strategy}_martingales.pkl"
    )

    with open(filename, "wb") as f:
        pickle.dump((martingales, lambdas, distances, errors), f)
    return filename


def _run_seed_single_thread(s, args):
    # Multithreaded BLAS barely speeds up the small MLP fits, so with several
    # processes it only causes oversubscription
    with threadpool_limits(limits=1):
        return run_seed(s, args)


def main(args):
    if args.n_jobs <= 1 or len(args.seeds) == 1:
        for s in args.seeds:
            print(f"Saved {run_seed(s, args)}")
        return

    with ProcessPoolExecutor(max_workers=min(args.n_jobs, len(args.seeds))) as pool:
        futures = {pool.submit(_run_seed_single_thread, s, args): s for s in args.seeds}
        for future in as_completed(futures):
            print(f"Seed {futures[future]} done: saved {future.result()}")


if __name__ == "__main__":
    main(parse_args())
            

