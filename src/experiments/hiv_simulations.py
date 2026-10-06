import argparse
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from math import lcm
from pathlib import Path

import numpy as np
from sklearn.linear_model import LassoCV
from threadpoolctl import threadpool_limits

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from data.data import get_hiv_data
from AntisymmetricBet import AntisymmetricBet
from ExponentialBet import ExponentialBet
from ML_e_process import ML_e_process
from Sampler import DefaultSampler, ProbaSampler, RegressorSampler
from utils import (
    g_family_generalized_sign,
    g_family_kde,
    g_family_sign,
    g_family_tanh,
    generate_update_kde,
    initialize_kde_history,
    prepare_coin_betting_parameters,
    prepare_exponential_parameters,
    prepare_kernel_density_parameters,
    prepare_lambda_parameters,
    return_model,
    update_g_func_static,
)

ALL_STRATEGIES = ["exponential", "generalized_sign", "tanh", "sign", "sign_preq", "kde"]





class HIV_e_process(ML_e_process):
    """
    ML_e_process always calls `_compute_estimation_error`, which needs a `true_model` and `test_data`.
    Neither exists for real data, so the call is skipped here.
    """

    def _compute_estimation_error(self):
        if self.true_model is None or self.test_data is None:
            return
        super()._compute_estimation_error()

    def get_best_parameters(self):
        # Non-prequential strategies and ExponentialBet have no `get_best_parameters`, so read the attribute directly
        return {
            j: {b: {name: getattr(self.bets_js_bs[j][b][name], "best_params", None) for name in self.startegy_names}
                for b in self.batch_list}
            for j in self.study_j
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Sequential conditional independence tests on the HIV dataset")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="list of seeds")
    parser.add_argument("--n_init", type=int, default=800, help="number of pretraining samples, must be a multiple of every batch size")
    parser.add_argument("--n", type=int, default=None, help="total number of samples to use (default: whole dataset)")
    parser.add_argument("--features", type=int, nargs="+", default=[10, 61, 62], help="indices of the tested features")
    parser.add_argument("--batches", type=int, nargs="+", default=[5, 10, 20], help="list of batch sizes to test")
    parser.add_argument("--resamplings", type=int, default=20, help="number of dummy copies X_tilde per batch")
    parser.add_argument("--kde_resamplings", type=int, default=None, help="dummy copies per batch used to fit the KDE (default: --resamplings)")
    parser.add_argument("--kde_window", type=int, default=None, help="window size for the KDE history (default: no window)")
    parser.add_argument("--kde_two_dimensional", action="store_true", help="fit the KDE on (q, q_tilde) pairs instead of q - q_tilde")
    parser.add_argument("--strategies", type=str, nargs="+", default=ALL_STRATEGIES, choices=ALL_STRATEGIES)
    parser.add_argument("--model", type=str, default="lassocv", help="lassocv or any name accepted by utils.return_model")
    parser.add_argument("--shuffle", action="store_true", help="shuffle the rows of the dataset with the seed before running")
    parser.add_argument("--output_dir", type=str, default=str(ROOT_DIR / "results" / "csv" / "hiv"))
    parser.add_argument("--n_jobs", type=int, default=1, help="number of seeds to run in parallel (one process per seed)")
    return parser.parse_args()


def get_model(name, seed):
    if name == "lassocv":
        return LassoCV()
    model = return_model(regressor_name=name, seed=seed)
    if model is None:
        raise ValueError(f"Unknown model: {name}")
    return model


def build_strategies(names):
    strategies = {}
    if "exponential" in names:
        strategies["exponential"] = ExponentialBet(
            parameters=prepare_exponential_parameters(0.01, 5, 10),
            exact=True,
        )
    if "generalized_sign" in names:
        strategies["generalized_sign"] = AntisymmetricBet(
            g_family=g_family_generalized_sign,
            update_g_func=update_g_func_static,
            parameters=prepare_coin_betting_parameters(lam_start=0.01, lam_end=0.95, lam_num=10, M_start=0.01, M_end=5, M_num=10),
        )
    if "tanh" in names:
        strategies["tanh"] = AntisymmetricBet(
            g_family=g_family_tanh,
            update_g_func=update_g_func_static,
            parameters=prepare_lambda_parameters(lam_start=0.01, lam_end=0.95, lam_num=10),
            prequential=True,
        )
    if "sign" in names:
        strategies["sign"] = AntisymmetricBet(
            g_family=g_family_sign,
            update_g_func=update_g_func_static,
            parameters=prepare_lambda_parameters(lam_start=0.01, lam_end=0.95, lam_num=10),
            prequential=False,
        )
    if "sign_preq" in names:
        strategies["sign_preq"] = AntisymmetricBet(
            g_family=g_family_sign,
            update_g_func=update_g_func_static,
            parameters=prepare_lambda_parameters(lam_start=0.01, lam_end=0.95, lam_num=10),
            prequential=True,
        )
    return strategies


def build_bets_js_bs(strategies, X_init, Y_init, args, model):
    # Every (feature, batch) pair gets its own copy of every strategy
    bets_js_bs = {
        j: {b: {key: deepcopy(strategy) for key, strategy in strategies.items()} for b in args.batches}
        for j in args.features
    }
    if "kde" not in args.strategies:
        return bets_js_bs

    # The KDE bet needs a history of (q, q_tildes), built by cross-validation on the pretraining data only
    kde_resamplings = args.kde_resamplings if args.kde_resamplings is not None else args.resamplings
    updater_kde = generate_update_kde(
        resamplings=kde_resamplings,
        window_size=args.kde_window,
        one_dimensional=not args.kde_two_dimensional,
    )
    history_samplers = [DefaultSampler(j=j) for j in args.features]
    q, q_tilde = initialize_kde_history(
        X_init, Y_init, history_samplers, args.features, args.batches,
        model=model, resamplings=kde_resamplings, splits = 7
    )
    for j in args.features:
        for b in args.batches:
            bets_js_bs[j][b]["kde"] = AntisymmetricBet(
                g_family=g_family_kde,
                update_g_func=updater_kde,
                parameters=prepare_kernel_density_parameters(),
                prequential=True,
                past_qs=list(zip(q[b], q_tilde[b][j])),
            )
    return bets_js_bs


def check_args(args, n):
    # ML_e_process only updates the martingale of batch b at time points t with t % b == 0,
    # so n_init has to be aligned with every batch size, otherwise some batches are skipped
    step = lcm(*args.batches)
    if args.n_init % step != 0:
        raise ValueError(f"--n_init ({args.n_init}) must be a multiple of lcm(batches) = {step}")
    if args.n_init + max(args.batches) >= n:
        raise ValueError(f"--n_init ({args.n_init}) leaves no data for testing (n = {n})")
    if "kde" in args.strategies and args.n_init < 5 * max(args.batches):
        raise ValueError("--n_init is too small to build the KDE history (5 folds)")


def run_seed(s, args, X_full, Y_full, features_names, n, output_dir):
    np.random.seed(s)
    if args.shuffle:
        perm = np.random.default_rng(s).permutation(X_full.shape[0])
        X, Y = X_full[perm][:n], Y_full[perm][:n]
    else:
        X, Y = X_full[:n], Y_full[:n]

    model = get_model(args.model, s)
    strategies = build_strategies(args.strategies)
    bets_js_bs = build_bets_js_bs(strategies, X[:args.n_init], Y[:args.n_init], args, model)

    e_process = HIV_e_process(
        batch_list=args.batches,
        n_init=args.n_init,
        b_resamplings=args.resamplings,
        study_j=args.features,
        betting_strategies=strategies,
        model=get_model(args.model, s),
        samplers={j: DefaultSampler(j=j) for j in args.features},
        learn_conditional_distribution=True,
        optional_stopping=True,
        bets_js_bs=bets_js_bs,
    )

    print(f"seed {s}: n = {n}, n_init = {args.n_init}, features = {list(features_names[args.features])}")
    martingales = e_process.martingales(X, Y)
    best_params = e_process.get_best_parameters()

    filename = output_dir / (
        f"hiv_model{args.model}_ninit{args.n_init}_n{n}_B{args.resamplings}"
        f"{'_shuffled' if args.shuffle else ''}_seed{s}_martingales.pkl"
    )
    with open(filename, "wb") as f:
        pickle.dump({"martingales": martingales, "best_params": best_params, "config": vars(args),
                     "features_names": list(features_names[args.features])}, f)
    return filename


def _run_seed_single_thread(*run_args):
    # With several processes, multithreaded BLAS only causes oversubscription
    with threadpool_limits(limits=1):
        return run_seed(*run_args)


def main(args):
    X_full, Y_full, features_names = get_hiv_data()
    X_full = X_full.astype(float)
    Y_full = Y_full.astype(float).ravel()
    n = X_full.shape[0] if args.n is None else min(args.n, X_full.shape[0])
    check_args(args, n)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.n_jobs <= 1 or len(args.seeds) == 1:
        for s in args.seeds:
            print(f"saved {run_seed(s, args, X_full, Y_full, features_names, n, output_dir)}")
        return

    with ProcessPoolExecutor(max_workers=min(args.n_jobs, len(args.seeds))) as pool:
        futures = {pool.submit(_run_seed_single_thread, s, args, X_full, Y_full, features_names, n, output_dir): s
                   for s in args.seeds}
        for future in as_completed(futures):
            print(f"seed {futures[future]} done: saved {future.result()}")


if __name__ == "__main__":
    main(parse_args())
