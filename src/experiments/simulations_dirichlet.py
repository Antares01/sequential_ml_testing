import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pyreadr
from utils import return_model, generate_dataset_dirichlet

from ML_e_process import ML_e_process
from sklearn.linear_model import LassoCV

from ExponentialBet import ExponentialBet  
from AntisymmetricBet import AntisymmetricBet  # adjust import
from utils import prepare_exponential_parameters, prepare_coin_betting_parameters, g_family_cb, update_g_func_static, prepare_lambda_parameters, g_family_tanh, g_family_sign
from utils import prepare_kernel_density_parameters, g_family_kde, update_g_func_kernel_density, initialize_kde_history

from copy import deepcopy
from Sampler import DefaultSampler
from pathlib import Path
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Convergence rates")
    parser.add_argument("--seeds", type=int, nargs="+", help="List of seeds")
    parser.add_argument("--model", type=str)
    parser.add_argument("--beta_strength", type=float)
    return parser.parse_args()



def main(args):
    #correlation_strength = args.correlation
    regressor_name= args.model
    n = 2000
    n_init = 100
    batch_list = [2, 5, 10, 20]
    beta_strength = args.beta_strength
    list_js = [0]

    for s in args.seeds:
        X, Y = generate_dataset_dirichlet(n, beta=beta_strength, d=19, seed=s)
        model = return_model(regressor_name=regressor_name, seed=s)
        rng = np.random.default_rng(s)


        etas = prepare_exponential_parameters(0.01, 5, 10)
        strategy_exp = ExponentialBet(
            parameters=etas,
            exact=True
        )



        params_cb = prepare_coin_betting_parameters(lam_start = 0.01, lam_end = 0.95, lam_num = 10, M_start = 0.01, M_end = 5, M_num = 10)
        strategy_cb = AntisymmetricBet(
            g_family= g_family_cb, 
            update_g_func = update_g_func_static, 
            parameters=params_cb,
        )


        params_tanh = prepare_lambda_parameters(lam_start = 0.01, lam_end = 0.95, lam_num = 10)
        strategy_tanh = AntisymmetricBet(
            g_family= g_family_tanh, 
            update_g_func = update_g_func_static, 
            parameters=params_tanh,
            prequential=True,
        )

        params_sign = prepare_lambda_parameters(lam_start = 0.01, lam_end = 0.95, lam_num = 10)

        strategy_sign = AntisymmetricBet(
            g_family= g_family_sign, 
            update_g_func = update_g_func_static, 
            parameters=params_sign,
            prequential=False,
        )

        strategy_sign_preq = AntisymmetricBet(
            g_family= g_family_sign, 
            update_g_func = update_g_func_static, 
            parameters=params_sign,
            prequential=True,
        )


        betting_strategies = {
            "exponential": strategy_exp,
            "coin_betting": strategy_cb,
            "tanh": strategy_tanh,
            "sign": strategy_sign,
            "sign_preq": strategy_sign_preq,

        }

        bets_js_bs = {
                    j: {
                        b: {
                            key: deepcopy(strategy)
                            for key, strategy in betting_strategies.items()
                        }
                        for b in batch_list
                    }
                    for j in list_js
                }
        samplers_kde_init = [DefaultSampler(j=j) for j in list_js]
        q, q_tilde = initialize_kde_history(X[:n_init], Y[:n_init], samplers_kde_init, list_js, batch_list, model = model)
        for j in list_js:
            for b in batch_list:
                params_kde = prepare_kernel_density_parameters()
                
                
                past_qs = list(zip(q[b], q_tilde[b][j]))
                strategy_kde = AntisymmetricBet(
                    g_family= g_family_kde, 
                    update_g_func = update_g_func_kernel_density, 
                    parameters=params_kde,
                    prequential=True,
                    past_qs = past_qs
                )
                bets_js_bs[j][b]["kde"] =  strategy_kde

        betting_strategies = {
            "exponential": strategy_exp,
            "coin_betting": strategy_cb,
            "tanh": strategy_tanh,
            "sign": strategy_sign,
            "sign_preq": strategy_sign_preq,
            "kde": strategy_kde

        }# We use this just for the keys

        #batch_list = [1, 2, 5]
        e_process = ML_e_process(batch_list=batch_list, n_init=n_init, b_resamplings=50, study_j=list_js,
                        betting_strategies=betting_strategies, 
                        model=model,
                        #learn_conditional_distribution=get_data_statistics,
                        bets_js_bs=bets_js_bs,
                        samplers=None, #sampling_args={}
                        learn_conditional_distribution=True,
                        optional_stopping=True, 
                        )

        martingales = e_process.martingales(X, y=Y)


        results_dir = Path("../results/csv/simulations")
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            results_dir
            / f"simulations_dirich_beta{beta_strength}_model{regressor_name}_seed{s}_martingales.pkl"
        )

        with open(filename, "wb") as f:
            pickle.dump(martingales, f)
            





# This is the main entry point of the script. It will be executed when the script is 
# run directly, i.e. `python python_script.py --seeds 1 2 3`.
if __name__ == "__main__":
    args = parse_args()
    main(args)


