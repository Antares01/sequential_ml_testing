import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    StackingRegressor
)
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy.signal.windows import tukey

from copy import deepcopy
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, log_loss


def update_g_func_static(pretraining_qs, past_qs, parameters, g_family):
    return g_family

def g_family_cb(q, q_tilde, param):
    lambd = param['lambda']
    bound = param['M'] 
    return lambd * np.clip(q_tilde - q, -bound, bound) / bound

def g_family_sign(q, q_tilde, param):
    lambd = param['lambda']
    return lambd * np.sign(q_tilde - q) 

'''def g_family_tanh(q, q_tilde, param, scale=20):
    eps = 1e-12
    return param["lambda"]* np.tanh(scale * (q_tilde - q) / (max(q_tilde, q)+eps))'''

def g_family_tanh(q, q_tilde, param):
    lambd = param['lambda']
    return  np.tanh(lambd*(q_tilde - q))

def g_family_kde(q, q_tilde, param):
    raise NotImplementedError("This function is not implemented. Give a history to the input so that update_g_func_kernel_density will create a g_family.")


### KDE main functions

# TODO: Must change this to output a n.array of shape (n, 2) where n is the number of pairs (q, q_tilde) to evaluate.

def initialize_kde_history(X, y, samplers, j_list, batches=[5], splits=5, resamplings_for_kde=10, model=LassoCV(), loss=mean_squared_error):
    """Build KDE history from contiguous fold-based test blocks.

    The history is assembled by splitting the data into contiguous folds.
    In each fold, one contiguous test segment is held out and the remainder is
    used for fitting a fresh model and samplers. Each test segment is then
    partitioned into independent contiguous blocks of the requested batch size.
    For each block we evaluate one q value and ``resamplings_for_kde`` q_tilde values
    obtained by perturbing the relevant feature column with samples generated
    by the fold-specific sampler. Any leftover points that do not complete a
    full block are discarded.
    """
    if len(samplers) != len(j_list):
        raise ValueError("The number of samplers should be equal to the number of features to be tested.")

    if splits < 2:
        raise ValueError("splits must be at least 2")

    if model is None:
        model = LassoCV()

    X = np.asarray(X)
    y = np.asarray(y).ravel()

    n_samples = len(y)
    if n_samples < splits:
        raise ValueError("Not enough samples to create the requested number of folds")

    q_dict = {batch_size: [] for batch_size in batches}
    q_tildes_dict = {batch_size: {j: [] for j in j_list} for batch_size in batches}

    test_size = n_samples // splits
    remainder = n_samples % splits

    for fold_idx in range(splits):
        start = fold_idx * test_size + min(fold_idx, remainder)
        end = start + test_size + (1 if fold_idx < remainder else 0)

        if start >= end or end > n_samples:
            continue

        test_idx = np.arange(start, end)
        train_idx = np.concatenate([np.arange(0, start), np.arange(end, n_samples)])

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        fold_model = clone(model)
        fold_model.fit(X[train_idx], y[train_idx])

        fold_samplers = []
        for sampler in samplers:
            if hasattr(sampler, "get_params"):
                sampler_copy = clone(sampler)
            else:
                sampler_copy = deepcopy(sampler)
            sampler_copy.fit(X[train_idx])
            fold_samplers.append(sampler_copy)

        for batch_size in batches:
            if batch_size <= 0:
                raise ValueError("batch sizes must be positive")

            block_count = len(test_idx) // batch_size
            if block_count <= 0:
                continue

            for block_offset in range(block_count):
                block_start = block_offset * batch_size
                block_end = block_start + batch_size
                block_idx = test_idx[block_start:block_end]

                X_block = X[block_idx]
                y_block = y[block_idx]

                y_pred = fold_model.predict(X_block)
                q_value = float(loss(y_pred, y_block))
                q_dict[batch_size].append(q_value)

                for feature_pos, sampler in zip(j_list, fold_samplers):
                    q_tilde_values = []
                    for _ in range(resamplings_for_kde):
                        sampled_feature = sampler.sample(X_block)
                        X_block_tilde = X_block.copy()
                        X_block_tilde[:, feature_pos] = sampled_feature
                        y_pred_tilde = fold_model.predict(X_block_tilde)
                        q_tilde_value = float(loss(y_pred_tilde, y_block))
                        q_tilde_values.append(q_tilde_value)

                    q_tildes_dict[batch_size][feature_pos].append(np.asarray(q_tilde_values, dtype=float))

    pretraining_qs_dict = {batch_size: {j: np.column_stack([
                np.repeat([qs for qs in q_dict[batch_size]], resamplings_for_kde),
                np.vstack([q_tildes for q_tildes in q_tildes_dict[batch_size][j]]).ravel()])
                for j in j_list} for batch_size in batches}

    return pretraining_qs_dict



'''def _prepare_kde_training_data(history, resamplings_for_kde, window_size):
    #Converts history data (i.e. past_qs in AntisymmetricBet class) from format 
    #(q,q_tilde_1,q_tilde_2,...)_i to more convenient np.array of form (q,q_tilde_1)_1,(q,q_tilde_2)_1,... 
    X = []
    for q, q_tildes in history:
        B = min(resamplings_for_kde, q_tildes.shape[0])
        if(window_size is not None and len(X) >= window_size):
            break
        for i in range(B):
            pairs = np.column_stack([
                q,
                q_tildes[i]
            ])
            X.append(pairs)

    if len(X) == 0:
        return np.empty((0, 2))

    return np.vstack(X)'''


def _prepare_kde_training_data(pretraining_qs, past_qs, resamplings_for_kde, window_size):
    #Converts history data (i.e. past_qs in AntisymmetricBet class) from format 
    #(q,q_tilde_1,q_tilde_2,...)_i to more convenient np.array of form (q,q_tilde_1)_1,(q,q_tilde_2)_1,... 

    if window_size is not None and len(pretraining_qs) + len(past_qs) > window_size:
        if len(past_qs) >= window_size:
            past_qs = past_qs[-window_size:]
            pretraining_qs = []
        else:
            pretraining_qs = pretraining_qs[-(window_size - len(past_qs)):]

    #for _, q_tildes in history:
    #    if q_tildes.shape[0] < resamplings_for_kde:
    #        raise ValueError(f"Not enough q_tilde values for the requested resamplings_for_kde. "
    #                         f"Expected at least {resamplings_for_kde}, but got {q_tildes.shape[0]}.")

    # TODO: add more checks, what if the pretraining dataset is empty etc
    
    if len(past_qs) == 0:
        return pretraining_qs

    return np.vstack([pretraining_qs, 
        np.column_stack([
            np.repeat([q for q,_ in past_qs], resamplings_for_kde),
            np.vstack([q_tilde[0:resamplings_for_kde] for _, q_tilde in past_qs]).ravel()])
    ])



def _prepare_weight_vector(X, window_size):
    m = X.shape[0]
    if window_size is None:
        return np.ones(m)
    weight_vector = np.clip(tukey(2 * window_size, alpha=0.3), 0.1, 1.0)[:window_size] #lower bound to avoid zero weights
    if m <= window_size:
        weight_vector = weight_vector[window_size - m:]
    return weight_vector


def _kde_g(kdes, q, q_tilde, param):           
    '''
    The g-family function for the KDE-based antisymmetric bet.
    Inputs: q: scalar, q_tilde: scalar, this function is called resamplings_for_kde times for each batch.'''

    points = np.column_stack([q, q_tilde])
    points_inverse = np.column_stack([q_tilde, q])
    key = (param["kernel"], param["bandwidth"])
    q_orig = np.exp(kdes[key].score_samples(points))
    q_inverse = np.exp(kdes[key].score_samples(points_inverse))
    eps = 1e-12
    return (q_orig - q_inverse)/(q_orig + q_inverse + eps)


# used_history_upper_bound was just for debugging, can be removed if not neded.

def update_g_func_kernel_density(pretraining_qs, past_qs, parameters, g_family, resamplings_for_kde = 10, window_size = None, used_history_upper_bound = None):
    '''Returns the KDE g-family for given history'''

    if used_history_upper_bound is not None and len(pretraining_qs) + len(past_qs) > used_history_upper_bound:
        print("Restricting the used history.")
        if len(pretraining_qs) >= used_history_upper_bound:
            pretraining_qs = pretraining_qs[0:used_history_upper_bound]
            past_qs = []
        else:
            past_qs = past_qs[0:(used_history_upper_bound - len(pretraining_qs))]

    

    X = _prepare_kde_training_data(pretraining_qs, past_qs, resamplings_for_kde, window_size = window_size)
    weight_vector = _prepare_weight_vector(X, window_size = window_size)
    kdes = {}
    for param in parameters:
        key = (param["kernel"], param["bandwidth"])
        kdes[key] = KernelDensity(kernel=param['kernel'], bandwidth=param['bandwidth']).fit(X, sample_weight=weight_vector)

    return lambda q, q_tilde, param: _kde_g(kdes, q, q_tilde, param)


def generate_update_kde(resamplings_for_kde = 10, window_size = None, used_history_upper_bound = None):
    return lambda pretraining_qs, past_qs, parameters, g_family : update_g_func_kernel_density(pretraining_qs, past_qs, parameters, g_family, resamplings_for_kde = resamplings_for_kde, window_size = window_size, used_history_upper_bound = used_history_upper_bound)



############ Parameter preparation functions

def prepare_kernel_density_parameters(kernel_list = ["gaussian", "tophat", "epanechnikov", 'exponential', 'linear'], bandwidth_start = 0.01, bandwidth_end = 3, bandwidth_num = 10):
    bandwidth_list = np.linspace(bandwidth_start, bandwidth_end, bandwidth_num)
    parameters = []
    for kernel in kernel_list:
        for bandwidth in bandwidth_list:
            parameters.append({"kernel": kernel, "bandwidth": bandwidth})
    return parameters

def prepare_coin_betting_parameters(lam_start = 0.01, lam_end = 1, lam_num = 10, M_start = 0.01, M_end = 5, M_num = 10):
    lam_values = np.linspace(lam_start, lam_end, lam_num)
    M_values = np.linspace(M_start, M_end, M_num)
    parameters = []
    for lam in lam_values:
        for M in M_values:
            parameters.append({"lambda": lam, "M": M})
    return parameters

def prepare_lambda_parameters(lam_start = 0.01, lam_end = 1, lam_num = 10): # For the sign and tanh e-value
    lam_values = np.linspace(lam_start, lam_end, lam_num)
    parameters = []
    for lam in lam_values:
            parameters.append({"lambda": lam})
    return parameters

def prepare_exponential_parameters(eta_start = 0.01, eta_end = 1, eta_num = 10): # For the exponential e-value
    return np.linspace(eta_start, eta_end, eta_num)

               
################################


def get_martingale_values(martingale_dict):
    b_last_used_list = []
    st_list = []
    for b in martingale_dict.keys():
        st_list.append(martingale_dict[b]["St"])
        b_last_used_list.append(martingale_dict[b]["last_used_idx"])
    return np.array(st_list).mean(), np.array(b_last_used_list).max()


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def lasso_cv_online_learning(X, y, models_dict, val_prcg=0.2):
    """
    Online hyper-parameter tuning using ensemble of Lasso models
    :param X: The data matrix with size (n, d).
    :param y: A vector of labels with size (n, 1).
    :param models_dict: A dictionary contains M models.
    The keys are the values of the tuned parameter (the regularization constant the multiplies the L1 term
    in the loss function).
    The values in the dictionary are the lasso models with the corresponding parameter.
    :param val_prcg: The percentage of data to be used for validation.
    :return: The regularization constant of the model that got the best score on the validation data.
    """
    train_idx = int(len(y) * (1-val_prcg))
    X_train = X[:train_idx, :]
    y_train = y[:train_idx]
    X_val = X[train_idx:, :]
    y_val = y[train_idx:]
    alpha_vec = list(models_dict.keys())
    score = -1
    best_alpha = alpha_vec[0]
    for alpha in alpha_vec:
        models_dict[alpha].fit(X_train, y_train.ravel())
        model_score = models_dict[alpha].score(X_val, y_val.ravel())
        if model_score > score:
            best_alpha = alpha
            score = model_score
    return best_alpha



def return_model(regressor_name, seed):
    # "lr" "lasso" "dt" "rf" "gb" "nn" "svr"
    if regressor_name == "lr":
        return LinearRegression()
    elif regressor_name == "lasso":
        return Lasso()
    elif regressor_name == "dt":
        return DecisionTreeRegressor(random_state=seed)
    elif regressor_name == "rf":
        return RandomForestRegressor(random_state=seed)
    elif regressor_name == "gb":
        return GradientBoostingRegressor(random_state=seed)
    elif regressor_name == "nn":
        return MLPRegressor(random_state=seed, max_iter=1000)
    elif regressor_name == "svr":
        return SVR()
    


def generate_dataset(n, beta=1.0, d=19, seed=None):
    rng = np.random.default_rng(seed)

    # Fixed parameters
    W = rng.normal(size=d)
    U = rng.normal(size=d)

    # Generate Z ~ N(0, I_d)
    Z = rng.normal(size=(n, d))

    # Generate X | Z ~ N(U^T Z, 1)
    X = Z @ U + rng.normal(size=n)

    # Generate Y
    epsilon = rng.normal(size=n)
    Y = (Z @ W) ** 2 + beta * X + epsilon

    # Design matrix: first column is X, remaining columns are Z
    X_design = np.column_stack((X, Z))

    return X_design, Y

def generate_dataset_dirichlet(n, beta=1.0, d=19, seed=None):
    rng = np.random.default_rng(seed)

    # Fixed parameters
    W = np.random.dirichlet(np.ones(d), size = 1).flatten()
    U = np.random.dirichlet(np.ones(d), size = 1).flatten()

    # Generate Z ~ N(0, I_d)
    Z = rng.normal(size=(n, d))

    # Generate X | Z ~ N(U^T Z, 1)
    X = Z @ U + rng.normal(size=n)

    # Generate Y
    epsilon = rng.normal(size=n)
    Y = (Z @ W) ** 2 + beta * X + epsilon

    # Design matrix: first column is X, remaining columns are Z
    X_design = np.column_stack((X, Z))

    return X_design, Y