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

from sklearn.base import clone
from sklearn.metrics import mean_squared_error, log_loss


def update_g_func_static(history, parameters, g_family):
    return g_family

def g_family_cb(q, q_tilde, param):
    lambd = param['lambda']
    bound = param['M'] 
    return lambd * np.clip(q_tilde - q, -bound, bound) / bound

def g_family_sign(q, q_tilde, param):
    lambd = param['lambda']
    return lambd * np.sign(q_tilde - q) 

def g_family_tanh(q, q_tilde, param, scale=20):
    eps = 1e-12
    return param["lambda"]* np.tanh(scale * (q_tilde - q) / (max(q_tilde, q)+eps))

def g_family_kde(q, q_tilde, param):
    raise NotImplementedError("This function is not implemented. Give an history to the input so that update_g_func_kernel_density will create a g_family.")

def _prepare_kde_training_data(history, resamplings = 10, window_size = 800):
    X = []

    for q, q_tildes in history:
        B = min(resamplings, q_tildes.shape[0])
        if(len(X) >= window_size):
            break
        for i in range(B):
            pairs = np.column_stack([
                q,
                q_tildes[i]
            ])

            X.append(pairs)

    return np.vstack(X)

def _prepare_weight_vector(X, window_size = 800):
    m = X.shape[0]
    weight_vector = np.clip(tukey(2 * window_size, alpha=0.3), 0.1, 1.0)[:window_size] #lower bound to avoid zero weights
    if m <= window_size:
        weight_vector = weight_vector[window_size - m:]
    return weight_vector


def _kde_g(kdes, q, q_tilde, param):
    points = np.column_stack([q, q_tilde])
    points_inverse = np.column_stack([q_tilde, q])
    key = (param["kernel"], param["bandwidth"])
    q_orig = np.exp(kdes[key].score_samples(points))
    q_inverse = np.exp(kdes[key].score_samples(points_inverse))
    eps = 1e-12
    return (q_orig - q_inverse)/(q_orig + q_inverse + eps)

def update_g_func_kernel_density(history, parameters, g_family, resamplings = 10):
    window_size = 800
    X = _prepare_kde_training_data(history, resamplings, window_size = 800)
    weight_vector = _prepare_weight_vector(X, window_size = 800)
    kdes = {}
    for param in parameters:
        key = (param["kernel"], param["bandwidth"])
        kdes[key] = KernelDensity(kernel=param['kernel'], bandwidth=param['bandwidth']).fit(X, sample_weight=weight_vector)

    return lambda q, q_tilde, param: _kde_g(kdes, q, q_tilde, param)


def prepare_coin_betting_parameters(lam_start = 0.01, lam_end = 1, lam_num = 10, M_start = 0.01, M_end = 5, M_num = 10):
    lam_values = np.linspace(lam_start, lam_end, lam_num)
    M_values = np.linspace(M_start, M_end, M_num)
    parameters = []
    for lam in lam_values:
        for M in M_values:
            parameters.append({"lambda": lam, "M": M})
    return parameters

def prepare_kernel_density_parameters(kernel_list = ["gaussian", "tophat", "epanechnikov", 'exponential', 'linear'], bandwidth_start = 0.01, bandwidth_end = 3, bandwidth_num = 10):
    bandwidth_list = np.linspace(bandwidth_start, bandwidth_end, bandwidth_num)
    parameters = []
    for kernel in kernel_list:
        for bandwidth in bandwidth_list:
            parameters.append({"kernel": kernel, "bandwidth": bandwidth})
    return parameters

def prepare_lambda_parameters(lam_start = 0.01, lam_end = 1, lam_num = 10): # For the sign and tanh e-value
    lam_values = np.linspace(lam_start, lam_end, lam_num)
    parameters = []
    for lam in lam_values:
            parameters.append({"lambda": lam})
    return parameters

def prepare_exponential_parameters(eta_start = 0.01, eta_end = 1, eta_num = 10): # For the exponential e-value
    return np.linspace(eta_start, eta_end, eta_num)

def initialize_kde_history(X, y, samplers, j_list, batches = [5], splits = 5, batches_to_draw_randomly = 20, resamplings = 10, model= LassoCV(), loss= mean_squared_error ):
    if(len(samplers) != len(j_list)):
        raise ValueError("The number of samplers should be equal to the number of features to be tested.")

    cv = KFold(n_splits=splits, shuffle=False)
    q = {b : [] for b in batches}
    q_tilde = {b : { j : [] 
                    for j in j_list} 
                    for b in batches}
    y_pred = np.array([])
    y_pred_tildes = {j : [np.array([]) for r in range(resamplings)] for j in j_list} # TODO: This needs to be changed so resamplings is resamplings_kde_estimate
    for train_idx, test_idx in cv.split(X, y):
        model_fold = clone(model)
        model_fold.fit(X[train_idx], y[train_idx])
        y_pred = np.append(y_pred, model_fold.predict(X[test_idx]))
        for j, sampler in zip(j_list, samplers):
                sampler.fit(X[train_idx])
                for r in range(resamplings): 
                    X_j_tildes = sampler.sample(X[test_idx])
                    X_test_tilde = X[test_idx].copy()
                    X_test_tilde[:, j] = X_j_tildes
                    y_pred_tildes[j][r] = np.append(y_pred_tildes[j][r], model_fold.predict(X_test_tilde))
                
        for b in batches:
            for _ in range(batches_to_draw_randomly):
                indices = np.random.choice(len(y_pred), size=b, replace=False)
                y_pred_batch = y_pred[indices]
                q[b].append(loss(y_pred_batch, y[indices]))
                for j in j_list:
                    q_tilde_temp = np.array([])
                    for r in range(resamplings):
                        y_pred_tilde_batch = y_pred_tildes[j][r][indices]
                        l_tilde = loss(y_pred_tilde_batch, y[indices])
                        q_tilde_temp = np.append(q_tilde_temp, l_tilde)
                    q_tilde[b][j].append(q_tilde_temp)
                        
    return q, q_tilde
               

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