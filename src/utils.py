import numpy as np

from sklearn.neighbors import KernelDensity


def quadratic_loss(y_pred, y_true):
    return (y_pred-y_true)**2

def log_loss(y_pred, y_true, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

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


def _prepare_kde_training_data(history, resamplings = 10):
    X = []

    for q, q_tildes in history:
        B = min(resamplings, q_tildes.shape[1])
        for i in range(B):
            pairs = np.column_stack([
                q,
                q_tildes[:, i]
            ])

            X.append(pairs)

    return np.vstack(X)

def _kde_g(kdes, q, q_tilde, param):
    points = np.column_stack([q, q_tilde])
    points_inverse = np.column_stack([q_tilde, q])
    key = (param["kernel"], param["bandwidth"])
    q_orig = np.exp(kdes[key].score_samples(points))
    q_inverse = np.exp(kdes[key].score_samples(points_inverse))
    eps = 1e-12
    return (q_orig - q_inverse)/(q_orig + q_inverse + eps)

def update_g_func_kernel_density(history, parameters, g_family, resamplings = 10):
    X = _prepare_kde_training_data(history, resamplings)

    kdes = {}
    for param in parameters:
        key = (param["kernel"], param["bandwidth"])
        kdes[key] = KernelDensity(kernel=param['kernel'], bandwidth=param['bandwidth']).fit(X)

    return lambda q, q_tilde, param: _kde_g(kdes, q, q_tilde, param)


def prepare_coin_betting_parameters(lam_start = 0.01, lam_end = 1, lam_num = 10, M_start = 0.01, M_end = 5, M_num = 10):
    lam_values = np.linspace(lam_start, lam_end, lam_num)
    M_values = np.linspace(M_start, M_end, M_num)
    parameters = []
    for lam in lam_values:
        for M in M_values:
            parameters.append({"lambda": lam, "M": M})
    return parameters

def prepare_kernel_density_parameters(kernel_list = ["gaussian", "tophat", "epanechnikov", 'exponential', 'linear'], bandwidth_start = 0.01, bandwidth_end = 2, bandwidth_num = 10):
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
