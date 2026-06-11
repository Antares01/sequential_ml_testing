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
    return param["lambda"]* np.tanh(scale * (q_tilde - q) / (np.max(q_tilde, q)+eps))

def update_g_func_kernel_density(history, parameters, g_family, resamplings = 10):
    X = []

    for q, q_tildes in history:

        B = min(resamplings, q_tildes.shape[1])

        for i in range(B):

            pairs = np.column_stack([
                q,
                q_tildes[:, i]
            ])

            X.append(pairs)

    X = np.vstack(X)

    kdes = {}
    for param in parameters:
        key = (param["kernel"], param["bandwidth"])
        kdes[key] = KernelDensity(kernel=param['kernel'], bandwidth=param['bandwidth']).fit(X)

    def new_g(q, q_tilde, param):
        points = np.column_stack([q, q_tilde])
        points_inverse = np.column_stack([q_tilde, q])
        key = (param["kernel"], param["bandwidth"])
        q_orig = np.exp(kdes[key].score_samples(points))
        q_inverse = np.exp(kdes[key].score_samples(points_inverse))
        eps = 1e-12
        return (q_orig - q_inverse)/(q_orig + q_inverse + eps)

    return new_g



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
