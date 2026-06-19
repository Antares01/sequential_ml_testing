import numpy as np
import pandas as pd
from utils import return_model
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import pyreadr
from utils import return_model


from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Convergence rates")
    parser.add_argument("--seeds", type=int, nargs="+", help="List of seeds")
    parser.add_argument("--model", type=str)
    parser.add_argument("--beta_strength", type=float)
    return parser.parse_args()


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


def main(args):
    #correlation_strength = args.correlation
    regressor = args.model

    for s in args.seeds:
        # Load RData file
        result = pyreadr.read_r('data/bike.RData')  # returns a dictionary
        bike = result['bike']

        season_order = ['WINTER', 'SPRING', 'SUMMER', 'FALL']
        yr_order = ['2011', '2012']
        month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
                    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        binary_order = ['NO', 'YES']
        weekday_order = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        weather_order = ['CLEAR', 'PARTLY CLOUDY', 'MISTY/CLOUDY', 'LIGHT RAIN/SNOW/STORM']
        mapping_dicts = {
            'season': {v: i for i, v in enumerate(season_order)},
            'yr': {v: i for i, v in enumerate(yr_order)},
            'mnth': {v: i for i, v in enumerate(month_order)},
            'holiday': {v: i for i, v in enumerate(binary_order)},
            'weekday': {v: i for i, v in enumerate(weekday_order)},
            'workingday': {v: i for i, v in enumerate(binary_order)},
            'weathersit': {v: i for i, v in enumerate(weather_order)}
        }

        for col, mapping in mapping_dicts.items():
            bike[col] = bike[col].map(mapping).astype(float)
        assert bike.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all(), "Not all columns are numeric!"

        X = bike.drop(columns='cnt')
        y = bike['cnt']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=s)
        imputer = IterativeImputer(random_state=s)

        # Fit on training data
        X_train= imputer.fit_transform(X_train)

        # Transform test data
        X_test = imputer.transform(X_test)
        X_train_df = pd.DataFrame(X_train, columns=X.columns)
        X_test_df = pd.DataFrame(X_test, columns=X.columns)
        y_test_df = pd.DataFrame(y_test)
        y_train_df = pd.DataFrame(y_train)  
        rng = np.random.default_rng(s)

        signal_train = (
            0.5 * X_train_df["temp"] +
            0.3 * X_train_df["hum"] -
            0.2 * X_train_df["windspeed"]
        )

        signal_test = (
            0.5 * X_test_df["temp"] +
            0.3 * X_test_df["hum"] -
            0.2 * X_test_df["windspeed"]
        )

        
        X_train_df["spurious"] = (
            correlation_strength * (signal_train - signal_train.mean()) / signal_train.std()
            + np.sqrt(1 - correlation_strength**2) * rng.normal(size=len(signal_train))
        )
        X_test_df["spurious"] = (
            correlation_strength * (signal_test - signal_train.mean()) / signal_train.std()
            + np.sqrt(1 - correlation_strength**2) * rng.normal(size=len(signal_test))
        )
        X_train = X_train_df.to_numpy()
        X_test = X_test_df.to_numpy()
        X = np.vstack([X_train, X_test])

        p = X_train.shape[1]
        print(p)
        print(X_train_df.columns)
        n_jobs=5

        rng = np.random.RandomState(s)

        importance_score=np.zeros((12, p))# 12 because there are 12 methods

        
        
        model=return_model(
            X_train,
            y_train,
            seed=s,
            n_jobs=n_jobs,
            regressor=regressor,
            verbose=False
        )

        scoring = [mean_squared_error, r2_score]
        names = ['MSE', 'r2_score']

        for jj in np.arange(len(names)):
            print('{}: {}'.format(names[jj], scoring[jj](y_test, model.predict(X_test))))
        r2_sc = r2_score(y_test, model.predict(X_test))
        mse = mean_squared_error(y_test, model.predict(X_test))

        loco = LOCO(
            estimator=model,
            random_state=s,
            loss = mean_squared_error,
            n_jobs=n_jobs,
        )
        loco.fit(X_train, y_train)
        loco_importance = loco.score(X_test, y_test.values)
        importance_score[9]= loco_importance["importance"].reshape((p,))

        loci = LOCI(
            estimator=model,
            random_state=s,
            loss=mean_squared_error,
            n_jobs=n_jobs,
        )

        loci.fit(X_train_df, y_train)
        loci_importance = loci.score(X_test_df, y_test.values)
        importance_score[11]= np.array(list(loci_importance.values()))
       
        sobol_cpi= Sobol_CPI(
                estimator=model,
                imputation_model=LassoCV(alphas=np.logspace(-3, 3, 10), cv=5),
                n_permutations=1,
                random_state=s,
                n_jobs=n_jobs)
        sobol_cpi.fit(X_train, y_train)

        cpi_importance = sobol_cpi.score(X_test, y_test.values, n_cal=1)
        importance_score[0]= cpi_importance["importance"].reshape((p,))
        n_cal = 100
        sobol_importance = sobol_cpi.score(X_test, y_test.values, n_cal=n_cal)
        importance_score[1]= sobol_importance["importance"].reshape((p,))

        sampler = GaussianSampler(X_train_df)
        wrk = Explainer(model.predict, X_train_df, loss=mean_squared_error, sampler=sampler)
        # PFI
        pfi = wrk.pfi(X_test_df, y_test_df)
        pfi_scores = pfi.fi_means_stds()
        importance_score[2]= np.array([
            pfi_scores[k] for k in pfi_scores.index if k != 'std'
        ])
        # CFI
        cfi = wrk.cfi(X_test_df, y_test_df)
        cfi_scores = cfi.fi_means_stds()
        importance_score[3]= np.array([
            cfi_scores[k] for k in cfi_scores.index if k != 'std'
        ])
        # cSAGEvf
        cSAGEvf = wrk.csagevfs(X_test_df, y_test_df, C='empty', nr_resample_marginalize=50)
        cSAGEvf_scores = cSAGEvf.fi_means_stds()
        importance_score[4]= np.array([
            cSAGEvf_scores[k] for k in cSAGEvf_scores.index if k != 'std'
        ])
        # scSAGEvfj
        scSAGEvfj = wrk.csagevfs(X_test_df, y_test_df, C='remainder', nr_resample_marginalize=50)
        scSAGEvfj_scores = scSAGEvfj.fi_means_stds()
        importance_score[5]= np.array([
            scSAGEvfj_scores[k] for k in scSAGEvfj_scores.index if k != 'std'
        ])
        # mSAGEvf
        mSAGEvf = wrk.msagevfs(X_test_df, y_test_df, C='empty', nr_resample_marginalize=50)
        mSAGEvf_scores = mSAGEvf.fi_means_stds()
        importance_score[6]= np.array([
            mSAGEvf_scores[k] for k in mSAGEvf_scores.index if k != 'std'
        ])
            
        # cSAGE
        cSAGE, ordering = wrk.csage(X_test_df, y_test_df, nr_resample_marginalize=50)
        cSAGE_scores = cSAGE.fi_means_stds()
        importance_score[7]= np.array([
            cSAGE_scores[k] for k in cSAGE_scores.index if k != 'std'
        ])   

        # mSAGE
        mSAGE, ordering = wrk.msage(X_test_df, y_test_df, nr_resample_marginalize=50)
        mSAGE_scores = mSAGE.fi_means_stds()
        importance_score[8]= np.array([
            mSAGE_scores[k] for k in mSAGE_scores.index if k != 'std'
        ]) 

    
        #LOCO Williamson
        ntrees = np.arange(100, 500, 100)
        lr = np.arange(.01, .1, .05)
        param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]
        ## set up cv objects
        cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'squared_error', max_depth = 3), param_grid = param_grid, cv = 5, n_jobs=n_jobs)
        for j in range(p):
            print("covariate: "+str(j))
            vimp = vimpy.vim(y = y.values, x = X, s = j, pred_func = cv_full, measure_type = "r_squared")
            vimp.get_point_est()
            vimp.get_influence_function()
            vimp.get_se()
            vimp.get_ci()
            vimp.hypothesis_test(alpha = 0.05, delta = 0)
            importance_score[10,j]+=vimp.vimp_*np.var(y)

        

        #Save the results
        f_res={}
        f_res = pd.DataFrame(f_res)
        for i in range(12):
            f_res1={}
            if i==0:
                    f_res1["method"] = ["Sobol-CPI(1)"]
            elif i==1:
                f_res1["method"]=["Sobol-CPI(100)"]
            if i==2:
                f_res1["method"]=["PFI"]
            elif i==3:
                f_res1["method"]=["CFI"]
            elif i==4:
                f_res1["method"] = ["cSAGEvf"]
            elif i==5:
                f_res1["method"]=["scSAGEj"]
            elif i==6: 
                f_res1["method"]=["mSAGEvf"]
            elif i==7:
                f_res1["method"]=["cSAGE"]
            elif i==8:
                f_res1["method"] = ["mSAGE"]
            elif i==9:
                f_res1["method"]=["LOCO"]
            elif i==10: 
                f_res1["method"]=["LOCO-W"]
            elif i==11:
                f_res1["method"]=["LOCI"]        
            f_res1["mse"]=mse
            f_res1["r2"]=r2_sc
            for k in range(p):
                f_res1["imp_V"+str(k)]=importance_score[i, k]
            f_res1=pd.DataFrame(f_res1)
            f_res=pd.concat([f_res, f_res1], ignore_index=True)
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../results/csv/bike/bike_spurious_model{regressor}_corr{correlation_strength}_seed{s}.csv"))
        f_res.to_csv(
        csv_path,
        index=False,
        ) 



# This is the main entry point of the script. It will be executed when the script is 
# run directly, i.e. `python python_script.py --seeds 1 2 3`.
if __name__ == "__main__":
    args = parse_args()
    main(args)


