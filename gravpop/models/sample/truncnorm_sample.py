import numpy as np
import pandas as pd

def generate_trunc_norm_slow(mu_1, sigma_1, mu_2, sigma_2, rho=None, a = [0.0, 0.0], b = [1.0, 1.0], oversampling=200):
    N = len(mu_1)
    if rho is None:
        rho = np.zeros(N)
    inbounds = False
    X = np.zeros((N,2))
    for i in range(N):
        while not (inbounds):
            Z = np.random.randn(2)
            x1,x2 = sigma_1[i] * Z[0] , sigma_2[i] * (rho[i] * Z[0] + np.sqrt(1 - rho[i]**2) * Z[1])
            x1,x2 = mu_1[i] + x1, mu_2[i] + x2 

            inbounds = ( (x1 < b[0]) & (x1 > a[0]) ) & ((x2 < b[1]) & (x2 > a[1]))
        
        X[i,0] = x1
        X[i,1] = x2
        inbounds=False

    return X

def ppd_truncCorrelatedanalytic(model, df, oversample=1):
    df_samps = []
    for _ in range(oversample):
        X = generate_trunc_norm_slow(mu_1 = df[model.mu_name_1].values,
                                     sigma_1 = df[model.sigma_name_1].values,
                                     mu_2 = df[model.mu_name_2].values,
                                     sigma_2 = df[model.sigma_name_2].values,
                                     rho = df[model.rho_name].values,
                                     a = np.ones(2)*model.a, b = np.ones(2)*model.b)

        df_samp = pd.DataFrame(X, columns=model.var_names)
        df_samps.append(df_samp)
        
    return pd.concat(df_samps)

def ppd_truncUncorrelatedanalytic(model, df, oversample=1):
    df_samps = []
    for _ in range(oversample):
        X = generate_trunc_norm_slow(mu_1 = df[model.models[0].mu_name].values,
                                     sigma_1 = df[model.models[0].sigma_name].values,
                                     mu_2 = df[model.models[1].mu_name].values,
                                     sigma_2 = df[model.models[1].sigma_name].values, 
                                     a = np.ones(2)*model.a, b = np.ones(2)*model.b)

        df_samp = pd.DataFrame(X, columns=model.var_names)
        df_samps.append(df_samp)
        
    return pd.concat(df_samps)