import pymc as pm
import pytensor.tensor as pt
import pandas as pd
import numpy as np
import nutpie
import time


def make_model():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dustinstansbury/statistical-rethinking-2023/main/data/Primates301.csv",
        delimiter=";",
    )
    distance_matrix = pd.read_csv(
        "https://raw.githubusercontent.com/dustinstansbury/statistical-rethinking-2023/main/data/Primates301_distance_matrix.csv",
        delimiter=";",
    )
    distance_matrix.columns = distance_matrix.columns.map(int)

    df.dropna(subset="brain", inplace=True)
    distance_matrix = distance_matrix.loc[df.index, :].loc[:, df.index].copy()

    D_mat = distance_matrix / distance_matrix.max()

    def log_standardize(x):
        x = np.log(x)
        return (x - x.mean()) / x.std()

    coords = {"primate": df["name"].values}
    with pm.Model(coords=coords) as naive_imputation_model:
        G_obs = log_standardize(df.group_size.values)
        M_obs = log_standardize(df.body.values)
        B_obs = log_standardize(df.brain.values)

        # Priors
        alpha = pm.Normal("alpha", 0, 1)
        beta_G = pm.Normal("beta_G", 0, 0.5)
        beta_M = pm.Normal("beta_M", 0, 0.5)

        # Phylogenetic distance covariance prior, L1-kernel function
        eta_squared = pm.TruncatedNormal("eta_squared", 1, 0.25, lower=0.001)
        rho = pm.TruncatedNormal("rho", 3, 0.25, lower=0.001)
        K = pm.Deterministic("K", eta_squared * pt.exp(-rho * D_mat))

        # Naive imputation for G, M
        G = pm.Normal("G", 0, 1, observed=G_obs, dims="primate")
        M = pm.Normal("M", 0, 1, observed=M_obs, dims="primate")

        # Likelihood for B
        mu = alpha + beta_G * G + beta_M * M
        pm.MvNormal("B", mu=mu, cov=K, observed=B_obs)

    return naive_imputation_model


def sample_nutpie(model):
    compiled = nutpie.compile_pymc_model(model)
    start = time.time()
    nutpie.sample(compiled, chains=4, tune=1000, progressbar=False)
    end = time.time()
    return end - start


def sample_pymc(model):
    start = time.time()
    with model:
        pm.sample(progressbar=True, chain=4, cores=4)
    end = time.time()

    return end - start


def main():
    model = make_model()
    # time_nutpie = sample_nutpie(model)
    time_pymc = sample_pymc(model)
    # print(f"Nutpie: {time_nutpie}s")
    print(f"pymc: {time_pymc}s")


if __name__ == "__main__":
    main()
