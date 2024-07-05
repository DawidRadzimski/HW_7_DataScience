import numpy as np
print("NumPy version:", np.__version__)

from surprise import Dataset, SVD, SVDpp, NMF
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import accuracy

# Wczytanie danych MovieLens
data = Dataset.load_builtin("ml-100k", prompt=False)

# Wybór najlepszych parametrów za pomocą walidacji krzyżowej dla SVD
param_grid_svd = {"n_epochs": [20, 30], "lr_all": [0.002, 0.005], "reg_all": [0.4, 0.6]}
gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse'], cv=3)
gs_svd.fit(data)

print("Najlepszy wynik RMSE dla SVD:", gs_svd.best_score["rmse"])
print("Najlepsze parametry dla SVD:", gs_svd.best_params["rmse"])

algo_svd = SVD(n_epochs=gs_svd.best_params["rmse"]["n_epochs"],
               lr_all=gs_svd.best_params["rmse"]["lr_all"],
               reg_all=gs_svd.best_params["rmse"]["reg_all"])
results_svd = cross_validate(algo_svd, data, measures=["RMSE", "MAE"], cv=3, verbose=True)

# Wybór najlepszych parametrów za pomocą walidacji krzyżowej dla SVD++
param_grid_svdpp = {"n_epochs": [20, 30], "lr_all": [0.002, 0.005], "reg_all": [0.4, 0.6]}
gs_svdpp = GridSearchCV(SVDpp, param_grid_svdpp, measures=['rmse'], cv=3)
gs_svdpp.fit(data)

print("Najlepszy wynik RMSE dla SVD++:", gs_svdpp.best_score["rmse"])
print("Najlepsze parametry dla SVD++:", gs_svdpp.best_params["rmse"])

algo_svdpp = SVDpp(n_epochs=gs_svdpp.best_params["rmse"]["n_epochs"],
                   lr_all=gs_svdpp.best_params["rmse"]["lr_all"],
                   reg_all=gs_svdpp.best_params["rmse"]["reg_all"])
results_svdpp = cross_validate(algo_svdpp, data, measures=["RMSE", "MAE"], cv=3, verbose=True)

# Wybór najlepszych parametrów za pomocą walidacji krzyżowej dla NMF
param_grid_nmf = {"n_epochs": [20, 30], "n_factors": [15, 20], "reg_pu": [0.4, 0.6], "reg_qi": [0.4, 0.6]}
gs_nmf = GridSearchCV(NMF, param_grid_nmf, measures=["rmse"], cv=3)
gs_nmf.fit(data)

print("Najlepszy wynik RMSE dla NMF:", gs_nmf.best_score["rmse"])
print("Najlepsze parametry dla NMF:", gs_nmf.best_params["rmse"])

algo_nmf = NMF(n_epochs=gs_nmf.best_params["rmse"]["n_epochs"],
               n_factors=gs_nmf.best_params["rmse"]["n_factors"],
               reg_pu=gs_nmf.best_params["rmse"]["reg_pu"],
               reg_qi=gs_nmf.best_params["rmse"]["reg_qi"])
results_nmf = cross_validate(algo_nmf, data, measures=["RMSE", "MAE"], cv=3, verbose=True)

# Porównanie wyników
print("Wyniki SVD:", results_svd)
print("Wyniki SVD++:", results_svdpp)
print("Wyniki NMF:", results_nmf)

mean_rmse_svd = results_svd["test_rmse"].mean()
mean_mae_svd = results_svd["test_mae"].mean()

mean_rmse_svdpp = results_svdpp["test_rmse"].mean()
mean_mae_svdpp = results_svdpp["test_mae"].mean()

mean_rmse_nmf = results_nmf["test_rmse"].mean()
mean_mae_nmf = results_nmf["test_mae"].mean()

print(f"SVD: RMSE={mean_rmse_svd}, MAE={mean_mae_svd}")
print(f"SVD++: RMSE={mean_rmse_svdpp}, MAE={mean_mae_svdpp}")
print(f"NMF: RMSE={mean_rmse_nmf}, MAE={mean_mae_nmf}")
