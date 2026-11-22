"""Partie 1 : benchmark principal, 10 méthodes x 21 jeux de données réels x 10 découpages 70/30.

Ce que le sujet demande : comparer nos méthodes d'ensemble à des références et à d'autres ensembles
sur au moins 15 jeux de données, avec une métrique adaptée. C'est ce script qui produit les données de
cette comparaison.

le Protocole que nous avons choisi :
    - 21 jeux de données réels (bankmarketing, 45 211 exemples, est exclu : trop long pour 10 méthodes x 10 découpages) ;
    - 10 découpages stratifiés 70 % train / 30 % test, tirés avec la même graine ; chaque méthode voit les
      mêmes découpages, ce qui permet des comparaisons appariées ;
    - hyperparamètres fixes, proches des valeurs par défaut, sans réglage ;
    - un jeu est dit déséquilibré si sa classe minoritaire représente moins de 20 % des exemples.

Les 10 méthodes sont : arbre seul, régression logistique, 7-NN (références) ; bagging, AdaBoost avec stumps,
AdaBoost avec arbres de profondeur 3, stacking (nos implémentations) ; forêt aléatoire, gradient boosting,
HistGradientBoosting (scikit-learn).

Métriques calculées sur la partie test : erreur, balanced accuracy, F1 de la classe minoritaire, AUC,
et temps d'entraînement.

Résultats produites: Un CSV par jeu de données : c'est le cache du benchmark. 
benchmark.py y écrit chaque jeu dès qu'il est terminé, donc si on l'interrompe et le relanc,
il ne refait pas les jeux déjà calculés. 
benchmark.csv est la réunion de tous ces fichiers.

"""
import os
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchkit as bk
from ensemble_methods import Bagging, AdaBoost, Stacking

RES = os.path.join(HERE, "..", "results")
os.makedirs(os.path.join(RES, "bench"), exist_ok=True)

DATASETS = [d for d in bk.REAL_SMALL + bk.REAL_LARGE if d != "bankmarketing"]
IMBALANCE_THRESHOLD = 0.20
N_REP = 10


def models(seed):
    return {
        "tree": DecisionTreeClassifier(random_state=seed),
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "7nn": make_pipeline(StandardScaler(), KNeighborsClassifier(7)),
        "bagging": Bagging(100, random_state=seed),
        "adaboost_stumps": AdaBoost(200, 1),
        "adaboost_depth3": AdaBoost(200, 3),
        "stacking": Stacking(random_state=seed),
        "rf": RandomForestClassifier(200, random_state=seed, n_jobs=1),
        "gb": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed),
        "hgb": HistGradientBoostingClassifier(random_state=seed),
    }


def score(mdl, X):
    """Score réel du modèle sur X (utilisé pour l'AUC) : decision_function si elle existe, sinon P(y=+1)."""
    if hasattr(mdl, "decision_function"):
        return mdl.decision_function(X)
    return mdl.predict_proba(X)[:, 1]


def metrics(mdl, X, y, minority):
    """Erreur, balanced accuracy, F1 de la classe minoritaire et AUC du modèle déjà appris, sur (X, y).
    L'AUC vaut nan si le modèle ne fournit pas de score."""
    p = mdl.predict(X)
    try:
        auc = roc_auc_score(y, score(mdl, X))
    except Exception:
        auc = np.nan
    return dict(err=np.mean(p != y), bacc=balanced_accuracy_score(y, p),
                f1min=f1_score(y, p, pos_label=minority, zero_division=0), auc=auc)


def run_dataset(name):
    """Fait tourner les 10 méthodes sur les 10 découpages d'un jeu et renvoie un tableau (une ligne par
    méthode et par découpage)."""
    path = os.path.join(RES, "bench", f"{name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    X, y = bk.real_dataset(name)
    minority = 1 if np.mean(y == 1) < 0.5 else -1
    rows = []
    for r, (tr, te) in enumerate(StratifiedShuffleSplit(N_REP, test_size=0.3, random_state=0).split(X, y)):
        for mname, mdl in models(r).items():
            t0 = time.perf_counter(); mdl.fit(X[tr], y[tr]); tf = time.perf_counter() - t0
            rows.append(dict(data=name, rep=r, model=mname, t_fit=tf, **metrics(mdl, X[te], y[te], minority)))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print("done", name, flush=True)
    return df


def dataset_table():
    """Tableau décrivant les jeux : nombre d'exemples m, dimension d, part de la classe minoritaire, déséquilibré ou non."""
    rows = []
    for name in DATASETS:
        X, y = bk.real_dataset(name)
        pm = min(np.mean(y == 1), np.mean(y == -1))
        rows.append(dict(data=name, m=len(y), d=X.shape[1], minority=round(pm, 3), imbalanced=pm < IMBALANCE_THRESHOLD))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    dataset_table().to_csv(os.path.join(RES, "datasets.csv"), index=False)
    order = sorted(DATASETS, key=lambda n: -bk.real_dataset(n)[0].shape[0])
    out = Parallel(n_jobs=2)(delayed(run_dataset)(n) for n in order)
    pd.concat(out).to_csv(os.path.join(RES, "benchmark.csv"), index=False)
    print("benchmark.csv done")