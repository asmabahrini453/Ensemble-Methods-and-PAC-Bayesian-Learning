
"""
Partie 1 : expériences ciblées pour comprendre QUAND et POURQUOI
chaque méthode d'ensemble aide.

A - Problèmes simulés et bruit d'étiquettes  -> exp_A_sim_noise.csv
B - Erreur selon le nombre de tours T         -> exp_B_rounds.csv
C - Profondeur des arbres de base             -> exp_C_depth.csv
D - Courbes d'apprentissage                   -> exp_D_size.csv
E - Stacking : membres, OOF contre naïf, coût -> exp_E_stacking.csv
F - Bruit d'étiquettes sur jeux réels          -> exp_F_realnoise.csv

Le benchmark (benchmark.py) indique quelle méthode gagne en moyenne.
Ces expériences cherchent à expliquer pourquoi, en faisant varier
un seul facteur à la fois.

Les résultats sont mis en forme par analysis_part1.py.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    cross_val_score
)
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------
# Initialisation
# ---------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import benchkit as bk
from ensemble_methods import Bagging, AdaBoost, Stacking, default_base_models

RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)


# ---------------------------------------------------------
# Jeux de données simulés
# ---------------------------------------------------------

SIM = {
    "checkerboard": lambda m, s: bk.checkerboard(
        m, d=2, k=4, n_noise=2, seed=s
    ),
    "additive": lambda m, s: bk.additive(
        m, d=10, n_inf=5, seed=s
    ),
    "twonorm": lambda m, s: bk.twonorm(
        m, 20, seed=s
    ),
    "threenorm": lambda m, s: bk.threenorm(
        m, 20, seed=s
    ),
    "ringnorm": lambda m, s: bk.ringnorm(
        m, 20, seed=s
    )
}


def err(mdl, X, y):
    """Calcule le taux d'erreur d'un modèle déjà entraîné."""
    return float(np.mean(mdl.predict(X) != y))


# =========================================================
# EXPERIMENT A : Données simulées et bruit d'étiquettes
# =========================================================

def exp_A(m=500, n_test=5000, n_rep=5):

    def one(kind, noise, r):
        """Une répétition sur un problème simulé."""

        X, y = SIM[kind](m, 100 + r)
        Xte, yte = SIM[kind](n_test, 9000 + r)

        # Le bruit touche uniquement les étiquettes du train.
        yn = bk.flip_labels(y, noise, seed=r) if noise > 0 else y

        mods = {
            "tree": DecisionTreeClassifier(random_state=r),
            "bagging": Bagging(100, random_state=r),
            "adaboost_stumps": AdaBoost(200, 1),
            "adaboost_depth3": AdaBoost(200, 3),
            "rf": RandomForestClassifier(200, random_state=r),
            "gb": GradientBoostingClassifier(random_state=r)
        }

        return [
            dict(
                problem=kind,
                noise=noise,
                rep=r,
                model=k,
                err=err(mdl.fit(X, yn), Xte, yte)
            )
            for k, mdl in mods.items()
        ]

    jobs = [
        (k, nz, r)
        for k in SIM
        for nz in (0.0, 0.1, 0.2)
        for r in range(n_rep)
    ]

    out = Parallel(n_jobs=2)(
        delayed(one)(*j) for j in jobs
    )

    results = [row for L in out for row in L]

    pd.DataFrame(results).to_csv(
        os.path.join(RES, "exp_A_sim_noise.csv"),
        index=False
    )


# =========================================================
# EXPERIMENT B : Erreur selon le nombre de tours
# =========================================================

def exp_B(m=500, n_test=5000, n_rep=5, T=500):

    def one(noise, r):

        X, y = bk.twonorm(m, 20, seed=100 + r)
        Xte, yte = bk.twonorm(n_test, 20, seed=9000 + r)

        yn = bk.flip_labels(y, noise, seed=r) if noise > 0 else y

        rows = []

        # AdaBoost : suivi tour par tour.
        for name, depth in [
            ("adaboost_stumps", 1),
            ("adaboost_depth3", 3)
        ]:

            a = AdaBoost(T, depth).fit(X, yn)

            train_stages = list(a.staged_decision_function(X))
            test_stages = list(a.staged_decision_function(Xte))

            tr = [
                np.mean(np.where(s >= 0, 1, -1) != yn)
                for s in train_stages
            ]

            te = [
                np.mean(np.where(s >= 0, 1, -1) != yte)
                for s in test_stages
            ]

            for t in range(T):
                i = min(t, len(te) - 1)

                rows.append({
                    "noise": noise,
                    "rep": r,
                    "model": name,
                    "T": t + 1,
                    "train_err": tr[min(i, len(tr) - 1)],
                    "test_err": te[i]
                })

        # Bagging : cumul des votes arbre par arbre.
        b = Bagging(T, random_state=r).fit(X, yn)

        Vte = b.votes(Xte)
        S = np.cumsum(Vte, axis=0)

        for t in range(T):
            rows.append({
                "noise": noise,
                "rep": r,
                "model": "bagging",
                "T": t + 1,
                "train_err": np.nan,
                "test_err": np.mean(
                    np.where(S[t] >= 0, 1, -1) != yte
                )
            })

        return rows

    out = Parallel(n_jobs=2)(
        delayed(one)(nz, r)
        for nz in (0.0, 0.2)
        for r in range(n_rep)
    )

    results = [row for L in out for row in L]

    pd.DataFrame(results).to_csv(
        os.path.join(RES, "exp_B_rounds.csv"),
        index=False
    )


# =========================================================
# EXPERIMENT C : Influence de la profondeur des arbres
# =========================================================

def exp_C(
    names=("pima", "wdbc", "german", "vehicle", "splice", "spambase"),
    n_rep=5
):

    def one(name, r):

        X, y = bk.real_dataset(name)

        splits = list(
            StratifiedShuffleSplit(
                n_splits=n_rep,
                test_size=0.3,
                random_state=0
            ).split(X, y)
        )

        tr, te = splits[r]
        rows = []

        # Bagging et arbre seul.
        for depth in (1, 2, 3, 5, 8, None):

            mdl = Bagging(
                100,
                max_depth=depth,
                random_state=r
            ).fit(X[tr], y[tr])

            rows.append({
                "data": name,
                "rep": r,
                "method": "bagging",
                "depth": 0 if depth is None else depth,
                "err": err(mdl, X[te], y[te])
            })

            single = DecisionTreeClassifier(
                max_depth=depth,
                random_state=r
            ).fit(X[tr], y[tr])

            rows.append({
                "data": name,
                "rep": r,
                "method": "single tree",
                "depth": 0 if depth is None else depth,
                "err": err(single, X[te], y[te])
            })

        # AdaBoost.
        for depth in (1, 2, 3, 5, 8):

            mdl = AdaBoost(200, depth).fit(X[tr], y[tr])

            rows.append({
                "data": name,
                "rep": r,
                "method": "adaboost",
                "depth": depth,
                "err": err(mdl, X[te], y[te])
            })

        return rows

    out = Parallel(n_jobs=2)(
        delayed(one)(n, r)
        for n in names
        for r in range(n_rep)
    )

    results = [row for L in out for row in L]

    pd.DataFrame(results).to_csv(
        os.path.join(RES, "exp_C_depth.csv"),
        index=False
    )


# =========================================================
# EXPERIMENT D : Courbes d'apprentissage
# =========================================================

def exp_D(
    names=("spambase", "satimage"),
    sizes=(50, 100, 200, 500, 1000, 3000),
    n_rep=5
):

    def one(name, r):

        X, y = bk.real_dataset(name)

        splits = list(
            StratifiedShuffleSplit(
                n_splits=n_rep,
                test_size=0.3,
                random_state=0
            ).split(X, y)
        )

        tr, te = splits[r]

        rng = np.random.RandomState(r)
        rows = []

        for m in sizes:

            if m > len(tr):
                continue

            sub = tr[rng.permutation(len(tr))[:m]]

            if len(np.unique(y[sub])) < 2:
                continue

            mods = {
                "tree": DecisionTreeClassifier(random_state=r),
                "bagging": Bagging(100, random_state=r),
                "adaboost_stumps": AdaBoost(200, 1),
                "adaboost_depth3": AdaBoost(200, 3),
                "rf": RandomForestClassifier(200, random_state=r)
            }

            for k, mdl in mods.items():

                fitted = mdl.fit(X[sub], y[sub])

                rows.append({
                    "data": name,
                    "rep": r,
                    "m": m,
                    "model": k,
                    "err": err(fitted, X[te], y[te])
                })

        return rows

    out = Parallel(n_jobs=2)(
        delayed(one)(n, r)
        for n in names
        for r in range(n_rep)
    )

    results = [row for L in out for row in L]

    pd.DataFrame(results).to_csv(
        os.path.join(RES, "exp_D_size.csv"),
        index=False
    )


# =========================================================
# EXPERIMENT E : Evaluation du stacking
# =========================================================

def exp_E(n_rep=5):

    from benchmark import DATASETS

    def one(name):
        """Compare les différentes stratégies de stacking."""

        X, y = bk.real_dataset(name)
        rows = []

        splits = StratifiedShuffleSplit(
            n_splits=n_rep,
            test_size=0.3,
            random_state=0
        )

        for r, (tr, te) in enumerate(splits.split(X, y)):

            Xtr, ytr = X[tr], y[tr]
            Xte, yte = X[te], y[te]

            base = default_base_models(r)

            e_members = {}
            t_members = 0.0
            cv_members = {}

            # Entraînement et évaluation des modèles de base.
            for k, mdl in base:

                t0 = time.perf_counter()

                f = clone(mdl).fit(Xtr, ytr)

                t_members += time.perf_counter() - t0

                e_members[k] = err(f, Xte, yte)

                cv_members[k] = 1 - cross_val_score(
                    clone(mdl),
                    Xtr,
                    ytr,
                    cv=StratifiedKFold(
                        5,
                        shuffle=True,
                        random_state=r
                    )
                ).mean()

            # Stacking avec prédictions out-of-fold.
            t0 = time.perf_counter()

            st = Stacking(
                base,
                random_state=r
            ).fit(Xtr, ytr)

            t_st = time.perf_counter() - t0

            # Stacking naïf : sans OOF.
            nv = Stacking(
                base,
                oof=False,
                random_state=r
            ).fit(Xtr, ytr)

            # Vote uniforme des modèles de base.
            vote = np.sign(
                np.sum(
                    [
                        np.where(f.predict(Xte) >= 0, 1, -1)
                        for f in st.base_
                    ],
                    axis=0
                )
            )

            vote = np.where(vote == 0, 1, vote)

            # Meilleur modèle selon la validation croisée.
            best_cv = min(cv_members, key=cv_members.get)

            rows.append({
                "data": name,
                "rep": r,
                "stacking": err(st, Xte, yte),
                "stacking_naive": err(nv, Xte, yte),
                "uniform_vote": float(np.mean(vote != yte)),
                "best_member_oracle": min(e_members.values()),
                "best_member_cv": e_members[best_cv],
                "best_member_name": best_cv,
                "mean_member": float(np.mean(list(e_members.values()))),
                "t_members": t_members,
                "t_stacking": t_st,

                **{
                    f"m_{k}": v
                    for k, v in e_members.items()
                },

                **{
                    f"w_{n}": c
                    for n, c in zip(
                        st.names_,
                        st.meta_[-1].coef_[0]
                    )
                }
            })

        return rows

    out = Parallel(n_jobs=2)(
        delayed(one)(n)
        for n in sorted(
            DATASETS,
            key=lambda n: -bk.real_dataset(n)[0].shape[0]
        )
    )

    results = [row for L in out for row in L]

    pd.DataFrame(results).to_csv(
        os.path.join(RES, "exp_E_stacking.csv"),
        index=False
    )


# =========================================================
# EXPERIMENT F : Bruit d'étiquettes sur les jeux réels
# =========================================================

def exp_F(
    names=(
        "wdbc", "pima", "australian", "german",
        "heart", "iono", "spambase", "splice"
    ),
    noises=(0.0, 0.1, 0.2),
    n_rep=5
):

    def one(name, noise, r):
        """Évalue les méthodes avec différents niveaux de bruit."""

        X, y = bk.real_dataset(name)

        splits = list(
            StratifiedShuffleSplit(
                n_splits=10,
                test_size=0.3,
                random_state=0
            ).split(X, y)
        )

        tr, te = splits[r]

        # Le bruit est ajouté uniquement aux étiquettes du train.
        yn = (
            bk.flip_labels(y[tr], noise, seed=r)
            if noise > 0
            else y[tr]
        )

        mods = {
            "tree": DecisionTreeClassifier(random_state=r),
            "bagging": Bagging(100, random_state=r),
            "adaboost_stumps": AdaBoost(200, 1),
            "adaboost_depth3": AdaBoost(200, 3),
            "rf": RandomForestClassifier(200, random_state=r),
            "gb": GradientBoostingClassifier(random_state=r)
        }

        return [
            {
                "data": name,
                "noise": noise,
                "rep": r,
                "model": k,
                "err": err(
                    mdl.fit(X[tr], yn),
                    X[te],
                    y[te]
                )
            }
            for k, mdl in mods.items()
        ]

    out = Parallel(n_jobs=2)(
        delayed(one)(n, nz, r)
        for n in names
        for nz in noises
        for r in range(n_rep)
    )

    results = [row for L in out for row in L]

    pd.DataFrame(results).to_csv(
        os.path.join(RES, "exp_F_realnoise.csv"),
        index=False
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    which = sys.argv[1:] or list("ABCDEF")

    for w in which:

        if w not in "ABCDEF":
            raise ValueError(f"Unknown experiment: {w}")

        t0 = time.time()

        globals()["exp_" + w]()

        print(
            "Block", w,
            "done in", round(time.time() - t0), "s",
            flush=True
        )
