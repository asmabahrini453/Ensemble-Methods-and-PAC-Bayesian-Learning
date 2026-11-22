"""Partie 1 : vérification de nos implémentations contre celles de scikit-learn.

Le sujet demande de montrer que Bagging, AdaBoost et Stacking se comportent
comme leurs équivalents BaggingClassifier, AdaBoostClassifier et StackingClassifier.
Comment je fais? alors, on a : 
Protocole
    - 5 jeux de données : wdbc, pima, australian, spambase, satimage ;
    - pour chacun, 5 découpages stratifiés 70 % train / 30 % test (les mêmes pour tous les modèles) ;
    - pour chaque découpage, on apprend notre modèle et celui de scikit-learn avec les mêmes réglages
      (100 arbres complets pour le bagging, 200 stumps pour AdaBoost, mêmes modèles de base et même
      méta-modèle pour le stacking), puis on mesure l'erreur sur le test ;
    - on moyenne les erreurs sur les 5 découpages.

Ce qui est produit:
    -results/validation.csv : une ligne par jeu, avec l'erreur moyenne de test de chaque modèle
    les colonnes sont: data,bag_ours,bag_sk,bag_oob,ada_ours,ada_sk,stack_ours,stack_sk
    -Affichage console : la part de prédictions identiques entre notre AdaBoost et celui de scikit-learn,
    et la vérification de la borne du cours : erreur d'entraînement d'AdaBoost <= produit des Z_t.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchkit as bk
from ensemble_methods import Bagging, AdaBoost, Stacking, default_base_models

OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
NAMES = ["wdbc", "pima", "australian", "spambase", "satimage"]
N_SPLITS = 5


def main():
    rows, agree, bound_ok = [], [], []
    for name in NAMES:
        X, y = bk.real_dataset(name)
        sss = StratifiedShuffleSplit(N_SPLITS, test_size=0.3, random_state=0)
        R = {k: [] for k in ["bag_ours", "bag_sk", "bag_oob", "ada_ours", "ada_sk", "stack_ours", "stack_sk"]}
        for r, (tr, te) in enumerate(sss.split(X, y)):
            Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
            ours = Bagging(100, random_state=r).fit(Xtr, ytr)
            sk = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=r, n_jobs=-1).fit(Xtr, ytr)
            R["bag_ours"].append(np.mean(ours.predict(Xte) != yte)); R["bag_sk"].append(np.mean(sk.predict(Xte) != yte))
            R["bag_oob"].append(ours.oob_error_)
            a = AdaBoost(200, 1).fit(Xtr, ytr)
            s = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, random_state=r).fit(Xtr, ytr)
            R["ada_ours"].append(np.mean(a.predict(Xte) != yte)); R["ada_sk"].append(np.mean(s.predict(Xte) != yte))
            agree.append((name, np.mean(a.predict(Xte) == s.predict(Xte))))
            bound_ok.append((name, np.mean(a.predict(Xtr) != ytr) <= np.prod(a.Z_) + 1e-12))
            st = Stacking(default_base_models(r), random_state=r).fit(Xtr, ytr)
            ssk = StackingClassifier(default_base_models(r), final_estimator=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                                     cv=5, n_jobs=-1).fit(Xtr, ytr)
            R["stack_ours"].append(np.mean(st.predict(Xte) != yte)); R["stack_sk"].append(np.mean(ssk.predict(Xte) != yte))
        rows.append(dict(data=name, **{k: np.mean(v) for k, v in R.items()}))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(os.path.join(OUT, "validation.csv"), index=False)
    print(df.to_string(index=False))
    ag = pd.DataFrame(agree, columns=["data", "agree"]).groupby("data").agree.mean().round(3)
    print("agreement of AdaBoost predictions (ours vs sklearn):\n", ag.to_string())
    print("training error <= prod Z_t in all runs:", all(b for _, b in bound_ok))


if __name__ == "__main__":
    main()