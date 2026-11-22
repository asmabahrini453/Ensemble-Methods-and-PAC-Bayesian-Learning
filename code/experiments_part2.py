"""Partie 2 : expériences sur les bornes PAC-Bayésiennes (utilise pacbayes_tools.py).

    Q1  comportement numérique des bornes (sans données)                 -> results/p2_q1_*.csv
    Q2  borne A contre borne B sur des stumps, plusieurs jeux et graines -> results/p2_q2_compare.csv, p2_q2_lambda.csv
    Q3  prior uniforme ou nul contre prior appris                        -> results/p2_q3_prior.csv
    Q4  stumps (loi catégorielle) contre classifieurs linéaires (loi gaussienne), autres posteriors -> results/p2_q4_family.csv
    Q5  apprentissage classique contre minimisation de la borne          -> results/p2_q5_classical.csv

le Vocabulaire utilisé dans les fichiers de résultats :
    voteurs        les petits classifieurs de base : des stumps (une coupure sur une variable) ou des classifieurs linéaires
    P (prior)      poids des voteurs choisis AVANT de voir l'échantillon S de la borne
    Q (posterior)  poids appris à partir de S ; le classifieur de Gibbs tire un voteur selon Q
    KL             KL(Q||P) : de combien Q s'éloigne de P ; c'est la complexité qui entre dans les bornes
    bornes         A (Catoni, avec un paramètre lambda pris dans la grille 2^0..2^14), B (McAllester),
                   Seeger (la plus serrée, utilisée comme certificat), toutes au niveau delta = 0,05
    vote / Gibbs   risque du vote majoritaire pondéré par Q, et risque moyen de Gibbs ; vote <= 2 x Gibbs

Découpage de chaque jeu (pb.split) : 30 % de test ; les 70 % restants se coupent en 20 % pour apprendre le
prior (soit 14 % du total) et 80 % pour l'échantillon S de la borne (soit 56 % du total). Le test ne sert qu'à
vérifier que les bornes sont valides ; il n'entre dans aucun calcul de borne.

Justification de la démarche
    - Q1 sans données : on fixe un risque empirique (0,10) et on fait varier m, KL et lambda pour voir comment
      les trois bornes réagissent, indépendamment de tout jeu de données.
    - Q2 à Q5 sur des jeux réels, avec 10 graines (donc 10 découpages différents) par jeu : un seul découpage
      pourrait donner une borne favorable par hasard.
    - Le prior est toujours construit sans utiliser S (prior uniforme, nul, ou appris sur la partie à 14 %),
      sinon la borne ne serait plus valide.
    - Les résultats de chaque jeu et de chaque graine sont calculés en parallèle sur 2 coeurs.

"""
import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import benchkit as bk
import pacbayes_tools as pb
from ensemble_methods import Stacking

RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
SEEDS = range(10)
DATA_Q2 = ["wdbc", "australian", "pima", "heart", "iono", "sonar", "german"]
DATA_Q5 = ["wdbc", "pima", "australian", "german", "heart", "iono", "spambase", "splice"]


"""Bloc Q1 : les bornes en fonction de m, de KL et de lambda, pour un risque empirique de 0,10. Trois fichiers :
        p2_q1_vs_m.csv       m de 50 à 100 000, KL dans {1, 5, 20, 50}. Colonnes : borne A avec lambda dans la grille
                             (A_grid), borne A avec le lambda optimal continu (A_cont, même terme d'union),
                             borne A avec un seul lambda fixé sans union sur la grille (A_single_lambda),
                             borne B (McAllester) et Seeger.
        p2_q1_vs_KL.csv      m dans {200, 1000, 5000}, KL de 0 à 200 : bornes A (grille), B et Seeger.
        p2_q1_vs_lambda.csv  m = 1000, KL dans {1, 10, 50, 200}, lambda de 1 à 100 000 : borne A en fonction de lambda.
    Le lambda optimal continu vaut sqrt(8 m (KL + c)), avec c = ln(|grille| / delta)."""
def q1():
    
    emp, rows = 0.10, []
    for KL in (1, 5, 20, 50):
        for m in np.unique(np.logspace(np.log10(50), 5, 40).astype(int)):
            A = min(pb.bound_linear(emp, KL, m, lam) for lam in pb.GRID)
            c = np.log(len(pb.GRID) / pb.DELTA)
            lam_star = np.sqrt(8 * m * (KL + c))
            A_cont = pb.bound_linear(emp, KL, m, lam_star)
            A_nogrid = pb.bound_linear(emp, KL, m, np.sqrt(8 * m * (KL + np.log(1 / pb.DELTA))), n_grid=1)
            rows.append(dict(KL=KL, m=m, A_grid=A, A_cont=A_cont, A_single_lambda=A_nogrid,
                             B=pb.bound_mcallester(emp, KL, m), Seeger=pb.bound_seeger(emp, KL, m), lam_star=lam_star))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "p2_q1_vs_m.csv"), index=False)
    rows = []
    for m in (200, 1000, 5000):
        for KL in np.linspace(0, 200, 81):
            A = min(pb.bound_linear(emp, KL, m, lam) for lam in pb.GRID)
            rows.append(dict(m=m, KL=KL, A_grid=A, B=pb.bound_mcallester(emp, KL, m), Seeger=pb.bound_seeger(emp, KL, m)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "p2_q1_vs_KL.csv"), index=False)
    rows = []
    for m, KL in ((1000, 1), (1000, 10), (1000, 50), (1000, 200)):
        for lam in np.logspace(0, 5, 200):
            rows.append(dict(m=m, KL=KL, lam=lam, A=pb.bound_linear(emp, KL, m, lam)))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "p2_q1_vs_lambda.csv"), index=False)

"""Prépare une expérience sur des stumps. Découpe le jeu (pb.split), construit les stumps à partir de la
    partie prior seulement, calcule les votes de chaque stump sur S et sur le test, et le prior P :
    uniforme, ou appris (posterior de Gibbs sur la partie prior). Renvoie les stumps H, les votes sur S (V) et
    sur le test (Vte), les étiquettes de S et du test, et P."""
def cat_setup(X, y, seed, prior="uniform"):
    Xp, yp, Xs, ys, Xte, yte = pb.split(X, y, seed)
    H = pb.make_stumps(Xp)
    V, Vte = pb.stump_votes(H, Xs), pb.stump_votes(H, Xte)
    P = np.full(len(H), 1 / len(H))
    if prior == "learned":
        P = np.maximum(pb.gibbs_posterior(P, pb.voter_risks(pb.stump_votes(H, Xp), yp), len(yp) / 4), 1e-300)
        P /= P.sum()
    return H, V, Vte, ys, yte, P

"""Tous les chiffres d'un posterior catégoriel Q : risque empirique de Gibbs sur S (emp), KL(Q||P), nombre
    effectif de voteurs (neff = 1 / somme des Q^2, 1 si Q est concentré sur un voteur), les trois bornes
    (seeger, mcallester, catoni_grid), et les risques de Gibbs et du vote sur le test et du vote sur S."""
def describe(Q, P, V, Vte, ys, yte):
    L = pb.voter_risks(V, ys)
    emp, KL, m = float(Q @ L), pb.kl_categorical(Q, P), len(ys)
    return dict(emp=emp, KL=KL, neff=1 / np.sum(Q ** 2), seeger=pb.bound_seeger(emp, KL, m), mcallester=pb.bound_mcallester(emp, KL, m),
                catoni_grid=min(pb.bound_linear(emp, KL, m, lam) for lam in pb.GRID),
                test_gibbs=pb.gibbs_risk(Q, Vte, yte), test_vote=pb.vote_risk(Q, Vte, yte), train_vote=pb.vote_risk(Q, V, ys))


"""Bloc Q2 : borne A contre borne B sur des stumps, avec un prior uniforme, sur 7 jeux et 10 graines.
    Pour la borne A, on apprend le posterior de Gibbs pour chaque lambda de la grille et on garde celui qui a la
    plus petite borne ; pour la borne B, on minimise directement. Écrit p2_q2_compare.csv (le meilleur de chaque
    borne) et p2_q2_lambda.csv (une ligne par lambda de la grille, pour voir l'effet du choix de lambda)."""
def q2():
    def one(name, seed):
        """Un jeu et une graine : renvoie les lignes de comparaison et les lignes par lambda."""
        X, y = bk.real_dataset(name)
        out, lam_rows = [], []
        for prior in ("uniform",):
            H, V, Vte, ys, yte, P = cat_setup(X, y, seed, prior)
            L, m = pb.voter_risks(V, ys), len(ys)
            best, rows = pb.learn_linear_categorical(P, L, m)
            B = pb.learn_mcallester_categorical(P, L, m)
            for lab, r in (("A (linear)", best), ("B (McAllester)", B)):
                out.append(dict(data=name, seed=seed, bound_name=lab, bound=r["bound"], lam=r.get("lam", np.nan),
                                **describe(r["Q"], P, V, Vte, ys, yte)))
            for r in rows:
                lam_rows.append(dict(data=name, seed=seed, lam=r["lam"], bound_A=r["bound"], **describe(r["Q"], P, V, Vte, ys, yte)))
        return out, lam_rows
    res = Parallel(n_jobs=2)(delayed(one)(n, s) for n in DATA_Q2 for s in SEEDS)
    pd.DataFrame([r for o, _ in res for r in o]).to_csv(os.path.join(RES, "p2_q2_compare.csv"), index=False)
    pd.DataFrame([r for _, l in res for r in l]).to_csv(os.path.join(RES, "p2_q2_lambda.csv"), index=False)

"""Bloc Q3 : prior uniforme ou nul contre prior appris. Pour la famille des stumps : prior uniforme, puis prior
    appris sur la partie à 14 % ; pour les classifieurs linéaires gaussiens : prior nul, puis prior dont la moyenne est
    apprise sur la partie à 14 %. Dans les deux cas, bornes A et B. Écrit p2_q3_prior.csv."""

def q3():
    def one(name, seed):
        X, y = bk.real_dataset(name)
        rows = []
        for prior in ("uniform", "learned"):
            H, V, Vte, ys, yte, P = cat_setup(X, y, seed, prior)
            L, m = pb.voter_risks(V, ys), len(ys)
            best, _ = pb.learn_linear_categorical(P, L, m)
            B = pb.learn_mcallester_categorical(P, L, m)
            for lab, r in (("A (linear)", best), ("B (McAllester)", B)):
                rows.append(dict(data=name, seed=seed, family="stumps", prior=prior, bound_name=lab, bound=r["bound"],
                                 **describe(r["Q"], P, V, Vte, ys, yte)))
        for prior in ("zero", "learned"):
            o = pb.run_gaussian(X, y, seed=seed, prior=prior)
            for lab, r in o.items():
                rows.append(dict(data=name, seed=seed, family="linear", prior=prior, bound_name=lab, bound=r["bound"], emp=r["emp"],
                                 KL=r["KL"], seeger=r["seeger"], test_gibbs=r["test_gibbs"], test_vote=r["test_vote"]))
        return rows
    res = Parallel(n_jobs=2)(delayed(one)(n, s) for n in DATA_Q2 for s in SEEDS)
    pd.DataFrame([r for o in res for r in o]).to_csv(os.path.join(RES, "p2_q3_prior.csv"), index=False)


"""Bloc Q4 : stumps (loi catégorielle) contre classifieurs linéaires (loi gaussienne). Pour les stumps, en plus des
    posteriors des bornes A et B, des posteriors de comparaison construits à la main : Dirac sur le meilleur stump,
    uniforme sur les 5, 20 ou 100 meilleurs, uniforme sur tous. Ils montrent le compromis entre risque empirique
    (faible si Q est concentré) et KL (grande si Q est concentré). Pour la famille linéaire, prior nul. Écrit p2_q4_family.csv."""
def q4():
 
    def one(name, seed):
        X, y = bk.real_dataset(name)
        rows = []
        H, V, Vte, ys, yte, P = cat_setup(X, y, seed, "uniform")
        L, m = pb.voter_risks(V, ys), len(ys)
        order = np.argsort(L)
        best, _ = pb.learn_linear_categorical(P, L, m)
        B = pb.learn_mcallester_categorical(P, L, m)
        posts = {"stumps: bound A": best["Q"], "stumps: bound B": B["Q"]}
        q = np.zeros(len(H)); q[order[0]] = 1; posts["stumps: Dirac on best stump"] = q
        for k in (5, 20, 100):
            q = np.zeros(len(H)); q[order[:k]] = 1 / k; posts[f"stumps: uniform on {k} best"] = q
        posts["stumps: uniform on all"] = P
        for lab, Q in posts.items():
            rows.append(dict(data=name, seed=seed, posterior=lab, **describe(Q, P, V, Vte, ys, yte)))
        for prior in ("zero",):
            o = pb.run_gaussian(X, y, seed=seed, prior=prior)
            for lab, r in o.items():
                rows.append(dict(data=name, seed=seed, posterior=f"linear: bound {lab[0]}", emp=r["emp"], KL=r["KL"], seeger=r["seeger"],
                                 test_gibbs=r["test_gibbs"], test_vote=r["test_vote"]))
        return rows
    res = Parallel(n_jobs=2)(delayed(one)(n, s) for n in DATA_Q2 for s in SEEDS)
    pd.DataFrame([r for o in res for r in o]).to_csv(os.path.join(RES, "p2_q4_family.csv"), index=False)


"""Bloc Q5 : apprentissage classique contre minimisation de la borne, sur 8 jeux et 10 graines. Les méthodes classiques
    (AdaBoost avec 100 stumps, forêt de 200 arbres, HistGradientBoosting, notre stacking, régression logistique) sont
    apprises sur tous les 70 % d'entraînement et n'ont pas de certificat. Les méthodes PAC-Bayes (stumps avec bornes A
    et B ; linéaire gaussien avec prior nul ou appris) sont apprises sur S et donnent un certificat : la borne de
    Seeger. On rapporte l'erreur de test du vote et celle de Gibbs. Écrit p2_q5_classical.csv."""
def q5():
    
    def one(name, seed):
        X, y = bk.real_dataset(name)
        Xp, yp, Xs, ys, Xte, yte = pb.split(X, y, seed)
        Xtr, ytr = np.vstack([Xp, Xs]), np.concatenate([yp, ys])
        e = lambda mdl: float(np.mean(mdl.predict(Xte) != yte))
        rows = []
        classical = {"AdaBoost (stumps, T=100)": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=seed),
                     "random forest (200)": RandomForestClassifier(200, random_state=seed),
                     "gradient boosting (hist)": HistGradientBoostingClassifier(random_state=seed),
                     "stacking (ours)": Stacking(random_state=seed),
                     "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
        for lab, mdl in classical.items():
            rows.append(dict(data=name, seed=seed, approach="classical", method=lab, test_err=e(mdl.fit(Xtr, ytr)), certificate=np.nan))
        H, V, Vte, ys_, yte_, P = cat_setup(X, y, seed, "uniform")
        L, m = pb.voter_risks(V, ys_), len(ys_)
        best, _ = pb.learn_linear_categorical(P, L, m); B = pb.learn_mcallester_categorical(P, L, m)
        for lab, r in (("stumps, bound A", best), ("stumps, bound B", B)):
            d = describe(r["Q"], P, V, Vte, ys_, yte_)
            rows.append(dict(data=name, seed=seed, approach="PAC-Bayes", method=lab + " (vote)", test_err=d["test_vote"], certificate=d["seeger"]))
            rows.append(dict(data=name, seed=seed, approach="PAC-Bayes", method=lab + " (Gibbs)", test_err=d["test_gibbs"], certificate=d["seeger"]))
        for prior in ("zero", "learned"):
            o = pb.run_gaussian(X, y, seed=seed, prior=prior)
            for lab, r in o.items():
                rows.append(dict(data=name, seed=seed, approach="PAC-Bayes", method=f"linear, bound {lab[0]}, prior {prior} (vote)",
                                 test_err=r["test_vote"], certificate=r["seeger"]))
                rows.append(dict(data=name, seed=seed, approach="PAC-Bayes", method=f"linear, bound {lab[0]}, prior {prior} (Gibbs)",
                                 test_err=r["test_gibbs"], certificate=r["seeger"]))
        return rows
    res = Parallel(n_jobs=2)(delayed(one)(n, s) for n in DATA_Q5 for s in SEEDS)
    pd.DataFrame([r for o in res for r in o]).to_csv(os.path.join(RES, "p2_q5_classical.csv"), index=False)


if __name__ == "__main__":
    import time
    for w in (sys.argv[1:] or ["Q1", "Q2", "Q3", "Q4", "Q5"]):
        t0 = time.time(); globals()[w.lower()](); print(w, "done in", round(time.time() - t0), "s", flush=True)