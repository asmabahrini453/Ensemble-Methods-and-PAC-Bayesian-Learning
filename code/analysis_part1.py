"""Partie 1 : tableaux et figures à partir des résultats des expériences.

Entrées : results/benchmark.csv (ou les fichiers de results/bench/), results/datasets.csv et results/exp_*.csv.
Aucune expérience n'est refaite ici : le script lit les CSV et met en forme.

Outils statistiques : les jeux de données sont l'unité d'analyse (un score moyen par jeu et par méthode).
On classe les méthodes sur chaque jeu, on moyenne les rangs, puis on applique le test de Friedman et la
distance critique de Nemenyi (Demsar, 2006) ; les comparaisons par paires utilisent un test de Wilcoxon
sur ces moyennes, corrigé par la méthode de Holm.

Justification de la démarche
    - Un score par jeu et par méthode : les 10 découpages d'un même jeu ne sont pas indépendants entre eux,
      donc on les moyenne et on compare les méthodes sur les 21 jeux, qui sont les vraies unités.
    - Friedman puis Nemenyi : on ne suppose pas que les erreurs suivent une loi normale, on ne compare que des
      rangs, ce qui permet de mélanger des jeux dont les erreurs sont d'échelles très différentes.
    - Wilcoxon-Holm pour les paires : Nemenyi compare toutes les méthodes entre elles ; on y ajoute 17 paires
      ciblées, qui correspondent aux questions du sujet (chaque ensemble contre l'arbre, contre la régression
      logistique, bagging contre boosting, stacking contre ses concurrents). Holm corrige le risque de
      conclure à tort quand on fait plusieurs tests.
    - Quatre métriques : l'erreur ne suffit pas quand une classe est rare. Balanced accuracy, F1 de la classe
      minoritaire et AUC sont rapportées, et les rangs sont calculés séparément pour tous les jeux, les jeux
      équilibrés et les jeux déséquilibrés (classe minoritaire < 20 %).
    - Deux diagrammes de différence critique seulement : l'erreur sur tous les jeux, et la balanced accuracy
      sur les jeux déséquilibrés, car ce sont les deux cas où la conclusion est la plus utile ; les autres
      combinaisons sont dans ranks_summary.csv.
    - Les expériences ciblées A à F sont tracées à part : elles répondent chacune à une question (bruit, nombre
      de tours, profondeur, taille de l'échantillon, stacking), sans test d'hypothèse global.

Ordre des analyses (fonction main)
    1. Tableau des jeux de données.
    2. Tableaux de la métrique moyenne par jeu et par méthode (annexe), pour les 4 métriques.
    3. Rangs moyens, test de Friedman et distance critique, pour chaque métrique et chaque groupe de jeux
       (tous, équilibrés, déséquilibrés) ; tableau des rangs.
    4. Diagrammes de différence critique : erreur sur tous les jeux, balanced accuracy sur les déséquilibrés.
    5. Tests par paires (Wilcoxon-Holm) et gains par rapport à l'arbre et à la régression logistique.
    6. Bagging contre boosting selon la taille, la dimension et le déséquilibre.
    7. Jeux déséquilibrés : le classement des méthodes change avec la métrique ; précision contre balanced accuracy.
    8. Temps d'entraînement moyen par méthode.
    9. Expériences ciblées, si leurs CSV existent : A (problèmes simulés et bruit), B (erreur selon le nombre de
       tours), C (profondeur), D (taille de l'échantillon), E (stacking), F (bruit sur jeux réels).
    10. Écriture de results/part1_numbers.json.
"""

import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
TAB = os.path.join(RES, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200, "figure.dpi": 130})
LABEL = {"tree": "tree", "logreg": "log. reg.", "7nn": "7-NN", "bagging": "bagging", "adaboost_stumps": "AdaBoost (stumps)",
         "adaboost_depth3": "AdaBoost (depth 3)", "stacking": "stacking", "rf": "random forest", "gb": "grad. boosting",
         "hgb": "hist. grad. boosting"}
ORDER = list(LABEL)
COL = {"tree": "#8c8c8c", "logreg": "#6c4f9c", "7nn": "#9a7bd1", "bagging": "#1f5f99", "adaboost_stumps": "#c8553d",
       "adaboost_depth3": "#e9a13b", "stacking": "#2a9d8f", "rf": "#4f8fc4", "gb": "#a63d2b", "hgb": "#d9792b"}
METRICS = {"err": "error", "bacc": "balanced accuracy", "f1min": "F1 (minority)", "auc": "AUC"}
HIGHER = {"err": False, "bacc": True, "f1min": True, "auc": True}
NUM = {}

# Valeurs critiques q(alpha=0,05) de l'étendue studentisée divisées par racine de 2 (Demsar 2006, tableau 5)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def load_bench():
    """Charge benchmark.csv"""
    p = os.path.join(RES, "benchmark.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "bench", "*.csv"))])


def esc(s):
    import re
    return "".join(p if p.startswith("$") else p for p in re.split(r"(\$[^$]*\$)", s))

#latex
def tex(df, name, **kw):
    s = df.to_latex(escape=False, **kw)
    open(os.path.join(TAB, name + ".tex"), "w").write(s)


def ranks_table(M, higher):
    R = M.rank(axis=1, ascending=not higher, method="average")
    return R

"""Test de Friedman sur la matrice M. Renvoie les rangs moyens des méthodes, la statistique et la p-valeur."""
def friedman(M, higher):
    R = ranks_table(M, higher)
    stat, p = stats.friedmanchisquare(*[M[c].values for c in M.columns])
    return R.mean(), stat, p

"""Distance critique de Nemenyi pour k méthodes et N jeux : deux méthodes dont les rangs moyens diffèrent
    de moins que cette valeur ne sont pas significativement différentes (niveau 5 %)."""
def cd_nemenyi(k, N):
    return Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N))

"""Diagramme de différence critique de Demsar. Les méthodes sont placées selon leur rang moyen sur un axe ;
    celles reliées par une même barre épaisse ne sont pas significativement différentes. `avg` : rang moyen par
    méthode, `cd` : distance critique."""
def cd_diagram(ax, avg, cd, title=""):
    
    names = sorted(avg, key=avg.get)
    k = len(names); lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3); ax.set_ylim(0.4 - 0.15 * ((k + 1) // 2) - 0.05, 1.0); ax.axis("off")
    ax.plot([lo, hi], [0.55, 0.55], "k", lw=0.8)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0.55, 0.6], "k", lw=0.8); ax.text(t, 0.65, str(t), ha="center", va="bottom", fontsize=7)
    half = (k + 1) // 2
    for i, n in enumerate(names):
        r = avg[n]
        if i < half:
            y = 0.4 - 0.15 * i
            ax.plot([r, r, lo - 0.1], [0.55, y, y], color="0.3", lw=0.5)
            ax.text(lo - 0.15, y, f"{LABEL[n]} ({r:.2f})", ha="right", va="center", fontsize=7)
        else:
            y = 0.4 - 0.15 * (k - 1 - i)
            ax.plot([r, r, hi + 0.1], [0.55, y, y], color="0.3", lw=0.5)
            ax.text(hi + 0.15, y, f"({r:.2f}) {LABEL[n]}", ha="left", va="center", fontsize=7)
    cliques, last = [], -1
    for i in range(k):
        j = i
        while j + 1 < k and avg[names[j + 1]] - avg[names[i]] < cd:
            j += 1
        if j > i and j > last:
            cliques.append((i, j)); last = j
    for c, (i, j) in enumerate(cliques):
        y = 0.5 - 0.035 * c
        ax.plot([avg[names[i]] - 0.03, avg[names[j]] + 0.03], [y, y], "k", lw=2.2, solid_capstyle="butt")
    ax.plot([lo, lo + cd], [0.85, 0.85], "k", lw=1.2); ax.text(lo + cd / 2, 0.9, f"CD = {cd:.2f}", ha="center", fontsize=7)
    ax.set_title(title, fontsize=8, loc="center", y=0.99)

"""Matrice jeux x méthodes de la métrique moyennée sur les découpages ; subset = liste de jeux à garder."""
def per_dataset(B, metric, subset=None):
    M = B.groupby(["data", "model"])[metric].mean().unstack()[ORDER]
    return M if subset is None else M.loc[subset]

"""Test de Wilcoxon des rangs signés (bilatéral) sur les moyennes par jeu, pour chaque paire (a, b) de `pairs`,
    avec correction de Holm pour les comparaisons multiples. Renvoie, par paire : différence moyenne, nombre de
    jeux gagnés et perdus par a, p-valeur brute et corrigée."""
def holm_wilcoxon(M, higher, pairs):
    
    rows = []
    for a, b in pairs:
        d = (M[a] - M[b]).values
        try:
            p = stats.wilcoxon(d, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = 1.0
        wins = int(np.sum(d > 0)) if higher else int(np.sum(d < 0))
        losses = int(np.sum(d < 0)) if higher else int(np.sum(d > 0))
        rows.append(dict(a=a, b=b, mean_diff=d.mean(), wins_a=wins, losses_a=losses, p=p))
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    m = len(df)
    df["p_holm"] = np.minimum(1, np.maximum.accumulate([(m - i) * p for i, p in enumerate(df.p)]))
    return df


def main():
    B = load_bench()
    D = pd.read_csv(os.path.join(RES, "datasets.csv")).set_index("data")
    have = sorted(B.data.unique())
    D = D.loc[[d for d in D.index if d in have]]
    imb = list(D.index[D.imbalanced]); bal = list(D.index[~D.imbalanced])
    NUM["n_datasets"] = len(D); NUM["imbalanced"] = imb

    t = D.copy(); t["imbalanced"] = np.where(t.imbalanced, "yes", "")
    t = t.reset_index().rename(columns={"data": "dataset", "minority": "minority prop."})
    t["m"] = t.m.astype(int).astype(str); t["d"] = t.d.astype(int).astype(str); t["minority prop."] = t["minority prop."].map(lambda v: f"{v:.3f}")
    half = (len(t) + 1) // 2
    left, right = t.iloc[:half].reset_index(drop=True), t.iloc[half:].reset_index(drop=True)
    right = right.reindex(range(half)).fillna("")
    both = pd.concat([left, right], axis=1)
    open(os.path.join(TAB, "datasets.tex"), "w").write(esc(both.to_latex(index=False, escape=False, float_format="%.3f",
         column_format="lrrrc|lrrrc", header=["dataset", "$m$", "$d$", "min.", "imb."] * 2)))

    for met in METRICS:
        M = per_dataset(B, met)
        T = M.copy(); T.columns = [LABEL[c] for c in T.columns]
        best = M.max(axis=1) if HIGHER[met] else M.min(axis=1)
        cells = T.copy().astype(object)
        for d in M.index:
            for c in M.columns:
                s = f"{M.loc[d, c]:.3f}"
                cells.loc[d, LABEL[c]] = (r"\textbf{" + s + "}") if abs(M.loc[d, c] - best[d]) < 5e-4 else s
        cells.index.name = None
        open(os.path.join(TAB, f"mean_{met}.tex"), "w").write(esc(
            cells.to_latex(escape=False, column_format="l" + "c" * len(cells.columns))))

    rk, summary = {}, []
    for met in METRICS:
        for grp, subset in (("all", None), ("balanced", bal), ("imbalanced", imb)):
            M = per_dataset(B, met, subset)
            if len(M) < 3:
                continue
            avg, stat, p = friedman(M, HIGHER[met])
            rk[(met, grp)] = avg
            summary.append(dict(metric=met, group=grp, N=len(M), friedman=stat, p=p, CD=cd_nemenyi(len(ORDER), len(M)),
                                **{LABEL[k]: avg[k] for k in ORDER}))
    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(RES, "ranks_summary.csv"), index=False)
    cols = [LABEL[k] for k in ORDER]
    sel = S[S.group != "balanced"]
    short = {"err": "error", "bacc": "bal.\\ acc.", "f1min": "F1 min.", "auc": "AUC"}
    head = [f"\\shortstack{{{short[m]}\\\\{'all' if g == 'all' else 'imb.'} ($N={n}$)}}" for m, g, n in zip(sel.metric, sel.group, sel.N)]
    T = pd.DataFrame({h: [f"{r[c]:.2f}" for c in cols] for h, (_, r) in zip(head, sel.iterrows())}, index=cols)
    T.loc["Friedman $p$"] = [f"{p:.0e}" for p in sel.p]
    T.index.name = None
    open(os.path.join(TAB, "ranks.tex"), "w").write(esc(T.to_latex(escape=False, column_format="l" + "c" * len(T.columns))))
    NUM["ranks"] = {f"{m}|{g}": {LABEL[k]: round(float(v[k]), 2) for k in ORDER} for (m, g), v in rk.items()}

    panels = [("err", "all", None), ("bacc", "imbalanced", imb)]
    fig, axes = plt.subplots(len(panels), 1, figsize=(6.6, 2.15 * len(panels)))
    for ax, (met, grp, sub) in zip(axes, panels):
        M = per_dataset(B, met, sub)
        avg = rk[(met, grp)]
        cd_diagram(ax, {k: float(avg[k]) for k in ORDER}, cd_nemenyi(len(ORDER), len(M)), f"{METRICS[met]}, {grp} datasets (N={len(M)})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p1_cd_diagrams.png")); plt.close(fig)

    pairs = [("bagging", "tree"), ("adaboost_stumps", "tree"), ("adaboost_depth3", "tree"), ("rf", "tree"), ("gb", "tree"), ("hgb", "tree"),
             ("stacking", "tree"), ("rf", "logreg"), ("hgb", "logreg"), ("stacking", "logreg"), ("rf", "bagging"), ("adaboost_stumps", "bagging"),
             ("adaboost_depth3", "bagging"), ("hgb", "rf"), ("stacking", "rf"), ("stacking", "hgb"), ("gb", "adaboost_stumps")]
    M = per_dataset(B, "err"); H = holm_wilcoxon(M, False, pairs)
    H["a"] = H.a.map(LABEL); H["b"] = H.b.map(LABEL)
    H.to_csv(os.path.join(RES, "pairwise_err.csv"), index=False)
    rel = {}
    for k in ORDER:
        rel[LABEL[k]] = {"vs_tree_abs": float((M["tree"] - M[k]).mean()), "vs_tree_rel_median": float(((M["tree"] - M[k]) / M["tree"]).median()),
                         "vs_logreg_abs": float((M["logreg"] - M[k]).mean()), "vs_best_other": float((np.minimum(M["logreg"], M["7nn"]) - M[k]).mean()),
                         "mean_err": float(M[k].mean())}
    NUM["gains"] = rel
    NUM["pairwise_err"] = H.round(4).to_dict("records")
    Ht = H.copy(); Ht["mean_diff"] = Ht.mean_diff.map(lambda v: f"{v:+.3f}"); fmt = lambda v: "$<0.001$" if v < 0.001 else f"{v:.3f}"; Ht["p"] = Ht.p.map(fmt); Ht["p_holm"] = Ht.p_holm.map(fmt)
    Ht = Ht.rename(columns={"a": "A", "b": "B", "mean_diff": "mean err(A)$-$err(B)", "wins_a": "A better", "losses_a": "B better", "p": "$p$", "p_holm": "$p$ (Holm)"})
    open(os.path.join(TAB, "pairwise.tex"), "w").write(esc(Ht.to_latex(index=False, escape=False, column_format="llcccrr")))

    E = per_dataset(B, "err")
    dd = pd.DataFrame({"m": D.m, "d": D.d, "minority": D.minority, "bag": E.bagging, "ada1": E.adaboost_stumps, "ada3": E.adaboost_depth3,
                       "rf": E.rf, "gb": E.gb}).dropna()
    dd["bag_minus_ada1"] = dd.bag - dd.ada1; dd["bag_minus_ada3"] = dd.bag - dd.ada3
    dd.to_csv(os.path.join(RES, "bag_vs_boost.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    for ax, (xcol, xl, lg) in zip(axes, [("m", "number of examples $m$", True), ("d", "dimension $d$", True), ("minority", "minority proportion", False)]):
        for col, lab, c in (("bag_minus_ada1", "bagging $-$ AdaBoost stumps", COL["adaboost_stumps"]), ("bag_minus_ada3", "bagging $-$ AdaBoost depth 3", COL["adaboost_depth3"])):
            ax.scatter(dd[xcol], dd[col], s=14, color=c, label=lab, alpha=0.85)
        ax.axhline(0, color="k", lw=0.6)
        if lg:
            ax.set_xscale("log")
            ticks = [200, 1000, 5000] if xcol == "m" else [5, 20, 80]
            ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
        ax.set_xlabel(xl)
    axes[0].set_ylabel("test error: bagging $-$ AdaBoost")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p1_bag_vs_boost.png")); plt.close(fig)
    NUM["bag_vs_boost"] = {"spearman_m_stumps": [float(x) for x in stats.spearmanr(np.log(dd.m), dd.bag_minus_ada1)],
                           "spearman_d_stumps": [float(x) for x in stats.spearmanr(np.log(dd.d), dd.bag_minus_ada1)],
                           "spearman_m_d3": [float(x) for x in stats.spearmanr(np.log(dd.m), dd.bag_minus_ada3)],
                           "bag_beats_ada1": int((dd.bag_minus_ada1 < 0).sum()), "ada1_beats_bag": int((dd.bag_minus_ada1 > 0).sum()),
                           "bag_beats_ada3": int((dd.bag_minus_ada3 < 0).sum()), "ada3_beats_bag": int((dd.bag_minus_ada3 > 0).sum()),
                           "bag_wins": list(dd.index[dd.bag_minus_ada1 < 0]), "ada1_wins": list(dd.index[dd.bag_minus_ada1 > 0])}

    if len(imb) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.6, 1]})
        ax = axes[0]
        ranks = pd.DataFrame({METRICS[m]: rk[(m, "imbalanced")] for m in METRICS})
        def spread(vals, gap=0.42):
            """label positions: the sorted values pushed apart by at least `gap` (no overlapping labels)."""
            idx = np.argsort(vals); pos = np.array(vals, float)[idx].copy()
            for i in range(1, len(pos)):
                pos[i] = max(pos[i], pos[i - 1] + gap)
            pos -= (pos.mean() - np.mean(vals))
            out = np.empty(len(vals)); out[idx] = pos
            return out
        yl = spread([ranks.loc[k].values[0] for k in ORDER]); yr = spread([ranks.loc[k].values[-1] for k in ORDER])
        for i, k in enumerate(ORDER):
            ax.plot(range(4), ranks.loc[k].values, "o-", ms=3, lw=1, color=COL[k])
            ax.text(3.12, yr[i], LABEL[k], va="center", fontsize=6.5, color=COL[k])
            ax.text(-0.12, yl[i], LABEL[k], va="center", ha="right", fontsize=6.5, color=COL[k])
        ax.set_xticks(range(4)); ax.set_xticklabels(["error", "balanced\naccuracy", "F1\n(minority)", "AUC"]); ax.invert_yaxis()
        ax.set_xlim(-1.2, 4.2); ax.set_ylabel("average rank (1 = best)"); ax.set_title(f"imbalanced datasets (N={len(imb)})", fontsize=8)
        ax = axes[1]
        triv = D.loc[imb, "minority"].mean()
        e = per_dataset(B, "err", imb).mean(); b = per_dataset(B, "bacc", imb).mean(); f = per_dataset(B, "f1min", imb).mean()
        w = 0.27
        xs = np.arange(len(ORDER))
        ax.barh(xs - w, [1 - e[k] for k in ORDER], w, color="#8c8c8c", label="accuracy")
        ax.barh(xs, [b[k] for k in ORDER], w, color="#1f5f99", label="balanced acc.")
        ax.barh(xs + w, [f[k] for k in ORDER], w, color="#c8553d", label="F1 minority")
        ax.axvline(1 - triv, color="k", ls=":", lw=0.8); ax.text(1 - triv, -0.9, "always\nmajority", fontsize=6, ha="center")
        ax.set_yticks(xs); ax.set_yticklabels([LABEL[k] for k in ORDER], fontsize=6.5); ax.invert_yaxis(); ax.set_xlim(0.3, 1.0)
        ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3); ax.set_title("mean scores", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "p1_imbalance.png")); plt.close(fig)
        NUM["imbalanced_trivial_accuracy"] = float(1 - triv)
        for m in METRICS:
            NUM.setdefault("imb_means", {})[m] = {LABEL[k]: round(float(per_dataset(B, m, imb).mean()[k]), 3) for k in ORDER}
        NUM["imb_kendall"] = {f"{a}|{b}": round(float(stats.kendalltau(ranks[METRICS[a]], ranks[METRICS[b]])[0]), 2)
                              for a in METRICS for b in METRICS if a < b}

    tm = B.groupby("model").t_fit.mean()
    NUM["fit_time_mean"] = {LABEL[k]: round(float(tm[k]), 3) for k in ORDER}

    pA = os.path.join(RES, "exp_A_sim_noise.csv")
    if os.path.exists(pA):
        A = pd.read_csv(pA)
        g = A.groupby(["problem", "noise", "model"]).err.agg(["mean", "std"]).reset_index()
        meths = ["tree", "bagging", "adaboost_stumps", "adaboost_depth3", "rf", "gb"]
        rows = []
        for (pr, nz), sub in g.groupby(["problem", "noise"]):
            sub = sub.set_index("model"); best = sub["mean"].min()
            rows.append([pr, f"{int(nz * 100)}\\%"] + [(r"\textbf{%.3f}" if sub.loc[k, "mean"] <= best + 5e-4 else "%.3f") % sub.loc[k, "mean"] for k in meths])
        T = pd.DataFrame(rows, columns=["problem", "noise"] + [LABEL[k] for k in meths])
        open(os.path.join(TAB, "sim_noise.tex"), "w").write(esc(T.to_latex(index=False, escape=False, column_format="llcccccc")))
        NUM["sim_noise"] = {f"{pr}|{nz}": {k: round(float(sub.set_index('model').loc[k, 'mean']), 3) for k in meths} for (pr, nz), sub in g.groupby(["problem", "noise"])}
    pB = os.path.join(RES, "exp_B_rounds.csv")
    if os.path.exists(pB):
        R = pd.read_csv(pB)
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True)
        for ax, nz in zip(axes, (0.0, 0.2)):
            for mth, ls in (("bagging", "-"), ("adaboost_stumps", "-"), ("adaboost_depth3", "-")):
                s = R[(R.noise == nz) & (R.model == mth)].groupby("T").test_err.mean()
                ax.plot(s.index, s.values, ls, color=COL[mth], label=f"{LABEL[mth]}: test")
            for mth in ("adaboost_stumps", "adaboost_depth3"):
                s = R[(R.noise == nz) & (R.model == mth)].groupby("T").train_err.mean()
                ax.plot(s.index, s.values, ":", color=COL[mth], label=f"{LABEL[mth]}: train (noisy labels)")
            ax.set_xscale("log"); ax.set_xlabel("number of rounds / trees $T$"); ax.set_title(f"twonorm, {int(nz * 100)}% label noise", fontsize=8)
        axes[0].set_ylabel("error"); axes[0].legend(fontsize=6, loc="upper right")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "p1_rounds.png")); plt.close(fig)
        NUM["rounds"] = {}
        for nz in (0.0, 0.2):
            for mth in ("bagging", "adaboost_stumps", "adaboost_depth3"):
                s = R[(R.noise == nz) & (R.model == mth)].groupby("T").test_err.mean()
                NUM["rounds"][f"{mth}|{nz}"] = dict(min=round(float(s.min()), 4), argmin=int(s.idxmin()), at10=round(float(s.loc[10]), 4), final=round(float(s.iloc[-1]), 4), Tmax=int(s.index[-1]))
    pC = os.path.join(RES, "exp_C_depth.csv")
    if os.path.exists(pC):
        C = pd.read_csv(pC)
        names = list(C.data.unique())
        fig, axes = plt.subplots(1, len(names), figsize=(1.7 * len(names) + 0.4, 2.2), sharex=True)
        for ax, nm in zip(np.atleast_1d(axes), names):
            for meth, c, mk in (("single tree", COL["tree"], "s"), ("bagging", COL["bagging"], "o"), ("adaboost", COL["adaboost_stumps"], "^")):
                s = C[(C.data == nm) & (C.method == meth)].groupby("depth").err.mean()
                s = s.rename(index=lambda i: 9 if i == 0 else i).sort_index()
                ax.plot(range(len(s)), s.values, marker=mk, ms=3, lw=1, color=c, label=meth)
            ax.set_xticks(range(6)); ax.set_xticklabels(["1", "2", "3", "5", "8", "∞"], fontsize=6.5); ax.set_title(nm, fontsize=8)
            ax.set_xlabel("base-tree depth", fontsize=7)
        np.atleast_1d(axes)[0].set_ylabel("test error"); np.atleast_1d(axes)[0].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "p1_depth.png")); plt.close(fig)
        NUM["depth"] = {nm: {f"{meth}|{dp}": round(float(v), 3) for (meth, dp), v in C[C.data == nm].groupby(["method", "depth"]).err.mean().items()} for nm in names}
    pD = os.path.join(RES, "exp_D_size.csv")
    if os.path.exists(pD):
        Z = pd.read_csv(pD)
        names = list(Z.data.unique())
        fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 2.3), sharey=False)
        for ax, nm in zip(np.atleast_1d(axes), names):
            for mth in ("tree", "bagging", "adaboost_stumps", "adaboost_depth3", "rf"):
                s = Z[(Z.data == nm) & (Z.model == mth)].groupby("m").err.mean()
                ax.plot(s.index, s.values, "o-", ms=2.5, lw=1, color=COL[mth], label=LABEL[mth])
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training size $m$"); ax.set_title(nm, fontsize=8)
        np.atleast_1d(axes)[0].set_ylabel("test error"); np.atleast_1d(axes)[0].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "p1_size.png")); plt.close(fig)
        NUM["size"] = {nm: {f"{mth}|{m}": round(float(v), 3) for (mth, m), v in Z[Z.data == nm].groupby(["model", "m"]).err.mean().items()} for nm in names}
    pE = os.path.join(RES, "exp_E_stacking.csv")
    if os.path.exists(pE):
        S5 = pd.read_csv(pE)
        G = S5.groupby("data").mean(numeric_only=True)
        cols = ["stacking", "stacking_naive", "uniform_vote", "best_member_cv", "best_member_oracle", "mean_member"]
        NUM["stacking"] = {c: round(float(G[c].mean()), 4) for c in cols}
        NUM["stacking"]["wins_vs_best_member_cv"] = int((G.stacking < G.best_member_cv - 1e-9).sum())
        NUM["stacking"]["wins_vs_best_member_oracle"] = int((G.stacking < G.best_member_oracle - 1e-9).sum())
        NUM["stacking"]["ties_vs_best_member_oracle"] = int((abs(G.stacking - G.best_member_oracle) <= 1e-9).sum())
        NUM["stacking"]["N"] = int(len(G))
        NUM["stacking"]["time_ratio_median"] = float((G.t_stacking / G.t_members).median())
        NUM["stacking"]["time_ratio_mean"] = float((G.t_stacking / G.t_members).mean())
        NUM["stacking"]["wilcoxon_vs_cv_best_p"] = float(stats.wilcoxon(G.stacking - G.best_member_cv).pvalue)
        NUM["stacking"]["wilcoxon_vs_oracle_p"] = float(stats.wilcoxon(G.stacking - G.best_member_oracle).pvalue)
        NUM["stacking"]["wilcoxon_naive_p"] = float(stats.wilcoxon(G.stacking - G.stacking_naive).pvalue)
        w = G[[c for c in G.columns if c.startswith("w_")]]
        NUM["stacking"]["mean_meta_weights"] = {c[2:]: round(float(w[c].mean()), 3) for c in w.columns}
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [2.2, 1]})
        ax = axes[0]
        G2 = G.sort_values("best_member_oracle")
        x = np.arange(len(G2))
        ax.plot(x, G2.stacking_naive - G2.best_member_oracle, "v", color="#c8553d", ms=4, label="naive stacking (in-sample)")
        ax.plot(x, G2.uniform_vote - G2.best_member_oracle, "s", color="#8c8c8c", ms=3.5, label="uniform vote of the members")
        ax.plot(x, G2.stacking - G2.best_member_oracle, "o", color="#2a9d8f", ms=4, label="stacking (out-of-fold)")
        ax.axhline(0, color="k", lw=0.7); ax.set_xticks(x); ax.set_xticklabels(G2.index, rotation=75, ha="right", fontsize=6)
        ax.set_ylabel("error $-$ error of the best member (test)"); ax.legend(fontsize=6)
        ax = axes[1]
        mw = w.mean(); ax.bar([c[2:] for c in mw.index], mw.values, color="#2a9d8f"); ax.set_title("mean meta-model coefficients", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "p1_stacking.png")); plt.close(fig)
        Tt = G[cols].copy(); Tt.columns = ["stacking", "naive stack.", "uniform vote", "best member (CV)", "best member (oracle)", "mean member"]
        open(os.path.join(TAB, "stacking.tex"), "w").write(esc(Tt.round(3).to_latex(escape=False, column_format="l" + "c" * 6)))

    pF = os.path.join(RES, "exp_F_realnoise.csv")
    if os.path.exists(pF):
        Fn = pd.read_csv(pF)
        meths = ["tree", "bagging", "adaboost_stumps", "adaboost_depth3", "rf", "gb"]
        g = Fn.groupby(["noise", "model"]).err.mean().unstack()[meths]
        T = g.copy(); T.index = [f"{int(i * 100)}\\%" for i in T.index]; T.columns = [LABEL[c] for c in T.columns]; T.index.name = "label noise"
        open(os.path.join(TAB, "real_noise.tex"), "w").write(esc(T.round(3).to_latex(escape=False, float_format="%.3f", column_format="l" + "c" * 6)))
        gd = Fn.groupby(["data", "noise", "model"]).err.mean().unstack()
        NUM["real_noise"] = {f"{int(nz * 100)}": {k: round(float(v), 3) for k, v in g.loc[nz].items()} for nz in g.index}
        NUM["real_noise_wins"] = {f"{int(nz * 100)}": {"bag<ada_stumps": int((gd.xs(nz, level="noise").bagging < gd.xs(nz, level="noise").adaboost_stumps).sum()),
                                                      "bag<ada_depth3": int((gd.xs(nz, level="noise").bagging < gd.xs(nz, level="noise").adaboost_depth3).sum()),
                                                      "N": int(len(gd.xs(nz, level="noise")))} for nz in g.index}

    json.dump(NUM, open(os.path.join(RES, "part1_numbers.json"), "w"), indent=1, default=str)
    print("analysis_part1 done:", len(D), "datasets")


if __name__ == "__main__":
    main()
