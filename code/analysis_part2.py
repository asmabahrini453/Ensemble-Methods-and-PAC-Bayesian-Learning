"""Partie 2 : tableaux et figures à partir des résultats de experiments_part2.py.

Entrées : results/p2_*.csv.le script lit les CSV et met en forme.

Une fonction par question, dans l'ordre Q1 à Q5 ; chacune produit un tableau, une figure et des chiffres.

Justification de la démarche
    - Moyennes sur les 10 graines : chaque graine donne un découpage différent ; on moyenne pour ne pas dépendre
      d'un tirage favorable, et on garde aussi des comparaisons graine par graine (proportion de cas où une méthode
      est meilleure) avec un test de Wilcoxon, pour savoir si un écart est plus que du hasard.
    - Le certificat rapporté est celui de Seeger, la borne la plus serrée ; la valeur de la borne A ou B est donnée
      à côté, car c'est elle que l'algorithme minimise.
    - Le risque du vote majoritaire est comparé à celui de Gibbs : le vote est au plus 2 fois le risque de Gibbs,
      donc un certificat sur Gibbs est un certificat plus faible pour le vote.
    - Une garantie n'a de valeur que si elle tient : on compare toujours le certificat au risque mesuré sur le test.

"""
import json
import os
import re
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
                     "legend.frameon": False, "savefig.bbox": "tight", "figure.dpi": 130})
C1, C2, C3, C4, C5, GREY = "#1f5f99", "#c8553d", "#2a9d8f", "#e9a13b", "#6c4f9c", "#8c8c8c"
NUM = {}

#Transforme un DataFrame dont les colonnes ont deux niveaux en dictionnaire simple (clés 'a|b'), pour le JSON.
def flat(df):
    df = df.copy(); df.columns = ["|".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
    return df.to_dict("index")


def esc(s):
    return "".join(p if p.startswith("$") else p.replace("_", r"\_") for p in re.split(r"(\$[^$]*\$)", s))


def write_tex(df, name, **kw):
    s = esc(df.to_latex(escape=False, **kw))
    open(os.path.join(TAB, name + ".tex"), "w").write(s)

"""Q1, les bornes sans données. Figure p2_q1_bounds : trois panneaux pour un risque empirique de 0,1 :
    bornes A (grille), B et Seeger selon m (KL = 5 et 50), selon KL (m = 200 et 5000), et borne A selon lambda
    pour m = 1000 (points = grille 2^k, trait pointillé = lambda optimal continu). Tableau p2_q1 : valeurs de
    chaque borne pour m proche de 200, 1000 et 10 000 et KL = 5 et 50, avec le coût de la grille (A grille contre A au
    meilleur lambda continu contre A à un seul lambda). Chiffres : rapport B/A, plus petit m où A passe sous B,
    et termes de l'union sur la grille (ln(15/delta), ln(1/delta), terme de McAllester)."""
def q1():
    
    V = pd.read_csv(os.path.join(RES, "p2_q1_vs_m.csv"))
    K = pd.read_csv(os.path.join(RES, "p2_q1_vs_KL.csv"))
    L = pd.read_csv(os.path.join(RES, "p2_q1_vs_lambda.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.4))
    ax = axes[0]
    for KL, ls in ((5, "-"), (50, "--")):
        s = V[V.KL == KL]
        ax.plot(s.m, s.A_grid, ls, color=C1, label=f"A (grid), KL={KL}")
        ax.plot(s.m, s.B, ls, color=C2, label=f"B, KL={KL}")
        ax.plot(s.m, s.Seeger, ls, color=C3, label=f"Seeger, KL={KL}")
    ax.axhline(0.1, color="k", lw=0.6, ls=":"); ax.set_xscale("log"); ax.set_xlabel("sample size $m$"); ax.set_ylabel("bound on $R_D(G_Q)$")
    ax.set_title(r"$\widehat{R}_S=0.1$", fontsize=8); ax.legend(fontsize=5.5, ncol=1)
    ax = axes[1]
    for m, ls in ((200, "--"), (5000, "-")):
        s = K[K.m == m]
        ax.plot(s.KL, s.A_grid, ls, color=C1, label=f"A, $m$={m}"); ax.plot(s.KL, s.B, ls, color=C2, label=f"B, $m$={m}"); ax.plot(s.KL, s.Seeger, ls, color=C3, label=f"Seeger, $m$={m}")
    ax.axhline(0.1, color="k", lw=0.6, ls=":"); ax.set_xlabel("KL$(Q\\|P)$"); ax.set_ylim(0.05, 1.05); ax.set_title(r"$\widehat{R}_S=0.1$", fontsize=8)
    ax = axes[2]
    for (m, KL), c in zip(((1000, 1), (1000, 10), (1000, 50), (1000, 200)), (C3, C1, C4, C2)):
        s = L[(L.m == m) & (L.KL == KL)]
        ax.plot(s.lam, s.A, color=c, label=f"KL={KL}")
        lam_star = np.sqrt(8 * m * (KL + np.log(15 / 0.05)))
        ax.axvline(lam_star, color=c, lw=0.5, ls=":")
        grid = 2.0 ** np.arange(15)
        ax.plot(grid, 0.10 + (KL + np.log(15 / 0.05)) / grid + grid / (8 * m), "o", ms=2.2, color=c)
    ax.set_xscale("log"); ax.set_xlabel("$\\lambda$ (dots: the grid $2^k$)"); ax.set_ylim(0.1, 1.0); ax.set_title("bound A, $m=1000$", fontsize=8); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p2_q1_bounds.pdf")); plt.close(fig)
    emp, rows = 0.10, []
    for m in (200, 1000, 10000):
        for KL in (5, 50):
            r = V[(V.KL == KL)].iloc[(V[V.KL == KL].m - m).abs().argmin()]
            rows.append(dict(m=int(r.m), KL=KL, A_grid=r.A_grid, A_cont=r.A_cont, A_single=r.A_single_lambda, B=r.B, Seeger=r.Seeger))
    NUM["q1_table"] = rows
    T = pd.DataFrame(rows)
    T.columns = ["$m$", "KL", "A (grid)", "A (best $\\lambda$)", "A (one $\\lambda$)", "B", "Seeger"]
    write_tex(T, "p2_q1", index=False, float_format="%.3f", column_format="rrrrrrr")
    NUM["q1_ratio_B_over_A"] = {f"KL={KL}": {int(m): round(float(b / a), 3) for m, a, b in zip(s.m, s.A_grid, s.B) if m in (50, 100, 1000, 10000, 100000)}
                                for KL, s in V.groupby("KL")}
    s = V[V.KL == 5]
    NUM["q1_A_beats_B_from_m"] = int(s[s.A_grid < s.B].m.min()) if (s.A_grid < s.B).any() else None
    c = np.log(15 / 0.05)
    NUM["q1_grid_price"] = {"ln(15/delta)": c, "ln(1/delta)": float(np.log(1 / 0.05)), "ln(2 sqrt(m)/delta) at m=1000": float(np.log(2 * np.sqrt(1000) / 0.05))}

"""Q2, borne A contre borne B sur des stumps. Tableau p2_q2_compare : par jeu et par borne, valeur de la borne,
    certificat de Seeger, KL, nombre effectif de voteurs, risque empirique de Gibbs, risques de test de Gibbs et du
    vote, plus une ligne de moyenne. Figure p2_q2_lambda (jeux wdbc et german) : borne A, certificat de Seeger,
    risques de test et risque empirique selon lambda, puis KL et nombre effectif de voteurs selon lambda. Chiffres :
    part des cas où A est plus serrée que B, écarts moyens, lambda choisi, écart certificat - risque de test, test
    de Wilcoxon sur la différence A - B."""
def q2():
    d = pd.read_csv(os.path.join(RES, "p2_q2_compare.csv"))
    cols = ["bound", "seeger", "KL", "neff", "emp", "test_gibbs", "test_vote"]
    g = d.groupby(["data", "bound_name"])[cols].mean()
    rows = []
    for (dt, bn), r in g.iterrows():
        rows.append([dt, bn[0]] + [r[c] for c in cols])
    T = pd.DataFrame(rows, columns=["dataset", "bound", "value of the bound", "Seeger", "KL", "$n_{\\rm eff}$", "$\\widehat R_S(G_Q)$", "test Gibbs", "test vote"])
    mean = d.groupby("bound_name")[cols].mean()
    for bn, r in mean.iterrows():
        rows.append(["mean", bn[0]] + [r[c] for c in cols])
    T = pd.DataFrame(rows, columns=T.columns)
    write_tex(T, "p2_q2_compare", index=False, float_format="%.3f", column_format="llrrrrrrr")
    p = d.pivot_table(index=["data", "seed"], columns="bound_name", values=cols)
    NUM["q2"] = {"mean": mean.round(4).to_dict("index"),
                 "frac_seeger_A_lt_B": float((p["seeger"]["A (linear)"] < p["seeger"]["B (McAllester)"]).mean()),
                 "frac_bound_A_lt_B": float((p["bound"]["A (linear)"] < p["bound"]["B (McAllester)"]).mean()),
                 "mean_abs_diff_seeger": float((p["seeger"]["A (linear)"] - p["seeger"]["B (McAllester)"]).abs().mean()),
                 "mean_abs_diff_test_vote": float((p["test_vote"]["A (linear)"] - p["test_vote"]["B (McAllester)"]).abs().mean()),
                 "mean_abs_diff_KL": float((p["KL"]["A (linear)"] - p["KL"]["B (McAllester)"]).abs().mean()),
                 "lam_counts": d[d.bound_name.str.startswith("A")].lam.value_counts().sort_index().to_dict(),
                 "gap_seeger_minus_test_gibbs": float((d.seeger - d.test_gibbs).mean()),
                 "gap_seeger_minus_test_vote": float((d.seeger - d.test_vote).mean()),
                 "wilcoxon_seeger_p": float(stats.wilcoxon(p["seeger"]["A (linear)"] - p["seeger"]["B (McAllester)"]).pvalue)}
    Lm = pd.read_csv(os.path.join(RES, "p2_q2_lambda.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), sharex=True)
    for row, name in enumerate(("wdbc", "german")):
        s = Lm[Lm.data == name].groupby("lam").mean(numeric_only=True)
        ax = axes[row, 0]
        ax.plot(s.index, s.bound_A, "o-", ms=3, color=C1, label="bound A (value)")
        ax.plot(s.index, s.seeger, "o-", ms=3, color=C3, label="Seeger certificate")
        ax.plot(s.index, s.test_gibbs, "s-", ms=3, color=C2, label="test Gibbs risk")
        ax.plot(s.index, s.test_vote, "^-", ms=3, color=C4, label="test vote risk")
        ax.plot(s.index, s.emp, "d:", ms=3, color=GREY, label="empirical Gibbs risk on $S$")
        a = s.bound_A.idxmin(); ax.axvline(a, color=C1, lw=0.6, ls=":")
        ax.set_xscale("log", base=2); ax.set_ylim(0, 0.8); ax.set_title(f"{name} (stumps, mean of 10 splits)", fontsize=8); ax.set_ylabel("risk")
        ax = axes[row, 1]
        ax.plot(s.index, s.KL, "o-", ms=3, color=C5, label="KL$(Q\\|P)$")
        ax.plot(s.index, s.neff, "o-", ms=3, color=C2, label="$n_{\\rm eff}=1/\\sum_h Q(h)^2$")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.axvline(a, color=C1, lw=0.6, ls=":")
        if row == 1:
            axes[row, 0].set_xlabel("$\\lambda$"); axes[row, 1].set_xlabel("$\\lambda$")
    axes[0, 0].legend(fontsize=6, ncol=1); axes[0, 1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p2_q2_lambda.pdf")); plt.close(fig)
    s = Lm.groupby(["data", "lam"]).mean(numeric_only=True).reset_index()
    NUM["q2_lambda"] = {nm: {"lam_best_bound": float(x.loc[x.bound_A.idxmin(), "lam"]), "lam_best_test_vote": float(x.loc[x.test_vote.idxmin(), "lam"]),
                             "KL_at_lam1": float(x.KL.iloc[0]), "KL_at_lam_max": float(x.KL.iloc[-1]), "neff_at_lam_max": float(x.neff.iloc[-1]),
                             "test_vote_at_lam1": float(x.test_vote.iloc[0]), "test_vote_at_lam_max": float(x.test_vote.iloc[-1])} for nm, x in s.groupby("data")}

"""Q3, prior appris contre prior uniforme ou nul, pour la borne A. Tableau p2_q3 : par famille (stumps,
    linéaire gaussien) et par prior, KL, certificat de Seeger, valeur de la borne A et risques de test. Figure
    p2_q3_prior : par jeu, certificat de Seeger, KL et risque de test du vote pour les quatre cas. Chiffres : part des
    cas où le prior appris donne un certificat plus bas ou un meilleur vote, gain moyen sur le certificat, et
    tests de Wilcoxon."""
def q3():
    d = pd.read_csv(os.path.join(RES, "p2_q3_prior.csv"))
    cols = ["KL", "seeger", "bound", "test_gibbs", "test_vote"]
    A = d[d.bound_name.str.startswith("A")]
    g = A.groupby(["family", "prior"])[cols].mean()
    T = g.reset_index()
    T["family"] = T.family.map({"stumps": "stumps (categorical)", "linear": "linear (Gaussian)"})
    T["prior"] = T.prior.map({"uniform": "uniform $P$", "learned": "learned $P$", "zero": "$\\mathcal{N}(0,I)$"}).where(T.family.str.startswith("stumps") | (T.prior != "learned$P$"), T.prior)
    T.loc[(T.family == "linear (Gaussian)") & (T.prior == "learned $P$"), "prior"] = "$\\mathcal{N}(w_0,I)$"
    T.columns = ["family", "prior", "KL", "Seeger", "value of bound A", "test Gibbs", "test vote"]
    write_tex(T, "p2_q3", index=False, float_format="%.3f", column_format="llrrrrr")
    P = A.groupby(["data", "family", "prior"]).seeger.mean().unstack(["family", "prior"])
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3))
    ax = axes[0]
    x = np.arange(len(P)); w = 0.2
    ax.bar(x - 1.5 * w, P[("stumps", "uniform")], w, color=C1, label="stumps, uniform"); ax.bar(x - 0.5 * w, P[("stumps", "learned")], w, color="#8fb4d6", label="stumps, learned")
    ax.bar(x + 0.5 * w, P[("linear", "zero")], w, color=C2, label="linear, $\\mathcal{N}(0,I)$"); ax.bar(x + 1.5 * w, P[("linear", "learned")], w, color="#e3aa9c", label="linear, $\\mathcal{N}(w_0,I)$")
    ax.set_xticks(x); ax.set_xticklabels(P.index, rotation=60, ha="right", fontsize=6.5); ax.set_ylabel("Seeger certificate"); ax.legend(fontsize=5.5, ncol=2, loc="upper left"); ax.set_ylim(0, 0.75)
    ax = axes[1]
    K = A.groupby(["data", "family", "prior"]).KL.mean().unstack(["family", "prior"])
    ax.bar(x - 1.5 * w, K[("stumps", "uniform")], w, color=C1); ax.bar(x - 0.5 * w, K[("stumps", "learned")], w, color="#8fb4d6")
    ax.bar(x + 0.5 * w, K[("linear", "zero")], w, color=C2); ax.bar(x + 1.5 * w, K[("linear", "learned")], w, color="#e3aa9c")
    ax.set_xticks(x); ax.set_xticklabels(K.index, rotation=60, ha="right", fontsize=6.5); ax.set_ylabel("KL$(Q\\|P)$")
    ax = axes[2]
    V = A.groupby(["data", "family", "prior"]).test_vote.mean().unstack(["family", "prior"])
    ax.bar(x - 1.5 * w, V[("stumps", "uniform")], w, color=C1); ax.bar(x - 0.5 * w, V[("stumps", "learned")], w, color="#8fb4d6")
    ax.bar(x + 0.5 * w, V[("linear", "zero")], w, color=C2); ax.bar(x + 1.5 * w, V[("linear", "learned")], w, color="#e3aa9c")
    ax.set_xticks(x); ax.set_xticklabels(V.index, rotation=60, ha="right", fontsize=6.5); ax.set_ylabel("test risk of the vote")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p2_q3_prior.pdf")); plt.close(fig)
    pv = A.pivot_table(index=["data", "seed"], columns=["family", "prior"], values=["seeger", "KL", "test_vote", "test_gibbs"])
    NUM["q3"] = {"table": g.round(4).reset_index().to_dict("records"),
                 "frac_seeger_lower_stumps": float((pv["seeger"][("stumps", "learned")] < pv["seeger"][("stumps", "uniform")]).mean()),
                 "frac_seeger_lower_linear": float((pv["seeger"][("linear", "learned")] < pv["seeger"][("linear", "zero")]).mean()),
                 "frac_vote_better_linear": float((pv["test_vote"][("linear", "learned")] < pv["test_vote"][("linear", "zero")]).mean()),
                 "frac_vote_better_stumps": float((pv["test_vote"][("stumps", "learned")] < pv["test_vote"][("stumps", "uniform")]).mean()),
                 "mean_seeger_gain_linear": float((pv["seeger"][("linear", "zero")] - pv["seeger"][("linear", "learned")]).mean()),
                 "mean_seeger_gain_stumps": float((pv["seeger"][("stumps", "uniform")] - pv["seeger"][("stumps", "learned")]).mean()),
                 "wilcoxon_linear_p": float(stats.wilcoxon(pv["seeger"][("linear", "zero")] - pv["seeger"][("linear", "learned")]).pvalue),
                 "wilcoxon_stumps_p": float(stats.wilcoxon(pv["seeger"][("stumps", "uniform")] - pv["seeger"][("stumps", "learned")]).pvalue)}
    NUM["q3_per_dataset_seeger"] = flat(P.round(3))

"""Q4, stumps contre classifieurs linéaires, et posteriors de comparaison. Le vote de « uniforme sur tous les
    stumps » est mis à nan : les stumps vont par paires de signes opposés, qui s'annulent, donc chaque vote est une
    égalité. Tableau p2_q4 : par posterior, risque empirique, KL, nombre effectif de voteurs, certificat de Seeger et
    risques de test. Figure p2_q4_family : risque de test du vote contre celui de Gibbs pour chaque posterior (au-dessus
    de la diagonale, le vote est moins bon que Gibbs). Chiffres : par jeu, comparaison de la borne A sur stumps et
    sur classifieurs linéaires, et nombre de jeux où la famille linéaire gagne."""
def q4():
    
    d = pd.read_csv(os.path.join(RES, "p2_q4_family.csv"))
    d.loc[d.posterior == "stumps: uniform on all", "test_vote"] = np.nan
    g = d.groupby("posterior")[["emp", "KL", "neff", "seeger", "test_gibbs", "test_vote"]].mean()
    order = ["stumps: Dirac on best stump", "stumps: uniform on 5 best", "stumps: uniform on 20 best", "stumps: uniform on 100 best", "stumps: uniform on all",
             "stumps: bound A", "stumps: bound B", "linear: bound A", "linear: bound B"]
    g = g.loc[order]
    T = g.reset_index()
    T.columns = ["posterior", "$\\widehat R_S(G_Q)$", "KL", "$n_{\\rm eff}$", "Seeger", "test Gibbs", "test vote"]
    T["$n_{\\rm eff}$"] = T["$n_{\\rm eff}$"].map(lambda v: "--" if np.isnan(v) else f"{v:.1f}")
    write_tex(T, "p2_q4", index=False, float_format="%.3f", na_rep="--", column_format="lrrrrrr")
    NUM["q4"] = g.round(4).to_dict("index")
    fig, ax = plt.subplots(figsize=(3.6, 3.1))
    cmap = {"Dirac": C1, "uniform on 5": "#8fb4d6", "uniform on 20": "#8fb4d6", "uniform on 100": "#8fb4d6", "bound": C3, "linear": C2}
    for name, r in g.iterrows():
        if np.isnan(r.test_vote):
            continue
        c = C2 if name.startswith("linear") else (C3 if "bound" in name else C1)
        ax.scatter(r.test_gibbs, r.test_vote, color=c, s=22, zorder=3)
        lab = {"stumps: Dirac on best stump": "stumps: Dirac / bound A / bound B", "stumps: bound A": "", "stumps: bound B": "",
               "linear: bound A": "linear: bound A / B", "linear: bound B": ""}.get(name, name.replace("stumps: ", ""))
        if lab:
            ax.annotate(lab, (r.test_gibbs, r.test_vote), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    ax.plot([0.15, 0.4], [0.15, 0.4], "k:", lw=0.7); ax.set_xlabel("test risk of the Gibbs classifier"); ax.set_ylabel("test risk of the majority vote")
    ax.set_xlim(0.15, 0.4); ax.set_ylim(0.15, 0.32)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p2_q4_family.pdf")); plt.close(fig)
    s = d[d.posterior.isin(["stumps: bound A", "linear: bound A"])].groupby(["data", "posterior"])[["seeger", "test_vote", "test_gibbs"]].mean().unstack()
    NUM["q4_per_dataset"] = flat(s.round(3))
    NUM["q4_linear_better_vote_datasets"] = int((s[("test_vote", "linear: bound A")] < s[("test_vote", "stumps: bound A")]).sum())
    NUM["q4_linear_lower_seeger_datasets"] = int((s[("seeger", "linear: bound A")] < s[("seeger", "stumps: bound A")]).sum())

"""Q5, garantie contre performance. Compare, par jeu, l'erreur de test de cinq méthodes classiques (AdaBoost, forêt,
    HistGradientBoosting, stacking, régression logistique) à celle de quatre méthodes PAC-Bayes (vote). Tableau
    p2_q5_test : erreur de test par jeu et par méthode, la meilleure en gras, avec la moyenne. Tableau p2_q5_cert :
    certificat de Seeger des méthodes PAC-Bayes. Figure p2_q5_classical : barres de l'erreur de test par jeu.
    Chiffres : erreurs et rangs moyens, écart certificat - erreur de test, et comparaisons graine par graine
    (forêt contre PAC-Bayes linéaire à prior appris, AdaBoost contre PAC-Bayes sur stumps) avec un test de Wilcoxon."""
def q5():
    d = pd.read_csv(os.path.join(RES, "p2_q5_classical.csv"))
    m = d.groupby(["data", "approach", "method"])[["test_err", "certificate"]].mean().reset_index()
    keep_pb = ["stumps, bound A (vote)", "linear, bound A, prior zero (vote)", "linear, bound A, prior learned (vote)", "linear, bound B, prior learned (vote)"]
    meths = ["AdaBoost (stumps, T=100)", "random forest (200)", "gradient boosting (hist)", "stacking (ours)", "logistic regression"] + keep_pb
    M = m[m.method.isin(meths)].pivot(index="data", columns="method", values="test_err")[meths]
    C = m[m.method.isin(keep_pb)].pivot(index="data", columns="method", values="certificate")[keep_pb]
    short = {"AdaBoost (stumps, T=100)": "AdaBoost", "random forest (200)": "RF", "gradient boosting (hist)": "HGB", "stacking (ours)": "stacking",
             "logistic regression": "log. reg.", "stumps, bound A (vote)": "PB stumps A", "linear, bound A, prior zero (vote)": "PB lin. A, $\\mathcal{N}(0,I)$",
             "linear, bound A, prior learned (vote)": "PB lin. A, $\\mathcal{N}(w_0,I)$", "linear, bound B, prior learned (vote)": "PB lin. B, $\\mathcal{N}(w_0,I)$"}
    Tt = M.copy(); Tt.columns = [short[c] for c in Tt.columns]
    best = Tt.min(axis=1)
    cells = Tt.copy().astype(object)
    for dset in Tt.index:
        for c in Tt.columns:
            s = f"{Tt.loc[dset, c]:.3f}"
            cells.loc[dset, c] = (r"\textbf{" + s + "}") if abs(Tt.loc[dset, c] - best[dset]) < 5e-4 else s
    cells.loc["mean"] = [f"{Tt[c].mean():.3f}" for c in Tt.columns]
    cells.index.name = None
    write_tex(cells, "p2_q5_test", column_format="l" + "c" * len(Tt.columns))
    Ct = C.copy(); Ct.columns = [short[c] for c in Ct.columns]; Ct.loc["mean"] = Ct.mean(); Ct.index.name = None
    Ct = pd.concat([Ct, Tt.loc[:, [c for c in Tt.columns if c.startswith("PB")]].rename(columns=lambda c: c + " (test)").pipe(lambda x: x.loc[Ct.index[:-1]]) if False else pd.DataFrame()], axis=1)
    write_tex(Ct.round(3), "p2_q5_cert", float_format="%.3f", column_format="l" + "c" * len(Ct.columns))
    ranks = Tt.rank(axis=1).mean()
    NUM["q5"] = {"mean_test_err": Tt.mean().round(4).to_dict(), "mean_rank": ranks.round(2).to_dict(), "mean_certificate": Ct.loc["mean"].round(4).to_dict(),
                 "per_dataset": Tt.round(3).to_dict("index"), "cert_per_dataset": Ct.drop(index="mean").round(3).to_dict("index")}
    pbv = d[d.approach == "PAC-Bayes"].copy()
    gap = pbv.assign(gap=pbv.certificate - pbv.test_err).groupby("method").gap.mean().round(3).to_dict()
    NUM["q5"]["mean_certificate_minus_test"] = gap
    pv = d.pivot_table(index=["data", "seed"], columns="method", values="test_err")
    NUM["q5"]["wilcoxon_rf_vs_pb_lin_learned_p"] = float(stats.wilcoxon(pv["random forest (200)"] - pv["linear, bound A, prior learned (vote)"]).pvalue)
    NUM["q5"]["frac_rf_better_than_pb_lin_learned"] = float((pv["random forest (200)"] < pv["linear, bound A, prior learned (vote)"]).mean())
    NUM["q5"]["frac_logreg_better_than_pb_lin_learned"] = float((pv["logistic regression"] < pv["linear, bound A, prior learned (vote)"]).mean())
    NUM["q5"]["frac_adaboost_better_than_pb_stumps"] = float((pv["AdaBoost (stumps, T=100)"] < pv["stumps, bound A (vote)"]).mean())
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    order = list(Tt.columns)
    xs = np.arange(len(Tt))
    wd = 0.8 / len(order)
    cols = [C1, "#4f8fc4", "#8fb4d6", C3, GREY, C4, C2, "#e3aa9c", "#c98a7c"]
    for i, c in enumerate(order):
        ax.bar(xs + (i - len(order) / 2) * wd + wd / 2, Tt[c], wd, color=cols[i % len(cols)], label=c)
    ax.set_xticks(xs); ax.set_xticklabels(Tt.index, fontsize=7); ax.set_ylabel("test error"); ax.legend(fontsize=5.5, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "p2_q5_classical.pdf")); plt.close(fig)


if __name__ == "__main__":
    q1(); q2(); q3(); q4(); q5()
    json.dump(NUM, open(os.path.join(RES, "part2_numbers.json"), "w"), indent=1, default=str)
    print("analysis_part2 done")