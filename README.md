# Projet — Méthodes d'ensemble et apprentissage PAC-Bayésien (Master 2 MALIA)

Cours « Ensemble Methods » (G. Metzler).
Rapport : `report/rapport.pdf` (source `report/rapport.tex` ; page de garde + 12 pages de texte + annexes, soit 15 pages au total).

## Organisation du dossier

| Chemin | Contenu |
|--------|---------|
| `code/ensemble_methods.py` | **Partie 1**, nos implémentations : `Bagging` (avec erreur OOB), `AdaBoost` (discret), `Stacking` (méta-variables hors-pli) |
| `code/validate_implementations.py` | vérifie ces implémentations contre scikit-learn → `results/validation.csv` |
| `code/benchmark.py` | benchmark principal de la partie 1 (10 méthodes × 21 jeux × 10 découpages) → `results/benchmark.csv` |
| `code/experiments_part1.py` | expériences contrôlées A–F (simulations avec bruit d'étiquettes, nombre de tours, profondeur, taille, stacking, bruit sur données réelles) |
| `code/analysis_part1.py` | tableaux de la partie 1 (`results/tables/*.tex`), figures (`figures/p1_*.pdf`), chiffres (`results/part1_numbers.json`) |
| `code/experiments_part2.py` | partie 2, questions Q1 à Q5 (utilise `pacbayes_tools.py` fourni, inchangé) → `results/p2_*.csv` |
| `code/analysis_part2.py` | tableaux et figures de la partie 2 (`figures/p2_*.pdf`), chiffres (`results/part2_numbers.json`) |
| `code/benchkit.py`, `code/pacbayes_tools.py`, `common/`, `data/` | fichiers fournis, **non modifiés** |
| `notebooks/01_partie1_methodes_ensemble.ipynb`, `notebooks/02_partie2_pac_bayes.ipynb` | lecture guidée des parties 1 et 2 : ils rechargent `results/` et affichent tableaux et courbes ; rien de long n'est recalculé (`cd notebooks && jupyter notebook`, nécessite `pip install notebook`) |
| `report/rapport.tex` | le rapport (`latexmk -pdf rapport.tex` ; lit `../figures` et `../results/tables`) |

## Installation

Python 3 et les paquets de `requirements.txt` (`pip install -r requirements.txt` ; testé avec numpy 2.5, scipy 1.18, scikit-learn 1.9, matplotlib 3.11, pandas 3.0, joblib 1.6). Une installation LaTeX n'est nécessaire que pour recompiler le PDF.

## Tout reproduire

Depuis le dossier `code/` (toutes les graines sont fixées) :

```bash
cd code
python validate_implementations.py      # ~2 min   -> results/validation.csv
python benchmark.py                     # ~10 min sur 2 coeurs (cache par jeu de données dans results/bench/)
python experiments_part1.py             # ~6 min   (ou un seul bloc : python experiments_part1.py A, de A à F)
python analysis_part1.py                # tableaux + figures de la partie 1
python experiments_part2.py             # ~1 min   (ou Q1 ... Q5)
python analysis_part2.py                # tableaux + figures de la partie 2
cd ../report && latexmk -pdf rapport.tex
```

Remarque : `benchmark.py` met ses résultats en cache ; pour le relancer entièrement, vider `results/bench/` avant.

## Quel script produit quel tableau / quelle figure du rapport

| Rapport | Script |
|---------|--------|
| Tableau des jeux de données, tableau des rangs + diagrammes de différence critique, tests par paires (Holm), tableaux annexes des 4 métriques | `benchmark.py` → `analysis_part1.py` |
| Figure bagging − boosting selon taille / dimension / déséquilibre ; figure des jeux déséquilibrés | `benchmark.py` → `analysis_part1.py` |
| Courbes d'apprentissage / profondeur / simulations et nombre de tours / bruit sur données réelles / stacking | `experiments_part1.py`, blocs D / C / A et B / F / E → `analysis_part1.py` |
| Partie 2 : Q1 (figure + tableau), Q2 (tableau + figure), Q3 (tableau + figure), Q4 (tableau + figure), Q5 (deux tableaux) | `experiments_part2.py` Q1…Q5 → `analysis_part2.py` |

## Protocoles

- **Partie 1** : 10 découpages stratifiés 70/30 répétés (graine 0), mêmes découpages pour toutes les méthodes, hyperparamètres fixés (pas de réglage) ; jeu déséquilibré = classe minoritaire < 20 % ; `bankmarketing` écarté (trop volumineux pour 10 méthodes × 10 découpages).
- **Partie 2** : fonction `split` de `pacbayes_tools.py` (14 % partie prior, 56 % échantillon de la borne, 30 % test), graines 0 à 9, δ = 0,05.

## Remarques

- Les étiquettes des figures et les en-têtes des tableaux générés par les scripts sont en anglais ; le texte du rapport est en français.
- Un nouveau lancement peut donner de très faibles écarts (de l'ordre de 10⁻³) sur certaines valeurs (stacking, risques de test PAC-Bayes sur stumps) selon la version des bibliothèques et le parallélisme ; les conclusions ne changent pas.
