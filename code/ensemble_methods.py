"""Partie 1 : trois méthodes d'ensemble codées à la main.comme demandé dans l'énoncé de projet
 Les étiquettes sont dans {-1, +1}.
les methodes sont :
   - Bagging    arbres de décision appris sur des bootstraps, vote majoritaire, erreur out-of-bag
   - AdaBoost   arbres pondérés ajoutés un par un (profondeur 1 = stumps), poids mis à jour à chaque tour
   - Stacking   modèles de familles différentes combinés par un méta-modèle, via des prédictions out-of-fold

Chaque classe a les méthodes fit, predict et decision_function de scikit-learn, donc elle s'utilise
comme n'importe quel autre modèle, par exemple avec benchkit.evaluate.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def _vote(S, tie):
    """Renvoie +1 si le score S est positif, -1 s'il est négatif.
    Si S vaut exactement 0 (égalité parfaite des votes), on renvoie `tie`, qui est la classe
    la plus fréquente dans le jeu d'entraînement."""
    return np.where(S > 0, 1, np.where(S < 0, -1, tie))


"""  bagging
#c quoi ?: Bagging : B arbres, chacun appris sur un échantillon bootstrap, puis vote majoritaire.

    Un échantillon bootstrap est formé de m tirages avec remise parmi les m exemples du train.
    Environ 37 % des exemples n'y apparaissent jamais : ils sont out-of-bag pour cet arbre.

    Erreur OOB : pour un exemple, on ne garde que les votes des arbres qui ne l'ont pas utilisé
    pour leur apprentissage. La proportion d'exemples mal classés par ce vote est une estimation
    de l'erreur de test, obtenue sans jeu de validation séparé.

    Paramètres
        n_estimators  nombre d'arbres B
        max_depth     profondeur maximale ; None = arbres complets
        max_features  nombre de variables examinées à chaque coupure ; None = toutes (bagging),
                      une valeur plus petite (par exemple "sqrt") donne une forêt aléatoire

    Attributs après fit
        trees_                les B arbres
        oob_error_            erreur du vote out-of-bag
        oob_sum_, oob_cnt_    pour chaque exemple : somme des votes +-1 des arbres qui ne l'ont pas vu, et nombre de ces arbres
        tie_                  classe la plus fréquente du train, utilisée en cas d'égalité

    decision_function(X) est la moyenne des votes +-1, comprise dans [-1, 1] ; elle sert de score pour l'AUC.
#"""

class Bagging(BaseEstimator, ClassifierMixin):

    def __init__(self, n_estimators=100, max_depth=None, max_features=None, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        m = len(y)
        self.tie_ = 1 if np.mean(y == 1) >= 0.5 else -1
        self.trees_ = []
        self.oob_sum_, self.oob_cnt_ = np.zeros(m), np.zeros(m)
        for b in range(self.n_estimators):
            idx = rng.randint(0, m, m)
            oob = np.ones(m, bool); oob[idx] = False
            t = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.max_features,
                                       random_state=rng.randint(1 << 30)).fit(X[idx], y[idx])
            self.trees_.append(t)
            if oob.any():
                self.oob_sum_[oob] += t.predict(X[oob]); self.oob_cnt_[oob] += 1
        ok = self.oob_cnt_ > 0
        self.oob_error_ = float(np.mean(_vote(self.oob_sum_[ok], self.tie_) != y[ok]))
        return self

    def votes(self, X):
        """Matrice B x n : ligne b = prédictions +-1 de l'arbre b sur les n exemples de X."""
        return np.array([t.predict(X) for t in self.trees_])

    def decision_function(self, X):
        return self.votes(X).mean(0)

    def predict(self, X):
        return _vote(self.decision_function(X), self.tie_)


'''AdaBoost
c'est quoi:  AdaBoost discret : les arbres sont appris l'un après l'autre sur des exemples pondérés.

    On part de poids tous égaux w_i = 1/m. À chaque tour t :
        1. on apprend un arbre h_t en tenant compte des poids w ;
        2. eps_t = somme des poids des exemples qu'il classe mal ;
        3. alpha_t = 1/2 ln((1 - eps_t) / eps_t) : plus eps_t est petit, plus l'arbre compte ;
        4. w_i <- w_i exp(-alpha_t y_i h_t(x_i)) / Z_t : les exemples mal classés prennent du poids,
           Z_t remet la somme des poids à 1.
    La prédiction est le signe de sum_t alpha_t h_t(x).

    On s'arrête avant T tours si eps_t >= 1/2 (l'arbre ne fait pas mieux que le hasard) ou si
    eps_t = 0 (rien de plus à corriger sur le train). Si eps_t = 0, on le remplace par 1e-12 dans
    le calcul de alpha_t pour qu'il reste fini.

    Paramètres
        n_estimators  nombre maximal de tours T
        max_depth     profondeur des arbres ; 1 = stumps (une seule coupure)

    Attributs après fit
        trees_    les arbres h_t
        alphas_   les poids alpha_t
        eps_      les erreurs pondérées eps_t
        Z_        les constantes de normalisation Z_t ; le produit des Z_t majore l'erreur d'entraînement

    staged_decision_function(X) donne le score après 1, 2, ..., T arbres, pour tracer l'erreur en fonction de T.
    decision_function(X) est la somme pondérée divisée par sum(alpha_t), donc dans [-1, 1].
'''
class AdaBoost(BaseEstimator, ClassifierMixin):
   
    def __init__(self, n_estimators=200, max_depth=1, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        m = len(y)
        w = np.full(m, 1.0 / m)
        self.trees_, self.alphas_, self.eps_, self.Z_ = [], [], [], []
        for t in range(self.n_estimators):
            h = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            h.fit(X, y, sample_weight=w)
            pred = h.predict(X)
            eps = float(w[pred != y].sum())
            if eps >= 0.5 and t > 0:
                break
            eps_c = max(eps, 1e-12)
            alpha = 0.5 * np.log((1 - eps_c) / eps_c)
            w = w * np.exp(-alpha * y * pred)
            Z = w.sum(); w /= Z
            self.trees_.append(h); self.alphas_.append(alpha); self.eps_.append(eps); self.Z_.append(Z)
            if eps == 0:
                break
        self.alphas_ = np.array(self.alphas_)
        return self

    def staged_decision_function(self, X):
        S = np.zeros(len(X))
        for a, h in zip(self.alphas_, self.trees_):
            S = S + a * h.predict(X)
            yield S.copy()

    def decision_function(self, X):
        return sum(a * h.predict(X) for a, h in zip(self.alphas_, self.trees_)) / max(self.alphas_.sum(), 1e-12)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


# meth3: stacking
"""c quoi: un méta-modèle apprend à combiner les sorties de plusieurs modèles de base.

    Apprentissage
        1. On découpe le train en n_splits plis. Pour chaque pli, chaque modèle de base est appris
           sur les autres plis et prédit sur ce pli. Ainsi, chaque exemple du train reçoit, de
           chaque modèle, une prédiction faite sans qu'il l'ait utilisé : c'est la prédiction
           out-of-fold. On les range dans une matrice Z (une ligne par exemple, une colonne par modèle).
        2. Le méta-modèle (par défaut une régression logistique) est appris sur Z avec les vraies étiquettes.
        3. Les modèles de base sont ensuite réappris sur tout le train.
    Prédiction : les modèles de base prédisent X, et le méta-modèle transforme ces sorties en classe.

    Pourquoi out-of-fold : un modèle de base prédit très bien les exemples qu'il a appris
    (un arbre complet les retrouve tous). Entraîner le méta-modèle sur de telles prédictions lui
    ferait surestimer ce modèle. Avec oof=False, on obtient justement ce stacking naïf, utile pour
    montrer la différence dans les expériences.

    Paramètres
        base_models  liste de (nom, modèle) ; None = default_base_models
        meta         méta-modèle ; None = régression logistique après standardisation
        n_splits     nombre de plis
        oof          True = prédictions out-of-fold, False = prédictions sur les exemples déjà vus

    Attributs après fit : names_, base_ (modèles de base réappris), meta_, Z_ (matrice d'entrée du méta-modèle).
    Coût : chaque modèle de base est appris n_splits + 1 fois (avec oof=True).
"""
def default_base_models(random_state=0):
    """5 modèles de base de familles différentes 
    - Le StandardScaler est dans le pipeline, donc il
    est recalculé sur les seules données d'entraînement de chaque pli."""
    return [("tree", DecisionTreeClassifier(max_depth=5, random_state=random_state)),
            ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
            ("7nn", make_pipeline(StandardScaler(), KNeighborsClassifier(7))),
            ("nb", GaussianNB()),
            ("svm", make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")))]


def _meta_features(model, X):
    """Valeur réelle que le modèle de base donne à chaque exemple de X : la probabilité de la classe +1
    si le modèle sait en calculer une, sinon sa decision_function. C'est l'entrée du méta-modèle."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(X))
    return model.decision_function(X)


class Stacking(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models=None, meta=None, n_splits=5, oof=True, random_state=0):
        self.base_models = base_models
        self.meta = meta
        self.n_splits = n_splits
        self.oof = oof
        self.random_state = random_state

    def fit(self, X, y):
        base = self.base_models if self.base_models is not None else default_base_models(self.random_state)
        self.names_ = [n for n, _ in base]
        m = len(y)
        Z = np.zeros((m, len(base)))
        if self.oof:
            skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_state)
            for a, b in skf.split(X, y):
                for j, (_, mdl) in enumerate(base):
                    Z[b, j] = _meta_features(clone(mdl).fit(X[a], y[a]), X[b])
        self.base_ = [clone(mdl).fit(X, y) for _, mdl in base]
        if not self.oof:
            for j, mdl in enumerate(self.base_):
                Z[:, j] = _meta_features(mdl, X)
        meta = self.meta if self.meta is not None else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        self.meta_ = clone(meta).fit(Z, y)
        self.Z_ = Z
        return self

    def base_features(self, X):
        """Matrice n x (nombre de modèles) des sorties des modèles de base sur X."""
        return np.column_stack([_meta_features(mdl, X) for mdl in self.base_])

    def decision_function(self, X):
        return self.meta_.decision_function(self.base_features(X))

    def predict(self, X):
        return self.meta_.predict(self.base_features(X))