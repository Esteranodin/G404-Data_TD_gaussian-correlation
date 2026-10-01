Ce cours poursuit la comparaison des groupes **A et B**.

L'objectif principal est de comprendre et d'interpréter les outils statistiques. Les données et les graphiques déjà calculés sont fournis afin de ne pas dépendre du temps d'exécution du pipeline.

## Parcours conseillé

1. Utilisez [purchase_transactions.csv](data/purchase_transactions.csv) pour créer les trois nuages de points demandés. 
> *Cf.* [ici](#exporter-le-dataset), pour créer les data et [ici](#exécuter-le-projet) pour éxecuter tout le projet

2. Après votre première version, comparez votre code avec le [notebook 00_reference](notebooks/00_reference_nuages_xy_FR.ipynb).
3. Utilisez ensuite le [notebook 01_age_montant_association](notebooks/01_age_montant_association_FR.ipynb).
4. Terminez avec le [notebook convergence_moyenne_ecart_type](notebooks/02_convergence_moyenne_ecart_type_FR.ipynb).

Le dossier `output/` contient déjà les graphiques en PNG et HTML. Relancer le pipeline est facultatif.

## Présentation du projet

Un lanceur, un paquet local et des tests qui importent la même classe.

Le paquet `analysis` ajoute ensuite `descriptive.py` et `inference.py` pour décrire A et B et comparer leurs moyennes.

## Cinq colonnes, cinq fonctions

La nouvelle simulation conserve une structure fonctionnelle en cinq étapes :

```python
from distributions.age import add_age
from distributions.purchase_dataset import (
    add_checkout_wait_minutes,
    add_distance_to_store_km,
    add_purchase_amount,
    make_groups,
)

groups = make_groups(n=2000, weight_a=0.6)
with_age = add_age(groups, seed=405)
samples = add_purchase_amount(with_age, seed=406)
with_wait = add_checkout_wait_minutes(samples, seed=407)
complete_samples = add_distance_to_store_km(with_wait, seed=408)
```

Les cinq fonctions ont des rôles distincts :

- `make_groups()` crée 1 200 lignes A et 800 lignes B ;
- `add_age()` renvoie une copie et ajoute des âges centrés vers 40 ans dans A et 48 ans dans B ;
- `add_purchase_amount()` renvoie une nouvelle copie et applique la règle `round(baseline + age_effect * age + group_effect * is_group_b + noise, 1)` ;
- `add_checkout_wait_minutes()` ajoute un temps d'attente tiré avec la même loi pour toutes les transactions, sans lire les autres colonnes ;
- `add_distance_to_store_km()` utilise seulement `group` pour choisir une distribution de distance différente dans A et B.

Les colonnes finales sont :

- `group`
- `age`
- `purchase_amount`
- `checkout_wait_minutes`
- `distance_to_store_km`

Avec les paramètres du cours :

- `baseline=80`
- `age_effect=0.5`
- `group_effect=2`
- `noise_std=8`

L'âge explique ainsi une partie de l'écart de montant A/B, tandis qu'un petit effet de groupe reste présent à âge égal.

Le lanceur `compare_age_association.py` affiche les moyennes, la décomposition connue par construction et les corrélations dans A+B, A et B pour les trois variables quantitatives.

### Exporter le dataset

Le lanceur export_purchase_dataset.py exécute les cinq fonctions dans le même ordre, vérifie le contrat `2000 × 5` et écrit le dataset complet sans colonne d'index :

```bash
python export_purchase_dataset.py
```

Dans le dépôt du cours, cette commande reconstruit par défaut :

```text
data/purchase_transactions.csv
```

Le script est conçu pour être lancé directement dans VS Code et ne lit aucun argument de ligne de commande.

Le CSV utilise :

- une virgule comme séparateur ;
- un point comme séparateur décimal ;
- une ligne d'en-tête.

Les montants, calculés par incréments de 0,10 €, sont écrits avec deux décimales, par exemple `123.40`.

Après lecture, pandas peut afficher `123.4`, ce qui représente la même valeur numérique.

Le CSV est remis seul avant ce projet afin que les étudiants produisent leurs nuages XY sans lire le mécanisme de génération. La copie complète distribuée ensuite contient également le CSV et les exports graphiques déjà générés.

### Produire les nuages XY

Le lanceur run_purchase_xy_plots.py réutilise le dataset canonique et produit les trois nuages demandés, puis trois vues de débrief avec les droites d'ajustement linéaire A+B, A et B :

```bash
python run_purchase_xy_plots.py
```

Les figures utilisent les 2 000 lignes et exactement la même échelle verticale.

Chaque observation est représentée par un cercle :

- de 12 px ;
- à centre transparent ;
- avec un bord opaque de 2,5 px ;
- bleu marine pour A ;
- orange soutenu pour B.

Les trois figures dont le nom se termine par `_with_linear_fits` ajoutent :

- une droite A+B noire pointillée ;
- une droite pour A dans sa couleur ;
- une droite pour B dans sa couleur.

Chaque droite A ou B est limitée à la plage X de son propre groupe.

### Étudier la convergence

Le lanceur run_convergence.py produit cinq figures Plotly consacrées à la stabilisation des statistiques descriptives :

1. convergence de la moyenne avec `seed=406` ;
2. convergence de l'écart-type avec `seed=406` ;
3. convergence de la moyenne avec `seed=407` ;
4. cinq trajectoires de moyenne obtenues avec les seeds 406 à 410 ;
5. huit trajectoires de `avg(B) - avg(A)` avec `mu(A)=mu(B)=103`, `sigma(A)=18`, `sigma(B)=24` et 60 % de A.

Chaque trajectoire utilise des préfixes emboîtés et affiche 20 tailles de `n` entre 10 et 5 000.

```bash
python run_convergence.py
```

Les cinq PNG et les cinq HTML sont écrits dans `output/`.

## Interprétation de la simulation

Dans la narration :

- A représente un magasin de centre-ville ;
- B représente un magasin périurbain.

Cette convention appartient uniquement à la simulation.

L'attente ne possède aucune association programmée avec le montant.

La distance n'entre pas dans la formule du montant, mais une association globale peut apparaître parce que `group` déplace à la fois la distance et le montant.

## Organisation du projet

```text
gaussian_project_solution/
├── .gitignore
├── run_gaussian.py
├── run_sampling_variation.py
├── compare_processes.py
├── compare_age_association.py
├── export_purchase_dataset.py
├── run_convergence.py
├── run_purchase_xy_plots.py
├── data/                 # CSV générés, ignorés par Git
├── output/               # PNG et HTML générés, ignorés par Git
├── distributions/
│   ├── __init__.py
│   ├── age.py
│   ├── gaussian_mixture.py
│   ├── gaussian_plots.py
│   └── purchase_dataset.py
├── analysis/
│   ├── __init__.py
│   ├── association.py
│   ├── convergence.py
│   ├── descriptive.py
│   ├── inference.py
│   ├── inference_plots.py
│   ├── purchase_plots.py
│   └── sampling_variation.py
├── tests/
│   ├── test_age_association.py
│   ├── test_convergence.py
│   ├── test_gaussian_mixture.py
│   └── test_analysis.py
├── requirements.txt
└── README.md
```

## Imports et séparation des responsabilités

Le lanceur et les tests utilisent le même import :

```python
from distributions.gaussian_mixture import GaussianMixture
from distributions.gaussian_plots import plot_components, plot_dashboard
```

Dans le paquet, `from . import DEFAULT_SEED` lit la constante de `__init__.py`.

L'import des modules définit le code sans lancer de simulation ni afficher de figure.

Les responsabilités sont séparées :

- `gaussian_mixture.py` conserve les paramètres et produit les valeurs ;
- `gaussian_plots.py` lit l'objet échantillonné et construit les figures ;
- `sampling_variation.py` calcule les différences simulées sous H₀ ;
- `inference_plots.py` construit la figure correspondante ;
- les lanceurs décident d'afficher ou d'exporter les résultats.

Les contrôles rapides ne démarrent que lorsque le module concerné est choisi comme point d'entrée.

## Exécuter le projet

Depuis **la racine de ce dossier**, avec le Python du cours :

```bash
python -m pip install -r requirements.txt

python run_gaussian.py
python run_sampling_variation.py
python compare_processes.py
python compare_age_association.py
python export_purchase_dataset.py
python run_purchase_xy_plots.py
python run_convergence.py

python -m unittest discover -s tests -v
# lance les trente-trois tests.
```

Lancez les tests depuis la racine du projet.

Le paquet local est alors accessible sans réglage supplémentaire de `PYTHONPATH`. Aucun autre dossier du cours n'est nécessaire.

### Fichiers générés par `run_sampling_variation.py`

Le script conserve la distribution Plotly interactive d'échantillonnage sous H₀ étudiée le 29 septembre.

À chaque exécution, il écrit dans `output/` :

- `sampling_variation.png`
- `sampling_variation.html`

### Fichiers générés par `run_gaussian.py`

`run_gaussian.py` construit les six figures Plotly à partir du même échantillon et écrit chaque figure en PNG et en HTML :

- `gaussian_components.*`
- `gaussian_mixture.*`
- `gaussian_box.*`
- `gaussian_ecdf.*`
- `gaussian_violin.*`
- `gaussian_dashboard.*`

L'astérisque représente les deux extensions `.png` et `.html`, soit douze fichiers.

L'exécution directe de `distributions/gaussian_plots.py` produit les mêmes exports. Une nouvelle exécution remplace les fichiers portant les mêmes noms.

L'option suivante évite l'ouverture du tableau de bord sans désactiver les exports :

```bash
python run_gaussian.py --no-show
```

L'export statique utilise Kaleido et un navigateur Chrome ou Chromium installé.

L'option `--no-show` évite seulement l'ouverture du navigateur ; les fichiers sont tout de même exportés. `--output` et `--html-output` permettent de choisir d'autres chemins lorsque c'est nécessaire.

### Fichiers générés par `run_purchase_xy_plots.py`

Le script écrit les trois couples PNG/HTML suivants :

- `checkout_wait_minutes_vs_purchase_amount.png` et `.html`
- `distance_to_store_km_vs_purchase_amount.png` et `.html`
- `age_vs_purchase_amount.png` et `.html`

Les trois mêmes noms complétés par `_with_linear_fits` contiennent les droites A+B, A et B utilisées dans le débrief.

Les premiers nuages restent entièrement bruts. Les droites des vues de débrief sont descriptives : elles n'ajoutent ni coefficient, ni test, ni explication causale du mécanisme.

### Fichiers générés par `run_convergence.py`

Le script écrit cinq couples PNG/HTML :

- `sample_mean_convergence.png` et `.html` (`seed=406`)
- `sample_std_convergence.png` et `.html` (`seed=406`)
- `sample_mean_convergence_seed_407.png` et `.html`
- `sample_mean_convergence_multiple_seeds.png` et `.html` (seeds 406 à 410)
- `sample_mean_difference_convergence_multiple_seeds.png` et `.html` (`avg(B) - avg(A)`, seeds 1404 à 1411)


## Suivre les données

`run_gaussian.py` choisit les paramètres et crée un modèle.

La méthode `sample()` :

1. tire A puis B avec un même générateur ;
2. appelle la méthode privée `_combine_components()` pour construire le dataframe ;
3. conserve le dataframe dans `self.samples` ;
4. renvoie ce même objet.

Les fonctions de `gaussian_plots.py` reçoivent l'objet, lisent ce dataframe et renvoient les figures. Le lanceur les exporte dans `output/` et choisit d'afficher le tableau de bord.

### Comprendre `samples` et les annotations de type

Avant `sample()`, `model.samples` peut légitimement valoir `None`.

L'annotation :

```python
pd.DataFrame | None
```

décrit ces deux états possibles.

En revanche, la signature :

```python
sample() -> pd.DataFrame
```

garantit le type de la valeur renvoyée par cet appel.

Le lanceur la nomme `samples`, et Pylance peut alors proposer directement les méthodes comme `groupby()`.

Après l'appel, `samples` et `model.samples` désignent le même dataframe.

Ces annotations aident l'éditeur mais ne contrôlent pas l'exécution. `_require_samples()` reste la vérification au moment de construire une figure.

### Paramètres de l'échantillon

L'échantillon utilise :

- **1 200 observations A et 800 observations B** ;
- des moyennes théoriques de 100 pour A et 106 pour B ;
- des écarts-types de 18 pour A et 24 pour B ;
- 60 % de A ;
- 2 000 observations au total ;
- `seed=404`.

Le terminal décrit cet échantillon et donne les appels des cinq vues autonomes.

Une seule fenêtre affiche le **tableau de bord du modèle** avec :

- A et B séparés ;
- le mélange ;
- les violons ;
- les proportions cumulées.

Les données conservées dans l'objet restent inchangées.

Un test séparé crée deux instances et vérifie qu'elles ne partagent ni leur configuration, ni leur dataframe mutable.

Les mêmes paramètres et la même seed reproduisent les valeurs dans le même environnement.

## Lire les figures sans les confondre

### Histogrammes des groupes

`plot_components()` superpose deux **histogrammes observés, normalisés en densité**.

A et B utilisent exactement les mêmes intervalles d'histogramme.

Les courbes sont les lois normales théoriques calculées avec :

- `mean_a`
- `std_a`
- `mean_b`
- `std_b`

Elles ne sont pas ajustées aux observations.

### Mélange

`plot_mixture()` réunit toutes les observations dans un histogramme.

Les courbes A et B sont les **contributions théoriques pondérées** par `weight_a` et `1 - weight_a`.

Leur somme point par point donne la densité théorique du mélange.

Une contribution pondérée n'est pas, à elle seule, une densité normalisée à 1.

### Boîte et ECDF

`plot_box()` compare :

- le centre ;
- la dispersion ;
- les valeurs atypiques ;

pour A, B et A∪B.

`plot_ecdf()` trace la fonction de répartition empirique (ECDF). Pour chaque valeur `x`, elle donne la proportion d'observations inférieures ou égales à `x`.

### Violons

`plot_violin()` montre une estimation lissée de la densité locale.

Sa largeur ne représente pas l'effectif du groupe. La boîte et la ligne moyenne à l'intérieur aident à conserver des repères.

### Tableau de bord

`plot_dashboard()` réunit les quatre lectures complémentaires les plus utiles dans une même fenêtre.

Les couleurs de A et B ainsi que les styles de lignes restent identiques d'un panneau à l'autre.

### Poids théorique et proportion observée

Le poids théorique vient de la configuration du modèle.

Pour de très petits échantillons, `int(n * weight_a)` arrondit l'effectif de A. La proportion observée peut alors différer légèrement du poids du modèle.

Avec 2 000 observations et `weight_a=0.6`, les deux valent exactement 60 %.

## Comparer A et B

### Description des groupes

`analysis/descriptive.py` commence par les observations :

- effectif ;
- moyenne ;
- médiane ;
- écart-type ;
- quartiles ;
- intervalle interquartile de chaque groupe.

`mean_comparison()` calcule ensuite :

```text
avg(B) − avg(A)
```

où `avg(G)` désigne la moyenne observée du groupe G.

Une valeur positive indique donc que la moyenne observée de B est supérieure à celle de A.

La fonction calcule également une différence standardisée.

### Test de Welch

`analysis/inference.py` applique le test de Welch à deux groupes indépendants.

Ce test n'impose pas des variances égales.

Il renvoie :

- la différence observée ;
- un intervalle de confiance ;
- la statistique de test ;
- les degrés de liberté ;
- la valeur p.

L'intervalle et la valeur p quantifient l'incertitude statistique. Ils ne décident pas si l'écart est important dans le contexte étudié.

Avec la graine 404 :

- la différence observée `avg(B) − avg(A)` vaut environ **5.862** ;
- l'intervalle de confiance de `μ₂ − μ₁` va approximativement de **3.877 à 7.846** ;
- la différence standardisée vaut environ **0.281**.

Le grand nombre d'observations rend l'écart statistiquement net, mais son importance pratique reste à interpréter.

### Variation d'échantillonnage sous H₀

`run_sampling_variation.py` construit un modèle nul dans lequel les deux moyennes valent 103, tout en conservant :

- `sigma(A)=18` ;
- `sigma(B)=24` ;
- 2 000 observations ;
- une répartition 60 % / 40 %.

Il appelle `GaussianMixture.sample()` avec 5 000 seeds distinctes et conserve une valeur `avg(B) − avg(A)` par répétition.

La figure obtenue représente une **distribution de différences de moyennes sous H₀**.

Elle ne représente :

- ni la distribution de la statistique T de Welch ;
- ni une valeur p simulée.

## Limites de l'exemple

Le dataframe de l'extension contient cinq colonnes, de `group` à `distance_to_store_km`.

Deux groupes ne sont pas deux variables mesurées.

La simulation fixe les mécanismes à l'avance : elle sert à comprendre une méthode, pas à découvrir une cause inconnue.

Dire que l'âge explique une partie de l'écart est justifié ici par la règle de génération connue, **pas par la corrélation seule**.

Dans une étude réelle, une différence entre deux groupes peut également venir :

- du plan d'échantillonnage ;
- d'un biais de mesure ;
- d'une autre différence entre les groupes.

Le test de Welch répond à une question sur les moyennes de deux groupes indépendants. Il ne compare pas toutes les caractéristiques des distributions et ne démontre pas qu'une intervention a causé l'écart.