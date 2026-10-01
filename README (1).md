# Deux groupes gaussiens — projet de référence et comparaison

Cette version complète est remise après la présentation des réalisations. Elle
reprend la structure du projet Rectangle du 22 septembre : un lanceur, un paquet
local et des tests qui importent la même classe. Le paquet `analysis` ajoute
ensuite `descriptive.py` et `inference.py` pour décrire A et B et comparer leurs
moyennes.

## Extension du 1er octobre : cinq colonnes, cinq fonctions

Le projet et la classe `GaussianMixture` du 29 septembre restent disponibles
et inchangés. La nouvelle simulation conserve une structure fonctionnelle en
cinq étapes :

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

- `make_groups()` crée 1 200 lignes A et 800 lignes B ;
- `add_age()` renvoie une copie et ajoute des âges centrés vers 40 ans dans A
  et 48 ans dans B ;
- `add_purchase_amount()` renvoie une nouvelle copie et applique la règle
  `round(baseline + age_effect * age + group_effect * is_group_b + noise, 1)` ;
- `add_checkout_wait_minutes()` ajoute un temps d'attente tiré avec la même loi
  pour toutes les transactions, sans lire les autres colonnes ;
- `add_distance_to_store_km()` utilise seulement `group` pour choisir une
  distribution de distance différente dans A et B ;
- les colonnes finales sont `group`, `age`, `purchase_amount`,
  `checkout_wait_minutes`, `distance_to_store_km`.

Avec les paramètres du cours, `baseline=80`, `age_effect=0.5`,
`group_effect=2` et `noise_std=8`. L'âge explique ainsi une partie de l'écart
de montant A/B, tandis qu'un petit effet de groupe reste présent à âge égal.
Le lanceur [`compare_age_association.py`](compare_age_association.py) affiche
les moyennes, la décomposition connue par construction et les corrélations
dans A+B, A et B pour les trois variables quantitatives.

Le lanceur [`export_purchase_dataset.py`](export_purchase_dataset.py) exécute
les cinq fonctions dans le même ordre, vérifie le contrat `2000 × 5` et écrit
le dataset complet sans colonne d'index :

```bash
python export_purchase_dataset.py
```

Dans le dépôt du cours, cette commande reconstruit par défaut
`data/purchase_transactions.csv`. Le script est conçu pour être lancé
directement dans VS Code et ne lit aucun argument de ligne de commande.

Le CSV utilise une virgule comme séparateur, un point décimal et une ligne
d'en-tête. Les montants, calculés par incréments de 0,10 €, y sont écrits avec
deux décimales : `123.40`. Après lecture, pandas peut afficher `123.4`, qui est
la même valeur numérique. Le CSV est remis seul avant ce projet afin que les
étudiants produisent leurs nuages XY sans lire le mécanisme de génération.
La copie complète distribuée ensuite contient aussi ce CSV et les exports
graphiques déjà générés dans `output/` : relancer le pipeline est facultatif.

Le lanceur [`run_purchase_xy_plots.py`](run_purchase_xy_plots.py) réutilise ce
dataset canonique et produit les trois nuages demandés, puis trois vues de
débrief avec les droites d'ajustement linéaire A+B, A et B :

```bash
python run_purchase_xy_plots.py
```

Les six PNG et les six HTML sont écrits dans `output/`. Les figures utilisent
les 2 000 lignes et exactement la même échelle verticale. Chaque observation
est représentée par un cercle de 12 px à centre transparent et bord opaque de
2,5 px : bleu marine pour A, orange soutenu pour B. Les trois noms terminés par
`_with_linear_fits` ajoutent une droite A+B noire pointillée et les droites A
et B dans leur couleur, chacune limitée à sa propre plage X.

Le lanceur [`run_convergence.py`](run_convergence.py) produit cinq figures
Plotly consacrées à la stabilisation des statistiques descriptives : moyenne et
écart-type avec `seed=406`, moyenne avec `seed=407`, puis cinq trajectoires de
moyenne obtenues avec les seeds 406 à 410. Une cinquième figure relie cette
idée au projet : huit trajectoires de `avg(B) - avg(A)` avec
`mu(A)=mu(B)=103`, `sigma(A)=18`, `sigma(B)=24` et 60 % de A. Chaque
trajectoire utilise des préfixes emboîtés et affiche 20 tailles de `n` entre
10 et 5 000 :

```bash
python run_convergence.py
```

Les cinq PNG et les cinq HTML sont écrits dans `output/`.

Dans la narration, A représente un magasin de centre-ville et B un magasin
périurbain. Cette convention appartient à la simulation. L'attente ne possède
aucune association programmée avec le montant. La distance n'entre pas dans la
formule du montant, mais une association globale peut apparaître parce que
`group` déplace à la fois la distance et le montant.

## Organisation

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

| Projet Rectangle | Projet gaussien | Rôle |
| --- | --- | --- |
| `run_rectangle.py` | [run_gaussian.py](run_gaussian.py) | Choisir les paramètres, créer le modèle, exporter les six figures et afficher le tableau de bord |
| `geometry/__init__.py` | [distributions/__init__.py](distributions/__init__.py) | Définir une constante partagée : ici `DEFAULT_SEED` |
| `geometry/rectangle.py` | [distributions/gaussian_mixture.py](distributions/gaussian_mixture.py) | Définir la classe, tirer les valeurs et assembler le dataframe |
| — | [distributions/gaussian_plots.py](distributions/gaussian_plots.py) | Construire les figures à partir d'un objet déjà échantillonné |
| — | [analysis/sampling_variation.py](analysis/sampling_variation.py) | Répéter les tirages sous H₀ et conserver `avg(B) − avg(A)` |
| — | [analysis/inference_plots.py](analysis/inference_plots.py) | Construire l'histogramme Plotly des différences simulées |
| — | [run_sampling_variation.py](run_sampling_variation.py) | Reproduire la figure d'échantillonnage sous H₀ conservée du 29 septembre |
| `tests/test_rectangle.py` | [tests/test_gaussian_mixture.py](tests/test_gaussian_mixture.py) | Vérifier la classe utilisée par le lanceur |
| — | [analysis/descriptive.py](analysis/descriptive.py) | Décrire A et B et mesurer leur écart observé |
| — | [analysis/inference.py](analysis/inference.py) | Estimer et tester la différence de moyennes |
| — | [compare_processes.py](compare_processes.py) | Lancer la comparaison descriptive et formelle |
| — | [tests/test_analysis.py](tests/test_analysis.py) | Vérifier les nouveaux calculs |
| — | [distributions/age.py](distributions/age.py) | Ajouter `age` à une copie des groupes A/B |
| — | [distributions/purchase_dataset.py](distributions/purchase_dataset.py) | Créer les groupes puis ajouter `purchase_amount` |
| — | [analysis/association.py](analysis/association.py) | Mesurer et décomposer l'association âge–montant |
| — | [analysis/convergence.py](analysis/convergence.py) | Calculer les trajectoires emboîtées et construire les cinq figures de convergence |
| — | [analysis/purchase_plots.py](analysis/purchase_plots.py) | Construire les trois nuages XY bruts et leurs vues avec droites, avec une échelle Y commune |
| — | [export_purchase_dataset.py](export_purchase_dataset.py) | Construire, contrôler et exporter les cinq colonnes en CSV |
| — | [run_purchase_xy_plots.py](run_purchase_xy_plots.py) | Exécuter le pipeline canonique et exporter six vues en PNG/HTML |
| — | [run_convergence.py](run_convergence.py) | Exporter les cinq vues de convergence en PNG/HTML |

Le lanceur et les tests utilisent le même import :

```python
from distributions.gaussian_mixture import GaussianMixture
from distributions.gaussian_plots import plot_components, plot_dashboard
```

Dans le paquet, `from . import DEFAULT_SEED` lit la constante de `__init__.py`.
L'import des modules définit le code sans lancer de simulation ni afficher de
figure. `gaussian_mixture.py` conserve les paramètres et produit les valeurs ;
`gaussian_plots.py` lit l'objet échantillonné et construit les figures. Leurs
contrôles rapides ne démarrent que si le module concerné est choisi comme point
d'entrée. La simulation sous H₀ et sa figure Plotly suivent la même
séparation : un module calcule les différences, un autre construit la figure et
le lanceur choisit de l'afficher ou de l'exporter.

## Exécuter

Depuis **ce dossier**, avec le Python du cours :

```bash
python -m pip install -r requirements.txt
python run_gaussian.py
python run_sampling_variation.py
python -m unittest discover -s tests -v
python compare_processes.py
python compare_age_association.py
python export_purchase_dataset.py
python run_purchase_xy_plots.py
python run_convergence.py
```

Lancez les tests depuis la racine du projet.
Le paquet local est alors accessible sans réglage de `PYTHONPATH`.
Aucun autre dossier du cours n'est nécessaire.

`run_sampling_variation.py` conserve la distribution Plotly interactive
d'échantillonnage sous H₀ étudiée le 29 septembre. À chaque exécution, il écrit
les deux fichiers suivants dans le dossier ignoré `output/` :

- `sampling_variation.png` ;
- `sampling_variation.html`.

`run_gaussian.py` construit les six figures Plotly à partir du même échantillon
et écrit également chaque figure en PNG et en HTML dans `output/` :

- `gaussian_components.*` ;
- `gaussian_mixture.*` ;
- `gaussian_box.*` ;
- `gaussian_ecdf.*` ;
- `gaussian_violin.*` ;
- `gaussian_dashboard.*`.

L'astérisque représente les deux extensions `.png` et `.html`, soit douze
fichiers. L'exécution directe de `distributions/gaussian_plots.py` produit les
mêmes exports. Une nouvelle exécution remplace les fichiers portant les mêmes
noms. L'option `python run_gaussian.py --no-show` évite l'ouverture du tableau
de bord sans désactiver les exports.

L'export statique utilise Kaleido et un navigateur Chrome ou Chromium installé.
L'option `--no-show` évite seulement l'ouverture du navigateur ; les deux
fichiers sont tout de même exportés. `--output` et `--html-output` permettent
encore de choisir d'autres chemins lorsque c'est nécessaire.

`run_purchase_xy_plots.py` écrit douze fichiers supplémentaires dans `output/` :

- `checkout_wait_minutes_vs_purchase_amount.png` et `.html` ;
- `distance_to_store_km_vs_purchase_amount.png` et `.html` ;
- `age_vs_purchase_amount.png` et `.html`.

Les trois mêmes noms complétés par `_with_linear_fits` contiennent les
droites A+B, A et B utilisées dans le débrief.

Les premiers nuages restent entièrement bruts. Les droites des vues de débrief
sont descriptives : elles n'ajoutent ni coefficient, ni test, ni explication
causale du mécanisme.

`run_convergence.py` écrit dix fichiers supplémentaires dans `output/` :

- `sample_mean_convergence.png` et `.html` (`seed=406`) ;
- `sample_std_convergence.png` et `.html` (`seed=406`) ;
- `sample_mean_convergence_seed_407.png` et `.html` ;
- `sample_mean_convergence_multiple_seeds.png` et `.html` (seeds 406 à 410) ;
- `sample_mean_difference_convergence_multiple_seeds.png` et `.html`
  (`avg(B) - avg(A)`, seeds 1404 à 1411).

## Exécuter avec Quick Run dans l'espace de travail g404

Dans la version du cours située sous
`courses/2026-10-01/lab/gaussian_project_solution/`, ouvrez l'un des treize
fichiers exécutables, puis choisissez `Python: Run selected file` dans Quick
Run :

- `run_gaussian.py` décrit l'échantillon, écrit les six PNG et les six HTML
  dans `output/`, puis affiche le tableau de bord en une seule fenêtre ;
- `run_sampling_variation.py` répète les tirages sous H₀ et affiche la figure
  Plotly conservée du 29 septembre, puis écrit le PNG et le HTML dans
  `output/` ;
- `compare_processes.py` affiche la comparaison statistique ;
- `compare_age_association.py` construit les cinq colonnes par appels
  successifs et compare les corrélations globales et par groupe ;
- `export_purchase_dataset.py` reconstruit le dataset et écrit le CSV dans
  `data/` ;
- `run_purchase_xy_plots.py` reconstruit le même dataset et écrit les six
  paires PNG/HTML dans `output/` ;
- `run_convergence.py` calcule cinq vues avec 20 tailles de `n` et les écrit
  en PNG/HTML dans `output/` ;
- `distributions/gaussian_mixture.py` vérifie le tirage reproductible, les
  effectifs et les métadonnées conservées ;
- `distributions/gaussian_plots.py` construit les six figures, vérifie leur
  structure, écrit les douze fichiers et affiche uniquement le tableau de bord ;
- `tests/test_gaussian_mixture.py` lance ses dix tests du projet initial et de
  ses vues graphiques ;
- `tests/test_analysis.py` lance ses huit tests de l'extension statistique ;
- `tests/test_age_association.py` lance les douze tests de l'extension.
- `tests/test_convergence.py` lance les trois tests des calculs et exports de
  convergence.

Quick Run reconnaît les deux fichiers comme des modules du paquet et utilise
`python -m distributions.gaussian_mixture` ou
`python -m distributions.gaussian_plots`. Les autres fichiers de
`distributions/` et `analysis/` restent des modules importés : vérifiez-les avec
le fichier de tests correspondant plutôt qu'en les lançant seuls.

Cette option dépend de la tâche configurée dans l'espace de travail g404 : elle
ajoute la racine du projet au chemin des imports. Pour un fichier placé dans
`tests/`, le lanceur ajoute aussi automatiquement le dossier parent de `tests/` ;
les imports locaux restent donc disponibles même si le terminal n'a pas encore
rechargé son `PYTHONPATH`. Quick Run démarre ensuite le point d'entrée
sélectionné. Lorsqu'un bloc `if __name__ == "__main__"` est présent, il lance le
contrôle prévu.
Pour une copie extraite ailleurs, utilisez les commandes du terminal ci-dessus.
Les contrôles rapides facultatifs se lancent depuis la racine du projet avec
`python -m distributions.gaussian_mixture` et
`python -m distributions.gaussian_plots` ; `unittest discover` lance les
trente-trois tests et reste la vérification complète.

## Suivre les données

`run_gaussian.py` choisit les paramètres et crée un modèle. La méthode
`sample()` tire A puis B avec un même générateur, appelle la méthode privée
`_combine_components()` pour construire le dataframe, le conserve dans
`self.samples` et renvoie ce même objet. Les fonctions de `gaussian_plots.py`
reçoivent l'objet, lisent ce dataframe et renvoient les figures ; le lanceur
les exporte toutes dans `output/` et choisit d'afficher le tableau de bord.

Avant `sample()`, `model.samples` peut légitimement valoir `None`. L'annotation
`pd.DataFrame | None` décrit ces deux états possibles. En revanche, la signature
`sample() -> pd.DataFrame` garantit le type de la valeur renvoyée par cet appel :
le lanceur la nomme `samples`, et Pylance peut alors proposer directement les
méthodes comme `groupby()`. Après l'appel, `samples` et `model.samples` désignent
le même dataframe. Ces annotations aident l'éditeur mais ne contrôlent pas
l'exécution ; `_require_samples()` reste la vérification au moment de construire
une figure.

- Échantillon : **1 200 A et 800 B** ; moyennes théoriques 100/106,
  écarts-types 18/24, 60 % de A, 2 000 observations, `seed=404`.
- Le terminal décrit cet échantillon et donne les appels des cinq vues
  autonomes. Une seule fenêtre affiche le **tableau de bord du modèle** :
  A/B séparés, mélange, violons et proportions cumulées. Les données
  conservées dans l'objet restent inchangées.
- Un test séparé crée deux instances et vérifie qu'elles ne partagent ni leur
  configuration, ni leur dataframe mutable.

Les mêmes paramètres et la même seed reproduisent les valeurs dans le même
environnement. Cette version utilise les paramètres valides du cours ; le tirage
précède les graphiques.

## Lire les figures sans les confondre

- `plot_components()` superpose deux **histogrammes observés, normalisés en
  densité**. A et B utilisent exactement les mêmes intervalles d'histogramme.
  Les courbes sont les lois normales théoriques calculées avec `mean_a`,
  `std_a`, `mean_b` et `std_b` ; elles ne sont pas ajustées aux observations.
- `plot_mixture()` réunit toutes les observations dans un histogramme. Les
  courbes A et B sont les **contributions théoriques pondérées** par `weight_a`
  et `1 - weight_a`. Leur somme point par point donne la densité théorique du
  mélange. Une contribution pondérée n'est pas une densité normalisée à 1.
- `plot_box()` est une option autonome qui compare centre, dispersion et valeurs
  atypiques pour A, B et A∪B. `plot_ecdf()` trace la fonction de répartition
  empirique (ECDF) : pour chaque valeur x, elle donne la proportion
  d'observations inférieures ou égales à x.
- `plot_violin()` montre une estimation lissée de la densité locale ; sa largeur
  ne représente pas l'effectif du groupe. La boîte et la ligne moyenne à
  l'intérieur aident à garder des repères.
- `plot_dashboard()` réunit les quatre lectures complémentaires les plus utiles
  dans une même fenêtre. Les couleurs de A et B et les styles de lignes restent
  identiques d'un panneau à l'autre.

Le poids théorique vient de la configuration du modèle. Pour de très petits
échantillons, `int(n * weight_a)` arrondit l'effectif de A : la proportion
observée peut alors différer légèrement du poids du modèle. Avec 2 000
observations et `weight_a=0.6`, les deux valent exactement 60 %.

## Comparer A et B

`analysis/descriptive.py` commence par les observations : effectif, moyenne,
médiane, écart-type, quartiles et intervalle interquartile de chaque groupe.
`mean_comparison()` calcule la différence **avg(B) − avg(A)**, où `avg(G)`
désigne la moyenne observée du groupe G, ainsi qu'une différence standardisée.
Une valeur positive indique
donc que la moyenne observée de B est supérieure à celle de A.

`analysis/inference.py` applique le test de Welch à deux groupes indépendants.
Ce test n'impose pas des variances égales. Il renvoie la différence observée,
un intervalle de confiance, la statistique de test, les degrés de liberté et la
valeur p. L'intervalle et la valeur p quantifient l'incertitude statistique ;
ils ne décident pas si l'écart est important dans le contexte étudié.

Avec la graine 404, la différence observée avg(B) − avg(A) vaut environ 5.862.
L'intervalle de confiance de μ₂ − μ₁ va approximativement de 3.877 à 7.846 et la
différence standardisée vaut environ 0.281. Le grand nombre d'observations rend
l'écart statistiquement net, mais son importance pratique reste à interpréter.

`run_sampling_variation.py` construit ensuite un modèle nul dont les deux
moyennes valent 103, tout en conservant les écarts-types 18 et 24, l'effectif
total 2 000 et la répartition 60 % / 40 %. Il appelle
`GaussianMixture.sample()` avec 5 000 seeds distinctes et conserve une valeur
`avg(B) − avg(A)` par répétition. La figure obtenue est une distribution de
différences de moyennes sous H₀, pas la distribution de la statistique T de
Welch et pas une valeur p simulée.

## Limites de l'exemple

Le dataframe de l'extension contient cinq colonnes, de `group` à
`distance_to_store_km`.
Deux groupes ne sont pas deux variables mesurées. La simulation fixe les
mécanismes à l'avance : elle sert à comprendre une méthode, pas à découvrir une
cause inconnue. Dire que l'âge explique une partie de l'écart est justifié ici
par la règle de génération connue, pas par la corrélation seule.

Dans une étude réelle, une différence entre deux groupes peut aussi venir du
plan d'échantillonnage, d'un biais de mesure ou d'une autre différence entre les
groupes. Le test de Welch répond à une question sur les moyennes de deux groupes
indépendants ; il ne compare pas toutes les caractéristiques des distributions
et ne démontre pas qu'une intervention a causé l'écart.
