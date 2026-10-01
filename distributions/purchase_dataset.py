"""Construire les cinq colonnes du dataset d'achat simulé."""

import numpy as np
import pandas as pd

from . import DEFAULT_SEED


DEFAULT_PURCHASE_SEED = DEFAULT_SEED + 2
DEFAULT_WAIT_SEED = DEFAULT_SEED + 3
DEFAULT_DISTANCE_SEED = DEFAULT_SEED + 4
BASELINE = 80.0
AGE_EFFECT = 0.5
GROUP_EFFECT = 2.0
NOISE_STD = 8.0
WAIT_MEAN = 6.0
WAIT_STD = 2.0
DISTANCE_MEANS = {"A": 3.0, "B": 10.0}
DISTANCE_STDS = {"A": 1.5, "B": 3.0}


def make_groups(n: int = 2000, weight_a: float = 0.6) -> pd.DataFrame:
    """Créer les lignes A/B avant l'ajout de l'âge et du montant."""
    if n <= 0:
        raise ValueError("n doit être strictement positif.")
    if not 0 < weight_a < 1:
        raise ValueError("weight_a doit être strictement compris entre 0 et 1.")

    n_a = int(n * weight_a)
    groups = np.array(["A"] * n_a + ["B"] * (n - n_a), dtype=object)
    return pd.DataFrame({"group": groups})


def add_purchase_amount(
    samples: pd.DataFrame,
    baseline: float = BASELINE,
    age_effect: float = AGE_EFFECT,
    group_effect: float = GROUP_EFFECT,
    noise_std: float = NOISE_STD,
    seed: int = DEFAULT_PURCHASE_SEED,
) -> pd.DataFrame:
    """Retourner une copie avec `purchase_amount` calculé selon la règle.

    raw_purchase_amount = baseline + age_effect * age
                          + group_effect * is_group_b + noise
    purchase_amount = round(raw_purchase_amount, 1)
    """
    required = {"group", "age"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if not samples["group"].isin(["A", "B"]).all():
        raise ValueError("La colonne group doit contenir uniquement A et B.")
    if samples["age"].isna().any():
        raise ValueError("La colonne age contient des valeurs manquantes.")
    if noise_std < 0:
        raise ValueError("noise_std doit être positif ou nul.")

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, len(samples))
    is_group_b = samples["group"].eq("B").to_numpy(dtype=float)
    expected_purchase = (
        baseline
        + age_effect * samples["age"].to_numpy(dtype=float)
        + group_effect * is_group_b
    )
    raw_purchase_amount = expected_purchase + noise

    enriched = samples.copy()
    enriched["purchase_amount"] = np.round(raw_purchase_amount, 1)
    return enriched


def add_checkout_wait_minutes(
    samples: pd.DataFrame,
    seed: int = DEFAULT_WAIT_SEED,
) -> pd.DataFrame:
    """Retourner une copie avec un temps d'attente sans association programmée.

    La même loi est utilisée pour toutes les lignes. La fonction ne lit aucune
    colonne existante pour générer `checkout_wait_minutes`.
    """
    rng = np.random.default_rng(seed)
    wait = rng.normal(WAIT_MEAN, WAIT_STD, len(samples))
    wait = np.round(np.clip(wait, 1.0, 15.0), 1)

    enriched = samples.copy()
    enriched["checkout_wait_minutes"] = wait
    return enriched


def add_distance_to_store_km(
    samples: pd.DataFrame,
    seed: int = DEFAULT_DISTANCE_SEED,
) -> pd.DataFrame:
    """Retourner une copie avec une distance tirée selon le groupe.

    `group` choisit la distribution de distance. La fonction n'utilise ni
    `age` ni `purchase_amount` pour générer `distance_to_store_km`.
    """
    required = {"group"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if not samples["group"].isin(DISTANCE_MEANS).all():
        raise ValueError("La colonne group doit contenir uniquement A et B.")

    means = samples["group"].map(DISTANCE_MEANS).to_numpy()
    stds = samples["group"].map(DISTANCE_STDS).to_numpy()
    rng = np.random.default_rng(seed)
    distance = rng.normal(means, stds)
    distance = np.round(np.clip(distance, 0.2, None), 1)

    enriched = samples.copy()
    enriched["distance_to_store_km"] = distance
    return enriched
