"""Ajouter un âge simulé à partir du groupe A/B."""

import numpy as np
import pandas as pd

from . import DEFAULT_SEED

DEFAULT_AGE_SEED = DEFAULT_SEED + 1
AGE_MEANS = {"A": 40.0, "B": 48.0}
AGE_STD = 8.0


def add_age(
    samples: pd.DataFrame,
    seed: int = DEFAULT_AGE_SEED,
) -> pd.DataFrame:
    """Retourner une copie avec un âge tiré selon le groupe.

    L'âge est centré autour de 40 ans dans A et de 48 ans dans B. La fonction
    conserve les colonnes existantes et ne les modifie pas.

    Args:
        samples: Dataframe contenant la colonne `group`.
        seed: Graine du générateur utilisé pour les âges.
    """
    required = {"group"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if not samples["group"].isin(AGE_MEANS).all():
        raise ValueError("La colonne group doit contenir uniquement A et B.")

    means = samples["group"].map(AGE_MEANS).to_numpy()
    rng = np.random.default_rng(seed)
    ages = np.rint(rng.normal(means, AGE_STD)).clip(18, 80).astype(int)

    enriched = samples.copy()
    enriched["age"] = ages
    return enriched
