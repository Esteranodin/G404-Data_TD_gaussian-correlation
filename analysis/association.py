"""Mesurer l'association entre une variable quantitative et le montant."""

import numpy as np
import pandas as pd
from scipy import stats

from .inference import welch_mean_test

BASE_COLUMNS = {"purchase_amount", "group"}


def pearson_association(
    samples: pd.DataFrame,
    group: str | None = None,
    confidence_level: float = 0.95,
    predictor: str = "age",
) -> dict[str, float | str]:
    """Relier une variable quantitative à `purchase_amount` avec Pearson.

    Args:
        samples: Dataframe contenant `group`, `purchase_amount` et la variable.
        group: Groupe A ou B, ou None pour réunir les deux.
        confidence_level: Niveau de confiance de l'intervalle pour rho.
        predictor: Colonne quantitative placée sur l'axe X.
    """
    required = BASE_COLUMNS | {predictor}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")

    selected = (
        samples
        if group is None
        else samples.loc[samples["group"] == group]
    )
    if selected.empty:
        raise ValueError(f"Groupe inconnu ou vide : {group}")
    if selected[[predictor, "purchase_amount"]].isna().any().any():
        raise ValueError("Les paires analysées contiennent des valeurs manquantes.")

    predictor_values = selected[predictor].to_numpy(dtype=float)
    purchase_amount = selected["purchase_amount"].to_numpy(dtype=float)
    result = stats.pearsonr(predictor_values, purchase_amount)
    interval = result.confidence_interval(confidence_level)
    return {
        "scope": "A+B" if group is None else group,
        "count": float(len(selected)),
        "covariance": float(
            np.cov(predictor_values, purchase_amount, ddof=1)[0, 1]
        ),
        "correlation": float(result.statistic),
        "confidence_low": float(interval.low),
        "confidence_high": float(interval.high),
        "p_value": float(result.pvalue),
    }


def compare_correlations(
    samples: pd.DataFrame,
    predictor: str = "age",
) -> pd.DataFrame:
    """Comparer les corrélations dans A+B, A et B pour une variable."""
    results = [pearson_association(samples, predictor=predictor)]
    results.extend(
        pearson_association(samples, group, predictor=predictor)
        for group in ("A", "B")
    )
    return pd.DataFrame(results).set_index("scope")


def purchase_mean_test(
    samples: pd.DataFrame,
    confidence_level: float = 0.95,
) -> dict[str, float | str]:
    """Réutiliser le test de Welch pour `purchase_amount` selon `group`."""
    missing = BASE_COLUMNS.difference(samples.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    adapted = samples.rename(
        columns={"group": "component", "purchase_amount": "value"}
    )
    return welch_mean_test(adapted, confidence_level=confidence_level)


def purchase_gap_decomposition(
    samples: pd.DataFrame,
    age_effect: float,
    group_effect: float,
) -> dict[str, float]:
    """Décomposer l'écart B-A, avec le résidu du bruit et de l'arrondi."""
    required = BASE_COLUMNS | {"age"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    means = samples.groupby("group")[["age", "purchase_amount"]].mean()
    if not {"A", "B"}.issubset(means.index):
        raise ValueError("Les groupes A et B sont nécessaires.")

    age_gap = float(means.loc["B", "age"] - means.loc["A", "age"])
    observed_gap = float(
        means.loc["B", "purchase_amount"]
        - means.loc["A", "purchase_amount"]
    )
    age_profile_part = age_effect * age_gap
    expected_gap = age_profile_part + group_effect
    return {
        "age_gap": age_gap,
        "observed_purchase_gap": observed_gap,
        "age_profile_part": age_profile_part,
        "remaining_group_part": group_effect,
        "expected_purchase_gap": expected_gap,
        "observed_residual_gap": observed_gap - expected_gap,
    }
