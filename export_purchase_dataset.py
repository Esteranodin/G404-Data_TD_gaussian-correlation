"""Exporter le dataset d'achat complet dans un fichier CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from distributions.age import add_age
from distributions.purchase_dataset import (
    add_checkout_wait_minutes,
    add_distance_to_store_km,
    add_purchase_amount,
    make_groups,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OUTPUT = DATA_DIR / "purchase_transactions.csv"
EXPECTED_COLUMNS = [
    "group",
    "age",
    "purchase_amount",
    "checkout_wait_minutes",
    "distance_to_store_km",
]


def build_dataset() -> pd.DataFrame:
    """Construire les cinq colonnes avec les paramètres reproductibles du cours."""
    groups = make_groups(n=2000, weight_a=0.6)
    with_age = add_age(groups, seed=405)
    with_purchase = add_purchase_amount(with_age, seed=406)
    with_wait = add_checkout_wait_minutes(with_purchase, seed=407)
    return add_distance_to_store_km(with_wait, seed=408)


def validate_dataset(samples: pd.DataFrame) -> None:
    """Vérifier le contrat du CSV avant son écriture."""
    if samples.shape != (2000, 5):
        raise ValueError(f"Forme inattendue : {samples.shape}, attendu : (2000, 5).")
    if list(samples.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Colonnes inattendues : {list(samples.columns)}")
    if samples.isna().any().any():
        raise ValueError("Le dataset contient des valeurs manquantes.")
    if samples["group"].value_counts().to_dict() != {"A": 1200, "B": 800}:
        raise ValueError("Les effectifs A/B sont inattendus.")
    if not pd.api.types.is_integer_dtype(samples["age"]):
        raise ValueError("La colonne age doit contenir des entiers.")
    for column in EXPECTED_COLUMNS[2:]:
        if not pd.api.types.is_float_dtype(samples[column]):
            raise ValueError(f"La colonne {column} doit contenir des nombres décimaux.")


def export_dataset(output: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Construire et écrire le dataset avec deux décimales pour le montant."""
    samples = build_dataset()
    validate_dataset(samples)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exported = samples.copy()
    exported["purchase_amount"] = exported["purchase_amount"].map(
        lambda value: f"{value:.2f}"
    )
    exported.to_csv(output, index=False)
    return samples


def main() -> None:
    """Exporter le fichier du cours et afficher son contrat."""
    samples = export_dataset()
    print(f"CSV exporté : {DEFAULT_OUTPUT}")
    print(f"Forme : {samples.shape}")
    print(f"Colonnes : {', '.join(samples.columns)}")


if __name__ == "__main__":
    main()
