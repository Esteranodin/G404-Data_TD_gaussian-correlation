"""Construire le dataset d'achat et décrire son mécanisme."""

from analysis import compare_correlations
from distributions.age import add_age
from distributions.purchase_dataset import (
    AGE_EFFECT,
    GROUP_EFFECT,
    add_checkout_wait_minutes,
    add_distance_to_store_km,
    add_purchase_amount,
    make_groups,
)


def main():
    """Générer les cinq colonnes du dataset dans l'ordre du pipeline."""
    groups = make_groups(2000, weight_a=0.6)
    with_age = add_age(groups)
    with_purchase = add_purchase_amount(with_age)
    with_wait = add_checkout_wait_minutes(with_purchase)
    samples = add_distance_to_store_km(with_wait)

    summary = samples.groupby("group").agg(
        count=("group", "size"),
        age_mean=("age", "mean"),
        purchase_mean=("purchase_amount", "mean"),
        wait_mean=("checkout_wait_minutes", "mean"),
        distance_mean=("distance_to_store_km", "mean"),
    )
    age_gap = summary.loc["B", "age_mean"] - summary.loc["A", "age_mean"]
    purchase_gap = (
        summary.loc["B", "purchase_mean"]
        - summary.loc["A", "purchase_mean"]
    )
    observed_residual_gap = purchase_gap - AGE_EFFECT * age_gap - GROUP_EFFECT

    print(
        "Dataset : group, age, purchase_amount, "
        "checkout_wait_minutes, distance_to_store_km"
    )
    print(summary.round(3))
    print(f"\nÉcart d'âge B-A : {age_gap:.3f} ans")
    print(f"Écart de montant B-A : {purchase_gap:.3f} €")
    print(
        f"Part attendue liée au profil d'âge : "
        f"{AGE_EFFECT * age_gap:.3f} €"
    )
    print(f"Écart programmé restant à âge égal : {GROUP_EFFECT:.3f} €")
    print(
        f"Résidu observé (bruit + arrondi) : "
        f"{observed_residual_gap:.3f} €"
    )
    print("\nCorrélations avec purchase_amount :")
    for predictor in (
        "age",
        "checkout_wait_minutes",
        "distance_to_store_km",
    ):
        correlations = compare_correlations(
            samples,
            predictor=predictor,
        )[["count", "correlation"]]
        print(f"\n{predictor}")
        print(correlations.round(3))


if __name__ == "__main__":
    main()
