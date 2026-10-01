"""Calculer et tracer la convergence de statistiques descriptives."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
POPULATION_MEAN = 100.0
POPULATION_STD = 15.0
DEFAULT_CONVERGENCE_SEED = 406
SECOND_CONVERGENCE_SEED = 407
MULTIPLE_CONVERGENCE_SEEDS = (406, 407, 408, 409, 410)
MEAN_DIFFERENCE_SEEDS = tuple(range(1404, 1412))
REFERENCE_MEAN = 103.0
REFERENCE_STD_A = 18.0
REFERENCE_STD_B = 24.0
REFERENCE_WEIGHT_A = 0.6
SAMPLE_SIZES = np.array(
    [
        10,
        15,
        20,
        30,
        50,
        75,
        100,
        150,
        200,
        300,
        500,
        750,
        1_000,
        1_500,
        2_000,
        2_500,
        3_000,
        3_500,
        4_000,
        5_000,
    ],
    dtype=int,
)
MAJOR_SAMPLE_SIZES = np.array(
    [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000],
    dtype=int,
)
COLORS = ("#002D62", "#D94801", "#007F73", "#7B2CBF", "#8A6500")
DIFFERENCE_COLORS = (
    "#002D62",
    "#D94801",
    "#007F73",
    "#7B2CBF",
    "#8A6500",
    "#B51F4C",
    "#2F6B2F",
    "#4B5563",
)


def convergence_summary(
    seed: int = DEFAULT_CONVERGENCE_SEED,
    sample_sizes: np.ndarray = SAMPLE_SIZES,
) -> pd.DataFrame:
    """Calculer moyenne et écart-type sur les préfixes d'un même tirage."""
    sizes = np.asarray(sample_sizes, dtype=int)
    if sizes.ndim != 1 or len(sizes) < 2:
        raise ValueError("sample_sizes doit contenir au moins deux tailles.")
    if (sizes < 2).any() or (np.diff(sizes) <= 0).any():
        raise ValueError("sample_sizes doit être strictement croissant et supérieur à 1.")

    rng = np.random.default_rng(seed)
    values = rng.normal(POPULATION_MEAN, POPULATION_STD, int(sizes[-1]))
    means = []
    standard_deviations = []
    for sample_size in sizes:
        current_sample = values[:sample_size]
        means.append(current_sample.mean())
        standard_deviations.append(current_sample.std(ddof=1))

    return pd.DataFrame(
        {
            "seed": seed,
            "sample_size": sizes,
            "sample_mean": means,
            "sample_std": standard_deviations,
        }
    )


def mean_difference_convergence_summary(
    seed: int,
    sample_sizes: np.ndarray = SAMPLE_SIZES,
) -> pd.DataFrame:
    """Calculer avg(B) - avg(A) sur des sous-échantillons A/B emboîtés."""
    sizes = np.asarray(sample_sizes, dtype=int)
    if sizes.ndim != 1 or len(sizes) < 2:
        raise ValueError("sample_sizes doit contenir au moins deux tailles.")
    if (sizes < 2).any() or (np.diff(sizes) <= 0).any():
        raise ValueError("sample_sizes doit être strictement croissant et supérieur à 1.")

    maximum_size = int(sizes[-1])
    maximum_size_a = int(maximum_size * REFERENCE_WEIGHT_A)
    maximum_size_b = maximum_size - maximum_size_a
    rng = np.random.default_rng(seed)
    values_a = rng.normal(REFERENCE_MEAN, REFERENCE_STD_A, maximum_size_a)
    values_b = rng.normal(REFERENCE_MEAN, REFERENCE_STD_B, maximum_size_b)

    sizes_a = (sizes * REFERENCE_WEIGHT_A).astype(int)
    sizes_b = sizes - sizes_a
    differences = [
        values_b[:size_b].mean() - values_a[:size_a].mean()
        for size_a, size_b in zip(sizes_a, sizes_b, strict=True)
    ]
    return pd.DataFrame(
        {
            "seed": seed,
            "sample_size": sizes,
            "sample_size_a": sizes_a,
            "sample_size_b": sizes_b,
            "mean_difference": differences,
        }
    )


def _base_figure(title: str, y_title: str) -> go.Figure:
    """Créer le cadre commun des figures de convergence."""
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        width=1600,
        height=720,
        margin={"l": 110, "r": 45, "t": 95, "b": 100},
        plot_bgcolor="#FAFCFD",
        paper_bgcolor="#FFFFFF",
        title={
            "text": title,
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 30, "color": "#102A43"},
        },
        font={"family": "Aptos, Arial, sans-serif", "size": 24, "color": "#102A43"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "right",
            "x": 0.99,
            "font": {"size": 22},
        },
        hoverlabel={"font_size": 18},
    )
    figure.update_xaxes(
        type="log",
        title_text="Taille n de l'échantillon — échelle logarithmique",
        tickvals=MAJOR_SAMPLE_SIZES,
        ticktext=[f"{value:,}".replace(",", " ") for value in MAJOR_SAMPLE_SIZES],
        title_font={"size": 29},
        tickfont={"size": 21},
        showline=True,
        linecolor="#6F8798",
        gridcolor="#DDE7ED",
        zeroline=False,
    )
    figure.update_yaxes(
        title_text=y_title,
        title_font={"size": 29},
        tickfont={"size": 21},
        showline=True,
        linecolor="#6F8798",
        gridcolor="#DDE7ED",
        zeroline=False,
    )
    return figure


def plot_statistic_convergence(
    summary: pd.DataFrame,
    statistic: str,
    theoretical_value: float,
    observed_label: str,
    theoretical_label: str,
    y_title: str,
    color: str = COLORS[0],
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Tracer une statistique observée face à sa valeur théorique."""
    required = {"seed", "sample_size", statistic}
    if not required.issubset(summary.columns):
        raise ValueError(f"Colonnes manquantes : {sorted(required.difference(summary.columns))}")
    seeds = summary["seed"].unique()
    if len(seeds) != 1:
        raise ValueError("Une figure simple doit contenir un seul seed.")
    seed = int(seeds[0])
    figure = _base_figure(f"seed = {seed} · {len(summary)} tailles de n", y_title)
    figure.add_scatter(
        x=summary["sample_size"],
        y=summary[statistic],
        mode="lines+markers",
        name=f"{observed_label} · seed = {seed}",
        line={"color": color, "width": 4},
        marker={"color": color, "size": 10, "symbol": "circle"},
        hovertemplate="n = %{x}<br>Valeur = %{y:.3f}<extra></extra>",
    )
    figure.add_scatter(
        x=[summary["sample_size"].min(), summary["sample_size"].max()],
        y=[theoretical_value, theoretical_value],
        mode="lines",
        name=theoretical_label,
        line={"color": "#202A33", "width": 4, "dash": "dash"},
        hoverinfo="skip",
    )
    if y_range is not None:
        figure.update_yaxes(range=list(y_range))
    return figure


def plot_multiple_mean_convergence(
    summaries: dict[int, pd.DataFrame],
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Superposer plusieurs trajectoires de moyenne obtenues avec différents seeds."""
    figure = _base_figure(
        f"{len(summaries)} seeds · {len(SAMPLE_SIZES)} tailles de n par trajectoire",
        "Moyenne observée",
    )
    for color, (seed, summary) in zip(COLORS, summaries.items(), strict=True):
        figure.add_scatter(
            x=summary["sample_size"],
            y=summary["sample_mean"],
            mode="lines+markers",
            name=f"seed = {seed}",
            line={"color": color, "width": 3},
            marker={"color": color, "size": 7},
            hovertemplate=f"seed = {seed}<br>n = %{{x}}<br>Moyenne = %{{y:.3f}}<extra></extra>",
        )
    figure.add_scatter(
        x=[SAMPLE_SIZES.min(), SAMPLE_SIZES.max()],
        y=[POPULATION_MEAN, POPULATION_MEAN],
        mode="lines",
        name="μ théorique = 100",
        line={"color": "#202A33", "width": 4, "dash": "dash"},
        hoverinfo="skip",
    )
    if y_range is not None:
        figure.update_yaxes(range=list(y_range))
    return figure


def plot_multiple_mean_difference_convergence(
    summaries: dict[int, pd.DataFrame],
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Superposer plusieurs trajectoires de avg(B) - avg(A)."""
    figure = _base_figure(
        "seeds 1404 à 1411 · 20 valeurs de n par trajectoire",
        "avg(B) − avg(A) (€)",
    )
    for color, (seed, summary) in zip(
        DIFFERENCE_COLORS,
        summaries.items(),
        strict=True,
    ):
        figure.add_scatter(
            x=summary["sample_size"],
            y=summary["mean_difference"],
            mode="lines+markers",
            name=f"seed = {seed}",
            showlegend=False,
            line={"color": color, "width": 3},
            marker={"color": color, "size": 7},
            hovertemplate=(
                f"seed = {seed}<br>n = %{{x}}"
                "<br>avg(B) − avg(A) = %{y:.3f} €<extra></extra>"
            ),
        )
    figure.add_scatter(
        x=[SAMPLE_SIZES.min(), SAMPLE_SIZES.max()],
        y=[0, 0],
        mode="lines",
        name="μB − μA = 0",
        line={"color": "#202A33", "width": 4, "dash": "dash"},
        hoverinfo="skip",
    )
    figure.add_vline(
        x=2_000,
        line={"color": "#6F8798", "width": 2, "dash": "dot"},
    )
    if y_range is not None:
        figure.update_yaxes(range=list(y_range))
        figure.add_annotation(
            x=2_000,
            y=y_range[1] * 0.92,
            text="jeu observé<br>n = 2 000",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={"size": 18, "color": "#526777"},
        )
    return figure


def build_convergence_figures() -> dict[str, go.Figure]:
    """Construire les cinq figures avec des limites comparables."""
    summaries = {
        seed: convergence_summary(seed)
        for seed in MULTIPLE_CONVERGENCE_SEEDS
    }
    all_means = np.concatenate(
        [summary["sample_mean"].to_numpy() for summary in summaries.values()]
    )
    mean_padding = max((all_means.max() - all_means.min()) * 0.08, 0.5)
    mean_range = (
        float(min(all_means.min(), POPULATION_MEAN) - mean_padding),
        float(max(all_means.max(), POPULATION_MEAN) + mean_padding),
    )
    first = summaries[DEFAULT_CONVERGENCE_SEED]
    second = summaries[SECOND_CONVERGENCE_SEED]
    std_values = first["sample_std"].to_numpy()
    std_padding = max((std_values.max() - std_values.min()) * 0.10, 0.4)
    std_range = (
        float(min(std_values.min(), POPULATION_STD) - std_padding),
        float(max(std_values.max(), POPULATION_STD) + std_padding),
    )
    difference_summaries = {
        seed: mean_difference_convergence_summary(seed)
        for seed in MEAN_DIFFERENCE_SEEDS
    }
    all_differences = np.concatenate(
        [
            summary["mean_difference"].to_numpy()
            for summary in difference_summaries.values()
        ]
    )
    difference_limit = max(float(np.abs(all_differences).max()) * 1.08, 1.0)
    return {
        "sample_mean_convergence": plot_statistic_convergence(
            first,
            "sample_mean",
            POPULATION_MEAN,
            "Moyenne observée",
            "μ théorique = 100",
            "Moyenne observée",
            color=COLORS[0],
            y_range=mean_range,
        ),
        "sample_std_convergence": plot_statistic_convergence(
            first,
            "sample_std",
            POPULATION_STD,
            "Écart-type observé s",
            "σ théorique = 15",
            "Écart-type observé s",
            color=COLORS[0],
            y_range=std_range,
        ),
        "sample_mean_convergence_seed_407": plot_statistic_convergence(
            second,
            "sample_mean",
            POPULATION_MEAN,
            "Moyenne observée",
            "μ théorique = 100",
            "Moyenne observée",
            color=COLORS[1],
            y_range=mean_range,
        ),
        "sample_mean_convergence_multiple_seeds": plot_multiple_mean_convergence(
            summaries,
            y_range=mean_range,
        ),
        "sample_mean_difference_convergence_multiple_seeds": (
            plot_multiple_mean_difference_convergence(
                difference_summaries,
                y_range=(-difference_limit, difference_limit),
            )
        ),
    }


def save_convergence_figures(
    figures: dict[str, go.Figure],
    output_dir: Path | None = None,
) -> dict[str, dict[str, Path]]:
    """Exporter chaque figure en PNG et en HTML."""
    destination = OUTPUT_DIR if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    exported = {}
    for name, figure in figures.items():
        png_path = destination / f"{name}.png"
        html_path = destination / f"{name}.html"
        figure.write_image(png_path)
        figure.write_html(html_path)
        exported[name] = {"png": png_path, "html": html_path}
    return exported
