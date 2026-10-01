"""Build Plotly figures for inferential ideas used in the course."""

import numpy as np
import plotly.graph_objects as go

from distributions.gaussian_mixture import GaussianMixture


BLUE = "#0072B2"
COPPER = "#D55E00"
NAVY = "#102A43"
INK = "#203645"
GRID = "#D4DEE4"
MUTED = "#526777"


def plot_sampling_variation(
    differences,
    observed_difference,
    null_model: GaussianMixture,
    sample_size,
) -> go.Figure:
    """Plot simulated mean differences and the observed project difference.

    Args:
        differences: Simulated ``avg(B) - avg(A)`` values under H0.
        observed_difference: Difference calculated from the observed sample.
        null_model: Equal-means Gaussian model used for the simulation.
        sample_size: Total number of observations generated per repetition.
    """
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all() or not np.isfinite(observed_difference):
        raise ValueError("Differences and observed_difference must be finite.")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")

    repetitions = len(values)
    count_label = f"{repetitions:,}".replace(",", " ")
    weight_b = 1 - null_model.weight_a
    subtitle = (
        f"Sous H₀ : μ₁ = μ₂ = {null_model.mean_a:g} · "
        f"σ₁ = {null_model.std_a:g} · σ₂ = {null_model.std_b:g} · "
        f"n total = {sample_size:,} · A/B = "
        f"{null_model.weight_a:.0%} / {weight_b:.0%}"
    ).replace(",", " ")

    figure = go.Figure()
    figure.add_histogram(
        x=values,
        nbinsx=55,
        name="Différences simulées",
        marker={"color": BLUE, "line": {"color": "#FFFFFF", "width": 0.4}},
        opacity=0.88,
        hovertemplate=(
            "Intervalle centré sur %{x:.2f}<br>"
            "Nombre de répétitions : %{y}<extra></extra>"
        ),
    )
    figure.add_vline(
        x=0,
        line={"color": NAVY, "width": 3, "dash": "dash"},
        name="H₀ : μ₂ − μ₁ = 0",
        showlegend=True,
    )
    figure.add_vline(
        x=observed_difference,
        line={"color": COPPER, "width": 4},
        name=f"Différence observée : {observed_difference:.2f} unités",
        showlegend=True,
    )

    upper_limit = max(6.7, float(observed_difference) + 0.7)
    figure.update_layout(
        title={
            "text": (
                f"<b>Distribution de {count_label} différences de moyennes "
                f"simulées sous H₀</b><br>"
                f"<span style='font-size:24px;color:{MUTED}'>{subtitle}</span>"
            ),
            "x": 0.035,
            "xanchor": "left",
            "y": 0.94,
            "yanchor": "top",
            "font": {"size": 38, "color": NAVY},
        },
        template="plotly_white",
        width=2100,
        height=800,
        margin={"l": 120, "r": 70, "t": 190, "b": 115},
        font={"family": "Aptos, Arial, sans-serif", "size": 22, "color": INK},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 22},
            "title_text": "",
            "traceorder": "normal",
        },
        bargap=0.02,
        hoverlabel={"font_size": 18},
    )
    figure.update_xaxes(
        title_text="Différence simulée avg(B) − avg(A)",
        range=[-4.5, upper_limit],
        dtick=1,
        title_font={"size": 27},
        tickfont={"size": 21},
        showline=True,
        linecolor="#94A3B8",
        gridcolor=GRID,
        zeroline=False,
    )
    figure.update_yaxes(
        title_text="Nombre de répétitions",
        title_font={"size": 27},
        tickfont={"size": 21},
        showline=True,
        linecolor="#94A3B8",
        gridcolor=GRID,
        zeroline=False,
        rangemode="tozero",
    )
    return figure
