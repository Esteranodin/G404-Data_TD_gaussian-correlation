"""Construire les trois nuages XY du dataset d'achat."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"
COLORS = {"A": "#002D62", "B": "#D94801"}
POOLED_LINE_COLOR = "#111827"
TRANSPARENT_CENTER = "rgba(255, 255, 255, 0)"
MARKER_SIZE = 12
MARKER_LINE_WIDTH = 2.5
PLOT_SPECS = {
    "checkout_wait_minutes": "Attente en caisse (minutes)",
    "distance_to_store_km": "Distance domicile–magasin (km)",
    "age": "Âge (ans)",
}


def _validate_plot_data(samples: pd.DataFrame, predictor: str) -> None:
    """Vérifier les colonnes et les deux groupes avant le tracé."""
    if predictor not in PLOT_SPECS:
        raise ValueError(f"Variable X inconnue : {predictor}")
    required = {"group", "purchase_amount", predictor}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if samples[list(required)].isna().any().any():
        raise ValueError("Les colonnes du nuage contiennent des valeurs manquantes.")
    if set(samples["group"].unique()) != {"A", "B"}:
        raise ValueError("La colonne group doit contenir les groupes A et B.")


def _shared_y_range(samples: pd.DataFrame) -> tuple[float, float]:
    """Calculer une plage Y commune qui conserve toutes les observations."""
    lower = float(samples["purchase_amount"].min())
    upper = float(samples["purchase_amount"].max())
    padding = max((upper - lower) * 0.04, 1.0)
    return lower - padding, upper + padding


def plot_purchase_xy(
    samples: pd.DataFrame,
    predictor: str,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Tracer un nuage brut A/B avec de grands cercles à centre transparent."""
    _validate_plot_data(samples, predictor)
    shared_range = _shared_y_range(samples) if y_range is None else y_range
    figure = go.Figure()
    for group in ("A", "B"):
        selected = samples.loc[samples["group"] == group]
        figure.add_scatter(
            x=selected[predictor],
            y=selected["purchase_amount"],
            name=f"Magasin {group}",
            mode="markers",
            marker={
                "color": TRANSPARENT_CENTER,
                "size": MARKER_SIZE,
                "symbol": "circle",
                "opacity": 1.0,
                "line": {
                    "color": COLORS[group],
                    "width": MARKER_LINE_WIDTH,
                },
            },
            hovertemplate=(
                f"Magasin {group}<br>"
                f"{PLOT_SPECS[predictor]} : %{{x}}<br>"
                "Montant d'achat : %{y:.1f} €<extra></extra>"
            ),
        )

    figure.update_layout(
        template="plotly_white",
        width=1600,
        height=600,
        margin={"l": 115, "r": 40, "t": 55, "b": 90},
        font={
            "family": "Aptos, Arial, sans-serif",
            "size": 26,
            "color": "#172B4D",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "center",
            "x": 0.5,
            "title_text": "",
            "font": {"size": 28},
        },
        hoverlabel={"font_size": 22},
    )
    figure.update_xaxes(
        title_text=PLOT_SPECS[predictor],
        title_font={"size": 34},
        tickfont={"size": 26},
        showline=True,
        linecolor="#94A3B8",
        gridcolor="#E2E8F0",
        zeroline=False,
    )
    figure.update_yaxes(
        title_text="Montant d'achat (€)",
        range=list(shared_range),
        title_font={"size": 34},
        tickfont={"size": 26},
        showline=True,
        linecolor="#94A3B8",
        gridcolor="#E2E8F0",
        zeroline=False,
    )
    return figure


def build_purchase_xy_figures(samples: pd.DataFrame) -> dict[str, go.Figure]:
    """Construire les trois figures avec une même plage verticale."""
    y_range = _shared_y_range(samples)
    return {
        predictor: plot_purchase_xy(samples, predictor, y_range)
        for predictor in PLOT_SPECS
    }


def _add_linear_fit(
    figure: go.Figure,
    samples: pd.DataFrame,
    predictor: str,
    name: str,
    color: str,
    dash: str,
    width: float,
) -> None:
    """Ajouter une droite ajustée sur la plage X réellement observée."""
    x_values = samples[predictor].to_numpy(dtype=float)
    y_values = samples["purchase_amount"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    line_x = np.array([x_values.min(), x_values.max()])
    figure.add_scatter(
        x=line_x,
        y=intercept + slope * line_x,
        name=name,
        mode="lines",
        line={"color": color, "dash": dash, "width": width},
        hovertemplate=(
            f"{name}<br>{PLOT_SPECS[predictor]} : %{{x:.2f}}<br>"
            "Montant ajusté : %{y:.2f} €<extra></extra>"
        ),
    )


def plot_purchase_xy_with_linear_fits(
    samples: pd.DataFrame,
    predictor: str,
    y_range: tuple[float, float] | None = None,
) -> go.Figure:
    """Ajouter les droites descriptives A+B, A et B au nuage brut."""
    figure = plot_purchase_xy(samples, predictor, y_range)
    _add_linear_fit(
        figure,
        samples,
        predictor,
        "Droite A+B",
        POOLED_LINE_COLOR,
        "dash",
        7,
    )
    for group in ("A", "B"):
        selected = samples.loc[samples["group"] == group]
        _add_linear_fit(
            figure,
            selected,
            predictor,
            f"Droite {group}",
            COLORS[group],
            "solid",
            5,
        )
    figure.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "xanchor": "center",
            "x": 0.5,
            "title_text": "",
            "font": {"size": 23},
            "entrywidth": 190,
            "entrywidthmode": "pixels",
        }
    )
    return figure


def build_purchase_xy_linear_fit_figures(
    samples: pd.DataFrame,
) -> dict[str, go.Figure]:
    """Construire les trois vues analytiques avec une même plage verticale."""
    y_range = _shared_y_range(samples)
    return {
        predictor: plot_purchase_xy_with_linear_fits(samples, predictor, y_range)
        for predictor in PLOT_SPECS
    }


def save_purchase_xy_figures(
    figures: dict[str, go.Figure],
    output_dir: Path | None = None,
    stem_suffix: str = "",
) -> dict[str, dict[str, Path]]:
    """Exporter chaque nuage en PNG et en HTML."""
    destination = OUTPUT_DIR if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    exported = {}
    for predictor, figure in figures.items():
        stem = f"{predictor}_vs_purchase_amount{stem_suffix}"
        png_path = destination / f"{stem}.png"
        html_path = destination / f"{stem}.html"
        figure.write_image(png_path)
        figure.write_html(html_path)
        exported[predictor] = {"png": png_path, "html": html_path}
    return exported
