"""Build complementary plots from one sampled Gaussian-mixture model."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from . import DEFAULT_SEED
from .gaussian_mixture import GaussianMixture


COLORS = {
    "A": "#0072B2",
    "B": "#D55E00",
    "mixture": "#334155",
    "grid": "#E2E8F0",
    "ink": "#172B4D",
}
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "output"


def _require_samples(model):
    """Return stored samples or explain which method must run first.

    Args:
        model: Gaussian mixture expected to contain a stored sample.
    """
    if model.samples is None:
        raise RuntimeError("Call sample() before requesting a plot.")
    return model.samples


def _plot_range(model):
    """Use one numerical range for every directly comparable view.

    Args:
        model: Sampled Gaussian mixture providing values and parameters.
    """
    samples = _require_samples(model)
    lower = min(
        samples["value"].min(),
        model.mean_a - 4 * model.std_a,
        model.mean_b - 4 * model.std_b,
    )
    upper = max(
        samples["value"].max(),
        model.mean_a + 4 * model.std_a,
        model.mean_b + 4 * model.std_b,
    )
    padding = 0.02 * (upper - lower)
    return lower - padding, upper + padding


def _density_grid(model):
    """Return x values shared by all theoretical density curves.

    Args:
        model: Sampled Gaussian mixture providing values and parameters.
    """
    lower, upper = _plot_range(model)
    return np.linspace(lower, upper, 500)


def _common_bins(model, count=45):
    """Return identical histogram bins for A, B and the pooled sample.

    Args:
        model: Sampled Gaussian mixture providing values and parameters.
        count: Number of equal-width histogram intervals.
    """
    lower, upper = _plot_range(model)
    return {"start": lower, "end": upper, "size": (upper - lower) / count}


def _style_figure(figure, title, x_title, y_title, height=610):
    """Apply the same accessible visual system to every standalone view.

    Args:
        figure: Plotly figure to update.
        title: Text displayed above the figure.
        x_title: Text displayed on the horizontal axis.
        y_title: Text displayed on the vertical axis.
        height: Figure height in pixels.
    """
    figure.update_layout(
        title={
            "text": title,
            "x": 0.04,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
        },
        template="plotly_white",
        height=height,
        margin={"l": 78, "r": 36, "t": 120, "b": 78},
        font={
            "family": "Aptos, Arial, sans-serif",
            "size": 15,
            "color": COLORS["ink"],
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "title_text": "",
        },
        hoverlabel={"font_size": 14},
    )
    figure.update_xaxes(
        title_text=x_title,
        showline=True,
        linecolor="#94A3B8",
        gridcolor=COLORS["grid"],
        zeroline=False,
    )
    figure.update_yaxes(
        title_text=y_title,
        showline=True,
        linecolor="#94A3B8",
        gridcolor=COLORS["grid"],
        zeroline=False,
        rangemode="tozero",
    )
    return figure


def plot_components(model):
    """Compare observed A/B densities with their known normal models.

    Args:
        model: Sampled Gaussian mixture providing values and parameters.
    """
    samples = _require_samples(model)
    grid = _density_grid(model)
    bins = _common_bins(model)
    figure = go.Figure()

    for component in ("A", "B"):
        values = samples.loc[samples.component == component, "value"]
        figure.add_histogram(
            x=values,
            name=f"{component} · histogramme observé",
            histnorm="probability density",
            xbins=bins,
            marker={"color": COLORS[component], "line": {"width": 0}},
            opacity=0.42,
            hovertemplate=(
                f"Groupe {component}<br>intervalle=%{{x}}"
                "<br>densité=%{y:.4f}<extra></extra>"
            ),
        )

    for component, mean, std, dash in (
        ("A", model.mean_a, model.std_a, "solid"),
        ("B", model.mean_b, model.std_b, "dash"),
    ):
        figure.add_scatter(
            x=grid,
            y=norm.pdf(grid, mean, std),
            name=f"{component} · courbe normale théorique",
            mode="lines",
            line={"color": COLORS[component], "width": 3.2, "dash": dash},
            hovertemplate=(
                f"Modèle {component}<br>valeur=%{{x:.2f}}"
                "<br>densité=%{y:.4f}<extra></extra>"
            ),
        )

    figure.update_layout(barmode="overlay", bargap=0.02, hovermode="x unified")
    return _style_figure(
        figure,
        "A et B : observations et modèles connus",
        "Valeur",
        "Densité",
    )


def plot_mixture(model):
    """Show the pooled density and the model's weighted contributions.

    Args:
        model: Sampled Gaussian mixture providing values and parameters.
    """
    samples = _require_samples(model)
    grid = _density_grid(model)
    bins = _common_bins(model)
    weight_b = 1 - model.weight_a
    contribution_a = model.weight_a * norm.pdf(
        grid, model.mean_a, model.std_a
    )
    contribution_b = weight_b * norm.pdf(grid, model.mean_b, model.std_b)

    figure = go.Figure()
    figure.add_histogram(
        x=samples["value"],
        name="Mélange · histogramme observé",
        histnorm="probability density",
        xbins=bins,
        marker={"color": "#CBD5E1", "line": {"color": "#94A3B8", "width": 1}},
        opacity=0.72,
        hovertemplate="intervalle=%{x}<br>densité=%{y:.4f}<extra></extra>",
    )
    for component, values, weight, dash in (
        ("A", contribution_a, model.weight_a, "solid"),
        ("B", contribution_b, weight_b, "dash"),
    ):
        figure.add_scatter(
            x=grid,
            y=values,
            name=f"{component} · contribution pondérée ({weight:.0%})",
            mode="lines",
            line={"color": COLORS[component], "width": 2.5, "dash": dash},
            fill="tozeroy",
            fillcolor=(
                "rgba(0, 114, 178, 0.10)" if component == "A"
                else "rgba(213, 94, 0, 0.08)"
            ),
            hovertemplate=(
                f"Contribution {component}<br>valeur=%{{x:.2f}}"
                "<br>densité pondérée=%{y:.4f}<extra></extra>"
            ),
        )
    figure.add_scatter(
        x=grid,
        y=contribution_a + contribution_b,
        name="Mélange · densité théorique",
        mode="lines",
        line={"color": COLORS["mixture"], "width": 4},
        hovertemplate=(
            "Mélange théorique<br>valeur=%{x:.2f}"
            "<br>densité=%{y:.4f}<extra></extra>"
        ),
    )

    figure.update_layout(barmode="overlay", bargap=0.02, hovermode="x unified")
    return _style_figure(
        figure,
        "Le mélange : les contributions pondérées s'additionnent",
        "Valeur",
        "Densité",
    )


def plot_box(model):
    """Compare location and spread for A, B and their pooled sample.

    Args:
        model: Sampled Gaussian mixture providing values and parameters.
    """
    samples = _require_samples(model)
    plot_data = pd.concat(
        [samples, samples.assign(component="Mélange")],
        ignore_index=True,
    )
    figure = go.Figure()
    for component in ("A", "B", "Mélange"):
        values = plot_data.loc[plot_data.component == component, "value"]
        color = COLORS.get(component, COLORS["mixture"])
        figure.add_box(
            x=np.repeat(component, len(values)),
            y=values,
            name=component,
            boxmean=True,
            boxpoints="outliers",
            fillcolor=color,
            line={"color": color, "width": 2},
            marker={"color": color, "opacity": 0.55},
            opacity=0.72,
            hovertemplate=f"{component}<br>valeur=%{{y:.2f}}<extra></extra>",
        )
    figure.update_layout(showlegend=False)
    return _style_figure(
        figure,
        "Position, dispersion et valeurs atypiques",
        "Groupe",
        "Valeur",
    )


def plot_ecdf(model):
    """Show the cumulative proportion at or below each observed value.

    Args:
        model: Sampled Gaussian mixture providing values and parameters.
    """
    samples = _require_samples(model)
    figure = go.Figure()
    for component, dash in (("A", "solid"), ("B", "dash")):
        values = np.sort(
            samples.loc[samples.component == component, "value"].to_numpy()
        )
        cumulative = np.arange(1, len(values) + 1) / len(values)
        figure.add_scatter(
            x=values,
            y=cumulative,
            name=f"{component} · proportion cumulée",
            mode="lines",
            line={
                "color": COLORS[component],
                "width": 3,
                "dash": dash,
                "shape": "hv",
            },
            hovertemplate=(
                f"Groupe {component}<br>valeur=%{{x:.2f}}"
                "<br>proportion cumulée=%{y:.1%}<extra></extra>"
            ),
        )
    figure.update_yaxes(tickformat=".0%", range=[0, 1.02])
    figure.update_layout(hovermode="x unified")
    return _style_figure(
        figure,
        "Proportion cumulée sous chaque valeur",
        "Valeur",
        "Proportion des observations inférieures ou égales à cette valeur",
    )


def plot_violin(model):
    """Offer a smoothed view of each group's distributional shape.

    Args:
        model: Sampled Gaussian mixture providing values and parameters.
    """
    samples = _require_samples(model)
    figure = go.Figure()
    for component in ("A", "B"):
        values = samples.loc[samples.component == component, "value"]
        figure.add_violin(
            x=np.repeat(component, len(values)),
            y=values,
            name=component,
            box={
                "visible": True,
                "fillcolor": "rgba(255, 255, 255, 0.82)",
                "line": {"color": COLORS["ink"], "width": 1.4},
                "width": 0.18,
            },
            meanline={"visible": True, "color": COLORS["ink"], "width": 2},
            points=False,
            spanmode="hard",
            scalemode="width",
            fillcolor=COLORS[component],
            line={"color": COLORS[component], "width": 2},
            opacity=0.68,
            hovertemplate=f"Groupe {component}<br>valeur=%{{y:.2f}}<extra></extra>",
        )
    figure.update_layout(showlegend=False, violinmode="group")
    return _style_figure(
        figure,
        "Forme lissée, médiane et étendue des groupes",
        "Groupe",
        "Valeur",
    )


def plot_dashboard(model):
    """Combine four complementary readings of the same stored sample.

    Args:
        model: Sampled Gaussian mixture providing values and parameters.
    """
    components = plot_components(model)
    mixture = plot_mixture(model)
    violins = plot_violin(model)
    ecdf = plot_ecdf(model)
    figure = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.16,
        subplot_titles=(
            "A/B séparés : observé + modèle",
            "Mélange : contributions + somme",
            "Forme lissée et repères",
            "Proportions cumulées",
        ),
    )

    for source, row, col, show_legend in (
        (components, 1, 1, True),
        (mixture, 1, 2, True),
        (violins, 2, 1, False),
        (ecdf, 2, 2, False),
    ):
        for trace in source.data:
            trace_copy = deepcopy(trace)
            trace_copy.showlegend = show_legend
            figure.add_trace(trace_copy, row=row, col=col)

    lower, upper = _plot_range(model)
    for row, col in ((1, 1), (1, 2), (2, 2)):
        figure.update_xaxes(range=[lower, upper], row=row, col=col)
    figure.update_xaxes(title_text="Valeur", row=1, col=1)
    figure.update_xaxes(title_text="Valeur", row=1, col=2)
    figure.update_xaxes(title_text="Groupe", row=2, col=1)
    figure.update_xaxes(title_text="Valeur", row=2, col=2)
    figure.update_yaxes(title_text="Densité", row=1, col=1)
    figure.update_yaxes(title_text="Densité", row=1, col=2)
    figure.update_yaxes(title_text="Valeur", row=2, col=1)
    figure.update_yaxes(
        title_text="Proportion cumulée",
        tickformat=".0%",
        range=[0, 1.02],
        row=2,
        col=2,
    )
    figure.update_xaxes(
        showline=True,
        linecolor="#94A3B8",
        gridcolor=COLORS["grid"],
        zeroline=False,
    )
    figure.update_yaxes(
        showline=True,
        linecolor="#94A3B8",
        gridcolor=COLORS["grid"],
        zeroline=False,
    )
    figure.update_annotations(font={"size": 17, "color": COLORS["ink"]})
    figure.update_layout(
        title={
            "text": (
                "Comprendre les groupes et leur mélange"
                "<br><sup>Histogrammes = observations ; courbes = modèles "
                "connus de la simulation.</sup>"
            ),
            "x": 0.04,
            "xanchor": "left",
        },
        template="plotly_white",
        height=900,
        margin={"l": 82, "r": 42, "t": 125, "b": 145},
        font={
            "family": "Aptos, Arial, sans-serif",
            "size": 14,
            "color": COLORS["ink"],
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.13,
            "xanchor": "center",
            "x": 0.5,
            "title_text": "",
            "traceorder": "normal",
        },
        hoverlabel={"font_size": 13},
        barmode="overlay",
        bargap=0.02,
    )
    return figure


def build_all_figures(model):
    """Build every Gaussian Plotly figure from the same stored sample."""
    return {
        "components": plot_components(model),
        "mixture": plot_mixture(model),
        "box": plot_box(model),
        "ecdf": plot_ecdf(model),
        "violin": plot_violin(model),
        "dashboard": plot_dashboard(model),
    }


def save_figures(figures, output_dir=None):
    """Save every named Plotly figure as PNG and HTML in one folder."""
    destination = OUTPUT_DIR if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    exported = {}
    for name, figure in figures.items():
        png_path = destination / f"gaussian_{name}.png"
        html_path = destination / f"gaussian_{name}.html"
        figure.write_image(png_path)
        figure.write_html(html_path)
        exported[name] = {"png": png_path, "html": html_path}
    return exported


def main():
    """Build and export every plot, then display only the dashboard."""
    model = GaussianMixture(100, 18, 106, 24, weight_a=0.6)
    model.sample(2000, seed=DEFAULT_SEED)
    figures = build_all_figures(model)
    trace_counts = {name: len(figure.data) for name, figure in figures.items()}
    expected_traces = {
        "components": 4,
        "mixture": 4,
        "box": 3,
        "ecdf": 2,
        "violin": 2,
        "dashboard": 12,
    }

    if trace_counts != expected_traces:
        raise AssertionError(f"Unexpected figure traces: {trace_counts}")

    save_figures(figures)
    figures["dashboard"].show()
    print(f"Gaussian plots written to: {OUTPUT_DIR}")
    print("Gaussian plots smoke check passed.")


if __name__ == "__main__":
    main()
