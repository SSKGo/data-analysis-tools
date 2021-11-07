from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

# from sklearn.metrics import mean_squared_log_error

# max_error is not available in V0.23.2 package, even it is in document
# from sklearn.metrics import max_error

# To take the version dependency in consideration
# mean_absolute_percentage_error available after V0.24 is not used
# from sklearn.metrics import mean_absolute_percentage_error


def calculte_score(y_actual: np.ndarray, y_estimated: np.ndarray) -> Dict[str, float]:
    score_funcs = {
        "explained_variance": {"func": explained_variance_score},
        "r2": {"func": r2_score},
        "max_error": {"func": None},
        "mean_absolute_error": {"func": mean_absolute_error},
        "median_absolute_error": {"func": median_absolute_error},
        "mean_squared_error": {"func": mean_squared_error, "squared": True},
        "root_mean_squared_error": {"func": mean_squared_error, "squared": False},
        "mean_absolute_percentage_error": {"func": None},
    }
    scores = {}
    for score_type, score_func in score_funcs.items():
        if "squared" in score_func:
            score_val = score_func["func"](y_actual, y_estimated)
            # TODO squared=False for RMSE instead of np.sqrt
            if not score_func.get("squared"):
                score_val = np.sqrt(score_val)
        elif score_type == "max_error":
            abs_error = np.abs(y_estimated - y_actual)
            score_val = np.max(abs_error, axis=0)[0]
        elif score_type == "mean_absolute_percentage_error":
            epsilon = np.finfo(np.float64).eps
            abs_percent_error = (
                np.abs(y_estimated - y_actual)
                / np.maximum(np.abs(y_actual), epsilon)
                * 100
            )
            score_val = np.average(abs_percent_error, axis=0)[0]
        else:
            score_val = score_func["func"](y_actual, y_estimated)
        scores[score_type] = score_val
    return scores


def define_lim(*y_data) -> Tuple[float, float]:
    y_all = np.vstack(y_data)
    y_all[np.isinf(y_all)] = np.median(y_all)
    lim_min, lim_max = np.min(y_all), np.max(y_all)
    lim_width = abs(lim_max - lim_min)
    lim_min -= lim_width * 0.1
    lim_max += lim_width * 0.1
    lim_min, lim_max
    return lim_min, lim_max


def draw_act_est_plot(
    ax,
    y_actual: np.ndarray,
    y_estimated: np.ndarray,
    axis_limit: Optional[Tuple[float, float]] = None,
    add_title: Optional[str] = None,
):
    title = "Actual-Estimate Plot"
    if add_title:
        title += f" {add_title}"
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Estimated")
    if axis_limit:
        lim_min, lim_max = axis_limit
    else:
        lim_min, lim_max = define_lim(y_actual, y_estimated)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    for axis_type in ["x", "y"]:
        ax.grid(
            which="major",
            axis=axis_type,
            color="k",
            alpha=0.3,
            linestyle="--",
            linewidth=1,
        )
    ax.plot([lim_min, lim_max], [lim_min, lim_max], color="k", linewidth=0.5)
    ax.scatter(y_actual, y_estimated, marker="o")
    return ax


def draw_score_table(ax, scores: Dict[str, float], add_title: Optional[str] = None):
    ax.axis("off")
    title = "Scores"
    if add_title:
        title += f" {add_title}"
    ax.set_title(title)
    row_labels = []
    cell_texts = []
    for score_type, score_val in scores.items():
        row_labels.append(score_type)
        cell_texts.append([f"{score_val:.3g}"])
    table = ax.table(
        cellText=cell_texts,
        rowLabels=row_labels,
        colLabels=["Scores"],
        colWidths=[0.2],
        loc="best",
    )
    table.set_fontsize(12)
    table.scale(1.5, 1.5)
    return ax


def draw_q_q_plot(
    ax, y_actual: np.ndarray, y_estimated: np.ndarray, add_title: Optional[str] = None
):
    error = (y_estimated - y_actual).reshape(-1)
    stats.probplot(error, dist="norm", plot=ax)
    title = "Q-Q Plot"
    if add_title:
        title += f" {add_title}"
    ax.set_title(title)
    return ax


def draw_error_hist(
    ax,
    y_actual: np.ndarray,
    y_estimated: np.ndarray,
    bins: int = 20,
    h_range: Optional[Tuple[float, float]] = None,
    add_title: Optional[str] = None,
):
    title = "Error Histogram"
    if add_title:
        title += f" {add_title}"
    ax.set_title(title)
    ax.set_xlabel("Error")
    ax.set_ylabel("Frequency")
    error = (y_estimated - y_actual).reshape(-1)
    ax.hist(error, bins=bins, range=h_range)
    return ax


def draw_figures(data_list, title: str = None):
    n_data = len(data_list)

    nrows, ncols = 4, n_data
    fig, axes = plt.subplots(
        nrows=4, ncols=n_data, figsize=(ncols * 5, nrows * 5), constrained_layout=True
    )
    fig.suptitle(title, fontsize=16)

    # Create common limit
    y_all = []
    error_all = []
    for data_dict in data_list:
        error_all.append(data_dict.get("estimated") - data_dict.get("actual"))
        for data_type in ["actual", "estimated"]:
            if isinstance(data_dict.get(data_type), np.ndarray):
                y_all.append(data_dict.get(data_type))
    y_limit = define_lim(*y_all)
    error_limit = define_lim(*error_all)

    for i_data, data_dict in enumerate(data_list):
        name = data_dict.get("name")
        y_actual = data_dict.get("actual")
        y_estimated = data_dict.get("estimated")
        if (y_actual is not None) and (y_estimated is not None):
            ax_targets = []
            for i_ax in range(nrows):
                ax_targets.append(axes[i_ax] if n_data == 1 else axes[i_ax, i_data])
            scores = calculte_score(y_actual, y_estimated)
            draw_act_est_plot(
                ax_targets[0],
                y_actual,
                y_estimated,
                axis_limit=y_limit,
                add_title=f"[{name}]",
            )
            draw_q_q_plot(ax_targets[1], y_actual, y_estimated, add_title=f"[{name}]")
            draw_error_hist(
                ax_targets[2],
                y_actual,
                y_estimated,
                bins=40,
                h_range=error_limit,
                add_title=f"[{name}]",
            )
            draw_score_table(ax_targets[3], scores, add_title=f"[{name}]")
    return fig, axes
