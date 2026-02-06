"""Factor visualization functions for MANTRA."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

logger = logging.getLogger(__name__)


def rmse_loss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Root Mean Squared Error.

    Parameters
    ----------
    yhat : torch.Tensor
        Predicted values
    y : torch.Tensor
        True values

    Returns
    -------
    torch.Tensor
        RMSE value
    """
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def distplots(
    data: dict[str, torch.Tensor],
    keyorder: list[str],
    figsize: tuple[int, int] = (15, 12),
    hspace: float = 0.5,
    fontsize: float = 18,
    y: float = 0.95,
    **kwargs: Any,
) -> plt.Figure:
    """Plot distributions of factor matrices.

    Parameters
    ----------
    data : dict
        Dictionary mapping factor names to tensors
    keyorder : list
        Order of factors to plot
    figsize : tuple, optional
        Figure size, by default (15, 12)
    hspace : float, optional
        Vertical spacing, by default 0.5
    fontsize : float, optional
        Font size for titles, by default 18
    y : float, optional
        Y position for title, by default 0.95

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=hspace)
    plt.suptitle("Factor matrices", fontsize=fontsize, y=y)

    for n, key in enumerate(keyorder):
        ax = plt.subplot(4, 2, n + 1)
        sample = data[key]

        # Handle different device types
        if sample.is_cuda:
            sample = sample.cpu()

        sns.histplot(
            data=pd.DataFrame(sample.detach().numpy().ravel()),
            ax=ax,
            stat="density",
            **kwargs,
        )

        ax.set_title(key.upper())
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_xlabel("")

    return fig


def plot_elbo(
    history: list[float],
    figsize: tuple[int, int] = (10, 6),
    title: str = "ELBO Training History",
) -> plt.Figure:
    """Plot ELBO training history.

    Parameters
    ----------
    history : list
        List of ELBO values per epoch
    figsize : tuple, optional
        Figure size, by default (10, 6)
    title : str, optional
        Plot title, by default "ELBO Training History"

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ELBO")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def plot_rmse_comparison(
    data: pd.DataFrame,
    x_lab: str,
    title: str,
    figsize: tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot RMSE comparison between MANTRA and baseline methods.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with columns for x variable, 'Model', and 'Tensorly' RMSE
    x_lab : str
        Column name for x-axis variable
    title : str
        Plot title
    figsize : tuple, optional
        Figure size, by default (12, 8)

    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    logger.debug("Creating RMSE comparison plot")

    # Reshape data to long format
    df_long = pd.melt(
        data,
        id_vars=[x_lab],
        value_vars=["Model", "Tensorly"],
        var_name="Method",
        value_name="RMSE",
    )

    # Create boxplot
    sns.boxplot(x=x_lab, y="RMSE", hue="Method", data=df_long, ax=ax)

    ax.set_xlabel(x_lab)
    ax.set_ylabel("RMSE")
    ax.set_title(title)

    return fig
