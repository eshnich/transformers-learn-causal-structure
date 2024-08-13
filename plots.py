import numpy as np
from jax import numpy as jnp
from jax.tree_util import tree_map
from matplotlib import pyplot as plt


def plot_losses(test_losses, bayes, save_every, fontsize=20):
    fig = plt.figure(figsize=(6, 6))
    plt.plot(jnp.arange(len(test_losses)) * save_every, test_losses)
    plt.axhline(bayes, c="k", ls="--", label="Bayes")
    plt.legend(fontsize=14)
    fig.suptitle(f"Losses", fontsize=fontsize)
    return fig


def plot_A(A, fontsize=24):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    ax.imshow(A)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("Position", fontsize=fontsize)
    ax.set_ylabel("Position", fontsize=fontsize)
    return fig


def plot_A1(A1, vocab_size, seq_len, lower, upper, fontsize=24, patch=False):
    splits = [vocab_size]
    ratios = [vocab_size, seq_len]
    chunks = jnp.split(A1, splits, axis=0)
    chunks = [jnp.split(x, splits, axis=1) for x in chunks]
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=ratios,
        height_ratios=ratios,
        hspace=0.05,
        wspace=0.05,
    )
    kwargs = dict(
        vmin=jnp.percentile(A1, lower),
        vmax=jnp.percentile(A1, upper),
        aspect="equal",
    )
    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(chunks[i][j], **kwargs)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            ax.xaxis.set_label_position("top")
            if i == 0:
                ax.set_xlabel("Token" if j == 0 else "Position", fontsize=fontsize)
            if j == 0:
                ax.set_ylabel("Token" if i == 0 else "Position", fontsize=fontsize)
    if patch:
        ax.add_patch(
            plt.Rectangle(
                [-0.02, -0.02],
                1.04,
                1.04,
                transform=ax.transAxes,
                fill=False,
                linewidth=5,
                edgecolor="red",
                clip_on=False,
            )
        )
    return fig


def plot_A2(
    A2,
    vocab_size,
    seq_len,
    lower,
    upper,
    fontsize=18,
    bigfont=20,
    vpad=0.06,
    hpad=0.06,
    height=0.06,
    patch=False,
):
    fig = plt.figure(figsize=(6, 6))
    kwargs = dict(
        vmin=jnp.percentile(A2, lower),
        vmax=jnp.percentile(A2, upper),
        aspect="equal",
    )
    chunks = A2
    chunks = tree_map(lambda x: jnp.split(x, 2, axis=0), chunks)
    chunks = tree_map(lambda x: jnp.split(x, 2, axis=1), chunks)
    chunks = tree_map(lambda x: jnp.split(x, [vocab_size], axis=0), chunks)
    chunks = tree_map(lambda x: jnp.split(x, [vocab_size], axis=1), chunks)
    gs = fig.add_gridspec(2, 2, wspace=0.05, hspace=0.05)
    bbox = np.zeros((2, 2, 2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            inner = gs[i, j].subgridspec(
                2,
                2,
                wspace=0.05,
                hspace=0.05,
                width_ratios=[vocab_size, seq_len],
                height_ratios=[vocab_size, seq_len],
            )
            for k in range(2):
                for l in range(2):
                    ax = fig.add_subplot(inner[k, l])
                    bbox[i, j, k, l] = ax.get_position()
                    ax.imshow(chunks[i][j][k][l], **kwargs)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_frame_on(False)
                    ax.xaxis.set_label_position("top")
                    if i == 0 and k == 0:
                        ax.set_xlabel(
                            "Token" if l == 0 else "Position", fontsize=fontsize
                        )
                    if j == 0 and l == 0:
                        ax.set_ylabel(
                            "Token" if k == 0 else "Position", fontsize=fontsize
                        )
                    if patch and i == 0 and j == 1 and k == 0 and l == 0:
                        ax.add_patch(
                            plt.Rectangle(
                                [-0.03, -0.03],
                                1.06,
                                1.06,
                                transform=ax.transAxes,
                                fill=False,
                                linewidth=3,
                                edgecolor="red",
                                clip_on=False,
                            )
                        )

    for i in range(2):
        label = (
            r"Input Seq. $\widetilde{X}$"
            if (i == 0)
            else r"$\mathrm{attn}(\widetilde{X};\widetilde{A}^{(1)})$"
        )
        fig.patches.extend(
            [
                plt.Rectangle(
                    [bbox[0, i, 0, 0].x0, bbox[0, i, 0, 0].y1 + vpad],
                    bbox[0, i, 0, 1].x1 - bbox[0, i, 0, 0].x0,
                    height,
                    transform=fig.transFigure,
                    clip_on=False,
                    fill=None,
                    edgecolor="k",
                ),
                plt.text(
                    (bbox[0, i, 0, 0].x0 + bbox[0, i, 0, 1].x1) / 2,
                    bbox[0, i, 0, 0].y1 + vpad + height / 2,
                    label,
                    va="center",
                    ha="center",
                    transform=fig.transFigure,
                    fontsize=bigfont,
                ),
                plt.Rectangle(
                    [bbox[i, 0, 0, 0].x0 - hpad, bbox[i, 0, 0, 0].y1],
                    -height,
                    bbox[i, 0, 1, 0].y0 - bbox[i, 0, 0, 0].y1,
                    transform=fig.transFigure,
                    clip_on=False,
                    fill=None,
                    edgecolor="k",
                ),
                plt.text(
                    bbox[i, 0, 0, 0].x0 - hpad - height / 2,
                    (bbox[i, 0, 1, 0].y0 + bbox[i, 0, 0, 0].y1) / 2,
                    label,
                    va="center",
                    ha="center",
                    transform=fig.transFigure,
                    fontsize=bigfont,
                    rotation="vertical",
                ),
            ]
        )
    return fig


def plot_W(
    W,
    vocab_size,
    seq_len,
    lower,
    upper,
    fontsize=18,
    bigfont=20,
    vpad=0.06,
    hpad=0.06,
    height=0.06,
    patch=False,
):
    vpad /= 0.3
    height /= 0.3
    hpad /= 0.3
    kwargs = dict(
        vmin=jnp.percentile(W, lower),
        vmax=jnp.percentile(W, upper),
        aspect="equal",
    )
    chunks = jnp.split(W, [vocab_size + seq_len, 2 * (vocab_size + seq_len)], axis=1)
    chunks[0] = jnp.split(chunks[0], [vocab_size], axis=1)
    chunks[1] = jnp.split(chunks[1], [vocab_size], axis=1)
    chunks[2] = jnp.split(chunks[2], 2, axis=1)
    chunks[2][0] = jnp.split(chunks[2][0], [vocab_size], axis=1)
    chunks[2][1] = jnp.split(chunks[2][1], [vocab_size], axis=1)
    bbox = tree_map(lambda x: 0, chunks)
    fig = plt.figure(figsize=(18, 18 * W.shape[0] / W.shape[1]))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2], wspace=0.05, hspace=0.05)
    inner = gs[0].subgridspec(
        1, 2, wspace=0.05, hspace=0.05, width_ratios=[vocab_size, seq_len]
    )
    ax = fig.add_subplot(inner[0])
    bbox[0][0] = ax.get_position()
    ax.imshow(chunks[0][0], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Token", fontsize=fontsize)
    ax.set_ylabel("Token", fontsize=fontsize)
    ax.xaxis.set_label_position("top")
    ax = fig.add_subplot(inner[1])
    bbox[0][1] = ax.get_position()
    ax.imshow(chunks[0][1], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Position", fontsize=fontsize)
    ax.xaxis.set_label_position("top")

    inner = gs[1].subgridspec(
        1, 2, wspace=0.05, hspace=0.05, width_ratios=[vocab_size, seq_len]
    )
    ax = fig.add_subplot(inner[0])
    bbox[1][0] = ax.get_position()
    ax.imshow(chunks[1][0], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Token", fontsize=fontsize)
    ax.xaxis.set_label_position("top")
    ax = fig.add_subplot(inner[1])
    bbox[1][1] = ax.get_position()
    ax.imshow(chunks[1][1], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Position", fontsize=fontsize)
    ax.xaxis.set_label_position("top")

    outer = gs[2].subgridspec(1, 2, wspace=0.05, hspace=0.05)
    inner = outer[0].subgridspec(
        1, 2, wspace=0.05, hspace=0.05, width_ratios=[vocab_size, seq_len]
    )
    ax = fig.add_subplot(inner[0])
    bbox[2][0][0] = ax.get_position()
    ax.imshow(chunks[2][0][0], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Token", fontsize=fontsize)
    ax.xaxis.set_label_position("top")
    if patch:
        ax.add_patch(
            plt.Rectangle(
                [-0.04, -0.04],
                1.08,
                1.08,
                transform=ax.transAxes,
                fill=False,
                linewidth=4,
                edgecolor="red",
                clip_on=False,
            )
        )
    ax = fig.add_subplot(inner[1])
    bbox[2][0][1] = ax.get_position()
    ax.imshow(chunks[2][0][1], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Position", fontsize=fontsize)
    ax.xaxis.set_label_position("top")
    inner = outer[1].subgridspec(
        1, 2, wspace=0.05, hspace=0.05, width_ratios=[vocab_size, seq_len]
    )
    ax = fig.add_subplot(inner[0])
    bbox[2][1][0] = ax.get_position()
    ax.imshow(chunks[2][1][0], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Token", fontsize=fontsize)
    ax.xaxis.set_label_position("top")
    ax = fig.add_subplot(inner[1])
    bbox[2][1][1] = ax.get_position()
    ax.imshow(chunks[2][1][1], **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Position", fontsize=fontsize)
    ax.xaxis.set_label_position("top")

    aspect = fig.get_figheight() / fig.get_figwidth()
    fig.patches.extend(
        [
            plt.Rectangle(
                [bbox[0][0].x0, bbox[0][0].y1 + vpad],
                bbox[0][1].x1 - bbox[0][0].x0,
                height,
                transform=fig.transFigure,
                clip_on=False,
                fill=None,
                edgecolor="k",
            ),
            plt.text(
                (bbox[0][1].x1 + bbox[0][0].x0) / 2,
                bbox[0][0].y1 + vpad + height / 2,
                r"Input Seq. $\widetilde{X}$",
                va="center",
                ha="center",
                transform=fig.transFigure,
                fontsize=bigfont,
            ),
            plt.Rectangle(
                [bbox[1][0].x0, bbox[1][0].y1 + vpad],
                bbox[1][1].x1 - bbox[1][0].x0,
                height,
                transform=fig.transFigure,
                clip_on=False,
                fill=None,
                edgecolor="k",
            ),
            plt.text(
                (bbox[1][1].x1 + bbox[1][0].x0) / 2,
                bbox[1][0].y1 + vpad + height / 2,
                r"$\mathrm{attn}(\widetilde{X};\widetilde{A}^{(1)})$",
                va="center",
                ha="center",
                transform=fig.transFigure,
                fontsize=bigfont,
            ),
            plt.Rectangle(
                [bbox[2][0][0].x0, bbox[1][0].y1 + vpad],
                bbox[2][1][1].x1 - bbox[2][0][0].x0,
                height,
                transform=fig.transFigure,
                clip_on=False,
                fill=None,
                edgecolor="k",
            ),
            plt.text(
                (bbox[2][1][1].x1 + bbox[2][0][0].x0) / 2,
                bbox[1][0].y1 + vpad + height / 2,
                r"$\mathrm{attn}(h^{(1)};\widetilde{X}^{(2)})$",
                va="center",
                ha="center",
                transform=fig.transFigure,
                fontsize=bigfont,
            ),
            plt.Rectangle(
                [bbox[0][0].x0 - hpad * aspect, bbox[0][0].y0],
                -height * aspect,
                bbox[0][0].y1 - bbox[0][0].y0,
                transform=fig.transFigure,
                clip_on=False,
                fill=None,
                edgecolor="k",
            ),
            plt.text(
                bbox[0][0].x0 - (hpad + height / 2) * aspect,
                (bbox[0][0].y0 + bbox[0][0].y1) / 2,
                r"Output",
                va="center",
                ha="center",
                transform=fig.transFigure,
                fontsize=bigfont,
                rotation="vertical",
            ),
        ]
    )
    return fig
