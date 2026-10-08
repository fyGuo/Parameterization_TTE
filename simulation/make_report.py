"""Render the simulation results as a PDF report.

Reads the CSVs written by run_simulation.py (both nuisance modes) and writes
results/simulation_report.pdf.

    python make_report.py [--results results]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import dgp
from config import Params

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
COLORS = {"naive": "#eb6834", "corrected": "#2a78d6"}
LABELS = {"naive": "Naive g-estimation", "corrected": "Bias-corrected g-estimation"}
PARAM_TEX = {"psi0": r"$\psi_0$ (intercept)", "psi1": r"$\psi_1$ (slope in $L_0$)"}
PAGE = (8.5, 11)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "mathtext.fontset": "dejavusans",
})


def minus(text):
    """Typographic minus sign."""
    return text.replace("-", "\u2212")


def coef(value, name):
    """Signed term of a linear predictor, e.g. 'L_0' for 1 and '-0.3A_0' for -0.3."""
    return name if value == 1 else f"{value:g}{name}"


def draw_panel(ax, draws, param, truth, naive_limit):
    """Mean and central 95% range of the estimates, by sample size and method."""
    sizes = sorted(draws["n"].unique())
    offsets = {"naive": -0.14, "corrected": 0.14}

    for method, color in COLORS.items():
        for i, n in enumerate(sizes):
            est = draws.query("n == @n and method == @method and param == @param")["est"]
            lo, hi = np.percentile(est, [2.5, 97.5])
            x = i + offsets[method]
            ax.plot([x, x], [lo, hi], color=color, lw=2, solid_capstyle="round", zorder=3)
            ax.plot(x, est.mean(), "o", color=color, ms=7, mec="white", mew=1.2, zorder=4,
                    label=LABELS[method] if i == 0 else None)

    right = len(sizes) + 0.6
    for value, text, style in ((truth, "truth", "-"), (naive_limit, "naive limit", (0, (4, 3)))):
        ax.axhline(value, color=INK_SECONDARY, lw=1, ls=style, zorder=2)
        ax.annotate(text, (right, value), xytext=(-2, 4),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=INK_SECONDARY)

    ax.set_xlim(-0.5, right)
    ax.set_xticks(range(len(sizes)), [f"{n:,}" for n in sizes])
    ax.set_xlabel("Sample size n")
    ax.set_title(PARAM_TEX[param], loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def draw_table(ax, summary, title):
    """One row per (n, method); bias, SD, SE and coverage for each parameter."""
    stats = ["bias", "emp_sd", "mean_se", "coverage"]
    wide = summary.pivot(index=["n", "method"], columns="param", values=stats)
    wide = wide.reindex(
        [(n, m) for n in sorted(summary["n"].unique()) for m in COLORS], axis=0
    )

    header = ["n", "Method"] + [
        f"{p} {s}" for p in (r"$\psi_0$", r"$\psi_1$") for s in ("bias", "SD", "SE", "cover")
    ]
    cells = []
    for (n, method), row in wide.iterrows():
        line = [f"{n:,}", LABELS[method].replace(" g-estimation", "")]
        for param in ("psi0", "psi1"):
            line += [minus(f"{row[(s, param)]:.3f}") for s in stats[:3]]
            line.append(f"{100 * row[('coverage', param)]:.1f}%")
        cells.append(line)

    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
    table = ax.table(cellText=cells, colLabels=header, loc="upper center", cellLoc="right",
                     colWidths=[0.07, 0.17] + [0.095] * 8)
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        cell.visible_edges = "B"
        if c == 1:
            cell.set_text_props(ha="left")
            cell._loc = "left"
        if r == 0:
            cell.set_text_props(color=INK_SECONDARY)
        elif cells[r - 1][1] == "Naive":
            cell.set_facecolor("#faf9f7")


def page_one(pdf, draws, summary, p, reps):
    truth, limit = dgp.true_psi(p), dgp.naive_limit(p)
    fig = plt.figure(figsize=PAGE)

    fig.text(0.08, 0.945, "Naive vs. bias-corrected g-estimation", fontsize=16, weight="bold")
    fig.text(0.08, 0.925, "Three-time simulation, variation-independent contrasts that do not "
             "form an SNMM (Figure 4)", fontsize=10, color=INK_SECONDARY)
    fig.text(
        0.08, 0.905,
        "Target: the Trial 0 blip "
        r"$\gamma_0(L_0)=E(Y^{1,1,1}-Y^{0,1,1}\mid L_0)=\psi_0+\psi_1L_0$"
        f", with truth ({truth[0]:g}, {truth[1]:g}).\n"
        "Naive g-estimation (Equation 4) blips down with the Trial 1 regime, so it is "
        r"consistent for $E(Y^{1,1,0}-Y^{0,1,0}\mid L_0)$" "\n"
        + minus(f"= {limit[0]:g} + {limit[1]:g}") + r"$\,L_0$"
        " instead. The corrected estimator solves the Theorem 4 influence-function equation.\n"
        r"$\gamma_1,\gamma_2$ are treated as known; "
        r"$\pi_0,\pi_1,q,\mu_0$ are fitted with correctly specified parametric models." "\n"
        f"{reps:,} replicates per sample size.",
        fontsize=9, va="top", linespacing=1.6,
    )

    fig.text(0.08, 0.76, "Naive estimates settle on the wrong contrast; "
             "corrected estimates on the truth", fontsize=11, weight="bold")
    axes = fig.subplots(1, 2, gridspec_kw=dict(left=0.08, right=0.94, top=0.72, bottom=0.455,
                                               wspace=0.22))
    for ax, param, t, l in zip(axes, ("psi0", "psi1"), truth, limit):
        draw_panel(ax, draws, param, t, l)
    axes[0].set_ylabel("Estimate")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncols=2, frameon=False,
                   fontsize=9, handletextpad=0.3, columnspacing=1.5)
    fig.text(0.94, 0.398, "Dots: Monte Carlo mean. Bars: central 95% of the estimates.",
             ha="right", fontsize=8, color=INK_SECONDARY)

    draw_table(fig.add_axes([0.08, 0.045, 0.86, 0.31]), summary,
               "Estimated nuisance functions")
    fig.text(0.08, 0.115, "SD: empirical standard deviation. SE: mean sandwich standard error. "
             "Cover: coverage of the truth by 95% Wald intervals.",
             fontsize=8, color=INK_SECONDARY)
    pdf.savefig(fig)
    plt.close(fig)


def page_two(pdf, summary, p, reps):
    fig = plt.figure(figsize=PAGE)
    fig.text(0.08, 0.945, "Oracle nuisance functions and design", fontsize=16, weight="bold")
    fig.text(0.08, 0.925, r"Same replicates, with the true $\pi_0,\pi_1,q,\mu_0$ plugged in",
             fontsize=10, color=INK_SECONDARY)
    draw_table(fig.add_axes([0.08, 0.57, 0.86, 0.31]), summary, "Oracle nuisance functions")

    tau = p.tau
    lines = [
        r"$\bf{Data\ generating\ mechanism}$",
        rf"$L_0\sim N(0,1)$,   $A_0\sim\mathrm{{Bern}}\{{\mathrm{{expit}}({p.a01:g}L_0)\}}$",
        rf"$L_1={p.l11:g}L_0+{p.l12:g}A_0+N(0,1)$,   "
        rf"$A_1\sim\mathrm{{Bern}}\{{\mathrm{{expit}}({p.a11:g}L_1{p.a12:+g}A_0)\}}$",
        rf"$L_2={p.l21:g}L_1+{p.l22:g}A_1+{p.l23:g}A_0+N(0,1)$,   "
        rf"$A_2\sim\mathrm{{Bern}}\{{\mathrm{{expit}}({p.a21:g}L_2{p.a22:+g}A_1)\}}$",
        rf"$Y={coef(p.b1, 'L_0')}+{coef(p.b2, 'L_1')}+{coef(p.b3, 'L_2')}+A_2\gamma_2(\bar S_2)+A_1d_1(\bar S_1)"
        rf"+A_0d_0(S_0)+N(0,{p.sigma_y**2:g})$",
        r"$d_1,d_0$ are back-solved so the Trial 1 and Trial 0 contrasts equal the blips below.",
        "",
        r"$\bf{Causal\ contrasts}$",
        rf"Trial 2:  $E(Y^{{A_0,A_1,1}}-Y^{{A_0,A_1,0}}\mid\bar S_2)=\gamma_2="
        rf"{p.t20:g}+{p.t21:g}L_2+A_0({p.t22:g}+{p.t23:g}L_0)$",
        rf"Trial 1:  $E(Y^{{A_0,1,0}}-Y^{{A_0,0,1}}\mid\bar S_1)=\gamma_1={p.t10:g}+{p.t11:g}L_1$"
        r",   $h_2^1=1-A_1$",
        rf"Trial 0:  $E(Y^{{1,1,1}}-Y^{{0,1,1}}\mid S_0)=\gamma_0={p.psi0:g}+{p.psi1:g}L_0$"
        r",   $h_1^0=h_2^0=1$",
        "",
        r"$\bf{Estimating\ equations}$",
        r"Both solve  $P_n[c(L_0)\{A_0-\pi_0(L_0)\}\{H-A_0\gamma_0(L_0;\psi)-\mu_0(L_0)\}]=0$"
        r"  with $c(L_0)=(1,L_0)^\top$.",
        r"Naive:  $H=Y-(A_2-h_2^1)\gamma_2-(A_1-h_1^0)\gamma_1$",
        r"Corrected:  $H=$ naive $+\,R(\Delta\gamma_2-q)+q$,   $R=A_1/\pi_1(\bar S_1)$,   "
        r"$\Delta=h_2^0-h_2^1$,   $q=E(\Delta\gamma_2\mid\bar S_1,A_1=1)$",
        "",
        r"$\bf{Naive\ limit}$",
        r"$\gamma_0(L_0)-\{\theta_{21}\tau+\theta_{22}+\theta_{23}L_0\}$, where "
        rf"$\tau={tau:g}$ is the effect of $A_0$ on $E(L_2)$ with $A_1$ fixed and "
        r"$\theta_{21},\theta_{22},\theta_{23}$",
        rf"$={p.t21:g},{p.t22:g},{p.t23:g}$ are the $L_2$, $A_0$ and $A_0L_0$ coefficients of "
        r"$\gamma_2$. The bias is how much $A_0$ modifies the effect of $A_2$.",
        "",
        f"{reps:,} replicates per sample size. Standard errors treat the nuisance functions "
        "as fixed.",
    ]
    fig.text(0.08, 0.63, "\n".join(lines), fontsize=9, va="top", linespacing=1.75)
    pdf.savefig(fig)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    p = Params()
    draws = pd.read_csv(args.results / "draws_estimated.csv")
    reps = draws["rep"].nunique()
    out = args.results / "simulation_report.pdf"
    with PdfPages(out) as pdf:
        page_one(pdf, draws, pd.read_csv(args.results / "summary_estimated.csv"), p, reps)
        page_two(pdf, pd.read_csv(args.results / "summary_oracle.csv"), p, reps)
    print(f"wrote {out}")
