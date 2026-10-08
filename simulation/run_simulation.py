"""Monte Carlo comparison of naive and bias-corrected g-estimation.

    python run_simulation.py [--n 500 1000 2000 5000] [--reps 1000] [--seed 2026]
                             [--nuisance estimated|oracle] [--out results]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import dgp
import estimators
from config import Params

PARAM_NAMES = ("psi0", "psi1")


def run(p, n, reps, seed, nuisance_mode):
    """One row per (replicate, method, parameter)."""
    rows = []
    for rep, child in enumerate(np.random.SeedSequence([seed, n]).spawn(reps)):
        data = dgp.simulate(n, np.random.default_rng(child), p)
        for method, (psi, se) in estimators.estimate(data, p, nuisance_mode).items():
            for name, est, s in zip(PARAM_NAMES, psi, se):
                rows.append(dict(n=n, rep=rep, method=method, param=name, est=est, se=s))
    return pd.DataFrame(rows)


def summarize(draws, p):
    truth = dict(zip(PARAM_NAMES, dgp.true_psi(p)))
    draws = draws.assign(truth=draws["param"].map(truth))
    draws["covered"] = (draws["est"] - draws["truth"]).abs() <= 1.96 * draws["se"]
    draws["sq_err"] = (draws["est"] - draws["truth"]) ** 2

    grouped = draws.groupby(["n", "param", "method"], sort=False)
    summary = grouped.agg(
        truth=("truth", "first"),
        mean=("est", "mean"),
        emp_sd=("est", "std"),
        mean_se=("se", "mean"),
        mse=("sq_err", "mean"),
        coverage=("covered", "mean"),
    )
    summary.insert(2, "bias", summary["mean"] - summary["truth"])
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    summary["coverage"] = summary.pop("coverage")
    return summary.reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nuisance", choices=["estimated", "oracle"], default="estimated")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    p = Params()
    draws = pd.concat(
        [run(p, n, args.reps, args.seed, args.nuisance) for n in args.n], ignore_index=True
    )
    summary = summarize(draws, p)

    args.out.mkdir(parents=True, exist_ok=True)
    draws.to_csv(args.out / f"draws_{args.nuisance}.csv", index=False)
    summary.to_csv(args.out / f"summary_{args.nuisance}.csv", index=False)

    print(f"truth (psi0, psi1)       = {dgp.true_psi(p)}")
    print(f"naive limit (psi0, psi1) = {dgp.naive_limit(p)}")
    print(f"nuisance = {args.nuisance}, reps = {args.reps}\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
