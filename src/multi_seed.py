#!/usr/bin/env python3
"""Multi-seed repetition of the pipeline: same model, different random streams.
Usage:
  python src/multi_seed.py --dataset_path data/quantum_dataset_4000.csv --output_dir outputs/multiseed --n_seeds 10
  python src/multi_seed.py --resample_inputs --output_dir outputs/multiseed_resampled --n_seeds 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P


def ci95(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean() - h, x.mean() + h


def run_seed(k, cfg, train_in, test_in):
    tr_seed, te_seed = 42 + 1000 * k, 100 + 1000 * k
    shots, tau = int(cfg["shots"]), float(cfg["confidence_threshold"])
    rows = []
    for e, enc in enumerate(P.ENCODERS):
        for n, eps in enumerate(cfg["noise_levels"]):
            eps = float(eps)
            P.say(f"seed {k}  {enc}  eps={eps:.2f}", 1)
            train, _ = P.synthesize(train_in, enc, eps, shots, P.child_rng(tr_seed, 2, e, n))
            model, _, _, _ = P.fit_calibrated_model(
                train, tr_seed + 10 * e + n, cfg["random_forest"], cfg["calibration_fraction"])
            test, _ = P.synthesize(test_in, enc, eps, shots, P.child_rng(te_seed, 1, e, n))
            proba = model.predict_proba(test[P.FEATURE_NAMES].to_numpy())
            pred = model.classes_[proba.argmax(1)]
            conf = proba.max(1)
            y_idx = np.array([list(P.LABELS).index(l) for l in test["label"]])
            row = dict(seed_index=k, train_seed=tr_seed, test_seed=te_seed, encoder=enc, noise=eps,
                       accuracy=float((pred == test["label"].to_numpy()).mean()),
                       ece=P.compute_ece(proba, y_idx), brier=P.compute_brier(proba, y_idx),
                       fidelity=float(test["fidelity"].mean()), mean_peak=float(conf.mean()),
                       flagged=int((conf < tau).sum()))
            for t in cfg["tau_sweep"]:
                row[f"flag_{t:g}"] = int((conf < t).sum())
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--output_dir", default="outputs/multiseed")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--resample_inputs", action="store_true",
                    help="draw new i.i.d. (a,b) inputs for every seed")
    a = ap.parse_args()
    cfg = dict(P.DEFAULT_CONFIG)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fixed = None
    if not a.resample_inputs:
        assert a.dataset_path, "give --dataset_path or use --resample_inputs"
        fixed = P.load_dataset(a.dataset_path)

    rows = []
    for k in range(a.n_seeds):
        if a.resample_inputs:
            tr = P.uniform_inputs(int(cfg["n_train_base"]), P.child_rng(42 + 1000 * k, 0))
            te = P.uniform_inputs(int(cfg["n_test_base"]), P.child_rng(100 + 1000 * k, 0))
        else:
            tr, te = fixed
        rows += run_seed(k, cfg, tr, te)
        pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)   # save progress

    df = pd.DataFrame(rows)

    # ---- summary: mean, std (ddof=1), 95% t-interval over seeds
    summ = []
    for (enc, eps), g in df.groupby(["encoder", "noise"], sort=False):
        r = dict(encoder=enc, noise=eps, n_seeds=len(g))
        for m in ["accuracy", "ece", "brier", "fidelity", "mean_peak", "flagged"]:
            lo, hi = ci95(g[m])
            r.update({f"{m}_mean": g[m].mean(), f"{m}_std": g[m].std(ddof=1),
                      f"{m}_lo": lo, f"{m}_hi": hi})
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)

    # ---- paired Entangling - Angle accuracy difference (same seed)
    pv = df.pivot_table(index=["seed_index", "noise"], columns="encoder", values="accuracy").reset_index()
    pv["diff"] = pv["entangling"] - pv["angle"]
    paired = []
    for eps, g in pv.groupby("noise"):
        lo, hi = ci95(g["diff"])
        paired.append(dict(noise=eps, mean_diff=g["diff"].mean(), std=g["diff"].std(ddof=1),
                           ci_lo=lo, ci_hi=hi, seeds_entangling_lower=int((g["diff"] < 0).sum()),
                           n_seeds=len(g)))
    pd.DataFrame(paired).to_csv(out / "paired_entangling_minus_angle.csv", index=False)

    # ---- tau sweep: total flagged over all 12 configs, per seed
    total = 12 * int(cfg["n_test_base"]) * int(cfg["shots"])
    sweep = []
    for t in cfg["tau_sweep"]:
        per_seed = df.groupby("seed_index")[f"flag_{t:g}"].sum()
        lo, hi = ci95(per_seed)
        sweep.append(dict(tau=t, flagged_mean=per_seed.mean(), flagged_std=per_seed.std(ddof=1),
                          flagged_min=per_seed.min(), flagged_max=per_seed.max(),
                          rate_pct_mean=100 * per_seed.mean() / total))
    pd.DataFrame(sweep).to_csv(out / "tau_sweep_over_seeds.csv", index=False)

    # ---- LaTeX table: mean +- std
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\caption{{Mean $\pm$ standard deviation over {a.n_seeds} independent runs "
             rf"({'resampled inputs' if a.resample_inputs else 'fixed dataset; shots, forest and split vary'}).}}",
             r"\label{tab:multiseed}", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Encoder & $\epsilon$ & Accuracy & ECE & Brier & Flagged / 10{,}000\\", r"\midrule"]
    prev = None
    for r in summ.itertuples():
        if prev and r.encoder != prev:
            lines.append(r"\midrule")
        prev = r.encoder
        lines.append(f"{r.encoder.title()} & {r.noise:.2f} & {r.accuracy_mean:.4f}$\\pm${r.accuracy_std:.4f} & "
                     f"{r.ece_mean:.4f}$\\pm${r.ece_std:.4f} & {r.brier_mean:.4f}$\\pm${r.brier_std:.4f} & "
                     f"{r.flagged_mean:,.0f}$\\pm${r.flagged_std:,.0f}\\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "table_multiseed.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    P.say(f"done -> {out.resolve()}")


if __name__ == "__main__":
    main()
