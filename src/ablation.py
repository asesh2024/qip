#!/usr/bin/env python3
"""Ablation separating entanglement from noise-channel strength / gate count.

The Entangling circuit is  Ry(a)@q0 -> [middle gate] -> Ry(b)@q1.  We cross two factors:
  middle gate  : identity (product state)  |  CNOT(0->1) (entangled state)
  noise on it  : none | local (independent 1q depolarizing, strength eps, on BOTH qubits)
                      | joint (2q depolarizing I/4, strength eps)
 identity+none  == original Angle encoder
 CNOT+joint     == original Entangling encoder        (both asserted below)
Every contrast keeps the Ry gates and their 1q noise identical, so a difference between two
variants is attributable to exactly one factor.

Part 1 (exact, no sampling, no classifier): Bayes-optimal accuracy of the bitstring by quadrature.
Part 2 (empirical, multi-seed): the same RF + Platt pipeline, paired by seed.

Usage:
  python src/ablation.py --output_dir outputs/ablation --n_seeds 10            # resampled inputs
  python src/ablation.py --dataset_path data/quantum_dataset_4000.csv --output_dir outputs/ablation_fixed --n_seeds 10
  python src/ablation.py --exact_only --output_dir outputs/ablation             # seconds
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P

VARIANTS = {                        # name: (middle gate, noise on middle gate)
    "angle":        ("identity", "none"),
    "id_local":     ("identity", "local"),
    "id_joint":     ("identity", "joint"),
    "cnot_none":    ("cnot", "none"),
    "cnot_local":   ("cnot", "local"),
    "cnot_joint":   ("cnot", "joint"),
}
# contrasts "A - B": what changes from B to A
CONTRASTS = {
    "entanglement | no mid-noise   (cnot_none - angle)":     ("cnot_none", "angle"),
    "entanglement | local mid-noise (cnot_local - id_local)": ("cnot_local", "id_local"),
    "entanglement | joint mid-noise (cnot_joint - id_joint)": ("cnot_joint", "id_joint"),
    "joint noise on product state   (id_joint - angle)":      ("id_joint", "angle"),
    "local noise on product state   (id_local - angle)":      ("id_local", "angle"),
    "joint vs local noise, entangled (cnot_joint - cnot_local)": ("cnot_joint", "cnot_local"),
    "ORIGINAL GAP                   (cnot_joint - angle)":     ("cnot_joint", "angle"),
}

_orig_circuit = P.circuit_density


def circuit_variant(a, b, gate, mid, eps):
    rho = np.zeros((4, 4), dtype=complex); rho[0, 0] = 1.0
    rho = P.apply_unitary(rho, P.expand_1q(P.ry(a), 0)); rho = P.depolarize(rho, eps, (0,))
    if gate == "cnot":
        rho = P.apply_unitary(rho, P.CNOT)
    if mid == "local":
        rho = P.depolarize(rho, eps, (0,)); rho = P.depolarize(rho, eps, (1,))
    elif mid == "joint":
        rho = P.depolarize(rho, eps, (0, 1))
    rho = P.apply_unitary(rho, P.expand_1q(P.ry(b), 1)); rho = P.depolarize(rho, eps, (1,))
    rho = (rho + rho.conj().T) / 2
    return rho / np.trace(rho)


def patched(a, b, encoder, eps=0.0):
    if encoder in VARIANTS:
        return circuit_variant(a, b, *VARIANTS[encoder], eps)
    return _orig_circuit(a, b, encoder, eps)


P.circuit_density = patched          # P.synthesize looks the function up at call time


def self_check():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a, b = rng.uniform(0, np.pi, 2)
        for eps in (0.0, 0.05, 0.2):
            assert np.allclose(circuit_variant(a, b, *VARIANTS["angle"], eps), _orig_circuit(a, b, "angle", eps))
            assert np.allclose(circuit_variant(a, b, *VARIANTS["cnot_joint"], eps), _orig_circuit(a, b, "entangling", eps))
    print("self-check: angle == identity+none and entangling == CNOT+joint (exact match)")


def bayes_accuracy(name, eps, n=24):
    """Exact Bayes-optimal accuracy of the 2-bit outcome, equal class priors, Gauss-Legendre per quadrant."""
    t, w = np.polynomial.legendre.leggauss(n); w = w / w.sum()
    Pm = np.zeros((4, 4))
    for ci, lab in enumerate(P.LABELS):
        ra = (0, np.pi / 2) if lab[0] == "0" else (np.pi / 2, np.pi)
        rb = (0, np.pi / 2) if lab[1] == "0" else (np.pi / 2, np.pi)
        xa = (ra[1] - ra[0]) / 2 * t + sum(ra) / 2
        xb = (rb[1] - rb[0]) / 2 * t + sum(rb) / 2
        for i, a in enumerate(xa):
            for j, b in enumerate(xb):
                Pm[ci] += w[i] * w[j] * np.real(np.diag(patched(a, b, name, eps)))
    return float(Pm.max(0).sum() / 4)


def exact_part(out, noise):
    rows = [dict(variant=v, noise=e, bayes_accuracy=bayes_accuracy(v, e)) for v in VARIANTS for e in noise]
    df = pd.DataFrame(rows); df.to_csv(out / "exact_bayes_accuracy.csv", index=False)
    wide = df.pivot(index="variant", columns="noise", values="bayes_accuracy").loc[list(VARIANTS)]
    print("\nEXACT Bayes-optimal accuracy (no sampling, no classifier):\n", wide.round(4).to_string())
    c = []
    for name, (A, B) in CONTRASTS.items():
        c.append({"contrast": name, **{f"eps={e:.2f}": wide.loc[A, e] - wide.loc[B, e] for e in noise}})
    c = pd.DataFrame(c); c.to_csv(out / "exact_contrasts.csv", index=False)
    print("\nEXACT contrasts (accuracy points):\n", c.round(4).to_string(index=False))
    return df


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2: return np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def run_seed(k, cfg, tr_in, te_in):
    trs, tes, shots = 42 + 1000 * k, 100 + 1000 * k, int(cfg["shots"])
    rows = []
    for v, name in enumerate(VARIANTS):
        for n, eps in enumerate(cfg["noise_levels"]):
            P.say(f"seed {k} {name} eps={eps:.2f}", 1)
            train, _ = P.synthesize(tr_in, name, eps, shots, P.child_rng(trs, 2, v, n))
            model, *_ = P.fit_calibrated_model(train, trs + 10 * v + n, cfg["random_forest"], cfg["calibration_fraction"])
            test, _ = P.synthesize(te_in, name, eps, shots, P.child_rng(tes, 1, v, n))
            proba = model.predict_proba(test[P.FEATURE_NAMES].to_numpy())
            pred = model.classes_[proba.argmax(1)]
            yi = np.array([list(P.LABELS).index(l) for l in test["label"]])
            rows.append(dict(seed_index=k, variant=name, noise=eps,
                             accuracy=float((pred == test["label"].to_numpy()).mean()),
                             ece=P.compute_ece(proba, yi), brier=P.compute_brier(proba, yi),
                             fidelity=float(test["fidelity"].mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path"); ap.add_argument("--output_dir", default="outputs/ablation")
    ap.add_argument("--n_seeds", type=int, default=10); ap.add_argument("--exact_only", action="store_true")
    ap.add_argument("--n_base", type=int, default=None, help="base inputs per split (quick tests only)")
    a = ap.parse_args()
    cfg = dict(P.DEFAULT_CONFIG)
    if a.n_base: cfg["n_train_base"] = cfg["n_test_base"] = a.n_base
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    self_check(); exact_part(out, cfg["noise_levels"])
    if a.exact_only: return
    fixed = P.load_dataset(a.dataset_path) if a.dataset_path else None
    rows = []
    for k in range(a.n_seeds):
        if fixed: tr, te = fixed
        else:
            tr = P.uniform_inputs(int(cfg["n_train_base"]), P.child_rng(42 + 1000 * k, 0))
            te = P.uniform_inputs(int(cfg["n_test_base"]), P.child_rng(100 + 1000 * k, 0))
        rows += run_seed(k, cfg, tr, te)
        pd.DataFrame(rows).to_csv(out / "per_seed.csv", index=False)
    df = pd.DataFrame(rows)
    summ = []
    for (v, e), g in df.groupby(["variant", "noise"], sort=False):
        r = dict(variant=v, noise=e, n_seeds=len(g))
        for m in ["accuracy", "ece", "brier", "fidelity"]:
            lo, hi = ci95(g[m]); r.update({f"{m}_mean": g[m].mean(), f"{m}_std": g[m].std(ddof=1), f"{m}_lo": lo, f"{m}_hi": hi})
        summ.append(r)
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    piv = {m: df.pivot_table(index=["seed_index", "noise"], columns="variant", values=m) for m in ("accuracy", "fidelity")}
    con = []
    for m, pv in piv.items():
        for name, (A, B) in CONTRASTS.items():
            for e, g in (pv[A] - pv[B]).groupby("noise"):
                lo, hi = ci95(g)
                con.append(dict(metric=m, contrast=name, noise=e, mean_diff=g.mean(), std=g.std(ddof=1), ci_lo=lo, ci_hi=hi,
                                seeds_A_lower=int((g < 0).sum()), n_seeds=len(g)))
    pd.DataFrame(con).to_csv(out / "paired_contrasts.csv", index=False)
    print("done ->", out.resolve())


if __name__ == "__main__":
    main()
