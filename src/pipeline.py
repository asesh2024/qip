#!/usr/bin/env python3
"""
Noise resilience and calibration of Basis, Angle and Entangling (CNOT-correlated)
encodings under gate-wise depolarizing noise.

Pure NumPy density-matrix simulation (no Qiskit).
Every shot is its own instance (no majority vote).
Random Forest + one-vs-rest sigmoid Platt calibrators are fit PER noise level on
a calibration split grouped by base input.

Usage:
  python src/pipeline.py --dataset_path data/quantum_dataset_10000.csv --output_dir outputs/run4
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless: no Tk errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from scipy.stats import spearmanr

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:                        # scikit-learn < 1.6
    FrozenEstimator = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, log_loss,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupShuffleSplit

# =============================================================================
# CONSTANTS / CONFIG
# =============================================================================
LABELS = np.array(["00", "01", "10", "11"])
FEATURE_NAMES = ["b0", "b1", "parity", "weight", "correlation"]
ENCODERS = ["basis", "angle", "entangling"]
I2 = np.eye(2, dtype=complex)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

DEFAULT_CONFIG = {
    "output_dir": "outputs/run4",
    "dataset_path": None,                  # CSV: feature_a, feature_b, label
    "train_seed": 42,
    "test_seed": 100,
    "n_train_base": 2000,                  # used only when no dataset is given
    "n_test_base": 2000,
    "shots": 5,
    "calibration_fraction": 0.25,
    "confidence_threshold": 0.65,
    "tau_sweep": [0.55, 0.58, 0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70, 0.705, 0.75],
    "lime_samples": 10000,
    "lime_examples_per_config": 3,         # explanations that are timed (naive path) and plotted
    "stability_instances": 20,
    "stability_seeds": 10,
    "decode_repeats": 5,
    "bootstrap_reps": 1000,
    "noise_levels": [0.00, 0.05, 0.10, 0.20],
    "random_forest": {"n_estimators": 100, "max_depth": 5, "n_jobs": -1},
}

T0 = time.perf_counter()


def say(msg: str, level: int = 0):
    print(f"[{time.perf_counter() - T0:7.1f}s] {'  ' * level}{msg}", flush=True)


def child_rng(*key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(list(key)))


# =============================================================================
# QUANTUM SIMULATION
# =============================================================================
def ry(theta):
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)


def expand_1q(gate, qubit):
    return np.kron(gate, I2) if qubit == 0 else np.kron(I2, gate)


def apply_unitary(rho, u):
    return u @ rho @ u.conj().T


def depolarize(rho, eps, qubits):
    """Eq.(7) single qubit: (1-e)rho + e (I/2 (x) Tr_q rho). Eq.(8) two qubit: (1-e)rho + e I/4."""
    if eps <= 0.0:
        return rho
    if len(qubits) == 2:
        return (1 - eps) * rho + eps * np.eye(4, dtype=complex) / 4
    t = rho.reshape(2, 2, 2, 2)
    if qubits[0] == 0:
        mixed = np.kron(I2 / 2, np.trace(t, axis1=0, axis2=2))
    else:
        mixed = np.kron(np.trace(t, axis1=1, axis2=3), I2 / 2)
    return (1 - eps) * rho + eps * mixed


def circuit_density(a, b, encoder, eps=0.0):
    rho = np.zeros((4, 4), dtype=complex)
    rho[0, 0] = 1.0

    def one(gate, q):
        nonlocal rho
        rho = apply_unitary(rho, expand_1q(gate, q))
        rho = depolarize(rho, eps, (q,))

    if encoder == "basis":                 # noise only after an executed X gate
        if a >= np.pi / 2:
            one(X_GATE, 0)
        if b >= np.pi / 2:
            one(X_GATE, 1)
    elif encoder == "angle":
        one(ry(a), 0)
        one(ry(b), 1)
    elif encoder == "entangling":          # Eq.(6): Ry(a), CNOT(0->1) [2q noise], Ry(b)
        one(ry(a), 0)
        rho = apply_unitary(rho, CNOT)
        rho = depolarize(rho, eps, (0, 1))
        one(ry(b), 1)
    else:
        raise ValueError(encoder)
    rho = (rho + rho.conj().T) / 2
    return rho / np.trace(rho)


def extract_features(b0, b1):
    b0, b1 = int(b0), int(b1)
    return np.array([b0, b1, b0 ^ b1, b0 + b1, b0 * b1], dtype=float)


UNIQUE_STATES = np.vstack([extract_features(i // 2, i % 2) for i in range(4)])  # 00000,01110,10110,11021


def quadrant_label(a, b):
    return f"{int(a >= np.pi / 2)}{int(b >= np.pi / 2)}"


# =============================================================================
# DATA
# =============================================================================
def load_dataset(path: str):
    """CSV rows = base inputs. First half -> train, second half -> test (matches the paper)."""
    say(f"Loading dataset {path}")
    df = pd.read_csv(path, dtype={"label": str}).rename(columns={"feature_a": "a", "feature_b": "b"})
    assert len(df) % 2 == 0, "dataset must have an even number of rows"
    expected = np.array([quadrant_label(x, y) for x, y in zip(df.a, df.b)])
    assert (df.label.to_numpy() == expected).all(), "labels disagree with the quadrant rule"
    n = len(df) // 2
    tr, te = df.iloc[:n].reset_index(drop=True), df.iloc[n:].reset_index(drop=True)
    for d in (tr, te):
        d["base_id"] = np.arange(len(d))
    say(f"train base inputs: {len(tr)}  test base inputs: {len(te)}", 1)
    say(f"test class counts: {te.label.value_counts().sort_index().to_dict()}", 1)
    return tr[["a", "b", "label", "base_id"]], te[["a", "b", "label", "base_id"]]


def uniform_inputs(n, rng):
    a, b = rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n)
    return pd.DataFrame({"a": a, "b": b, "label": [quadrant_label(x, y) for x, y in zip(a, b)],
                         "base_id": np.arange(n)})


def synthesize(inputs, encoder, eps, shots, rng):
    """Each shot is a separate instance. Returns frame and mean seconds per noisy circuit."""
    rows, elapsed = [], 0.0
    for rec in inputs.itertuples(index=False):
        t0 = time.perf_counter()
        rho = circuit_density(rec.a, rec.b, encoder, eps)
        elapsed += time.perf_counter() - t0
        ideal = circuit_density(rec.a, rec.b, encoder, 0.0)
        fid = float(np.real(np.trace(ideal @ rho)))
        p = np.clip(np.real(np.diag(rho)), 0, None)
        p /= p.sum()
        for shot, s in enumerate(rng.choice(4, size=shots, p=p)):
            rows.append([rec.base_id, rec.a, rec.b, rec.label, shot, *UNIQUE_STATES[s], fid])
    cols = ["base_id", "a", "b", "label", "shot", *FEATURE_NAMES, "fidelity"]
    return pd.DataFrame(rows, columns=cols), elapsed / len(inputs)


# =============================================================================
# MODEL / METRICS
# =============================================================================
def fit_calibrated_model(frame, seed, rf_cfg, cal_fraction):
    X, y = frame[FEATURE_NAMES].to_numpy(), frame["label"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=cal_fraction, random_state=seed)
    fit_idx, cal_idx = next(gss.split(X, y, groups=frame["base_id"].to_numpy()))
    assert set(y[fit_idx]) == set(LABELS) == set(y[cal_idx]), "class missing in a split"
    assert not set(frame["base_id"].iloc[fit_idx]) & set(frame["base_id"].iloc[cal_idx]), "leakage"
    rf = RandomForestClassifier(random_state=seed, **rf_cfg).fit(X[fit_idx], y[fit_idx])
    if FrozenEstimator is not None:
        model = CalibratedClassifierCV(FrozenEstimator(rf), method="sigmoid")
    else:
        model = CalibratedClassifierCV(rf, method="sigmoid", cv="prefit")
    model.fit(X[cal_idx], y[cal_idx])
    assert list(model.classes_) == list(LABELS)
    return model, rf, len(fit_idx), len(cal_idx)


def compute_ece(probs, y_idx, n_bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = pred == y_idx
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def compute_brier(probs, y_idx):
    oh = np.zeros_like(probs)
    oh[np.arange(len(probs)), y_idx] = 1
    return float(np.mean(np.sum((probs - oh) ** 2, 1)))


def bootstrap_acc_ci(correct, base_id, n_base, reps, rng):
    """95% CI resampling BASE INPUTS (shots of one input are correlated)."""
    per_base = np.bincount(base_id, weights=correct.astype(float), minlength=n_base)
    shots = len(correct) / n_base
    idx = rng.integers(0, n_base, size=(reps, n_base))
    accs = per_base[idx].sum(1) / (shots * n_base)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


# =============================================================================
# LIME-style local surrogate
# =============================================================================
def local_surrogate(model, instance, predicted, seed, n_samples, cached=True, lam=1e-6):
    """On-manifold binary perturbation of (b0,b1); p,w,c recomputed. Only 4 distinct
    neighbourhood points exist -> design rank <= 4 -> per-feature weights are NOT identifiable
    (ridge returns the minimum-norm solution). cached=False evaluates the model on all rows
    (naive cost, used for timing)."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(n_samples, 2))
    idx = 2 * raw[:, 0] + raw[:, 1]
    idx[0] = int(2 * instance[0] + instance[1])
    nbr = UNIQUE_STATES[idx]
    dist = np.linalg.norm(nbr - instance, axis=1)
    w = np.exp(-(dist ** 2) / (0.75 * np.sqrt(len(instance))) ** 2)
    cls = list(model.classes_).index(predicted)
    target = model.predict_proba(UNIQUE_STATES)[idx, cls] if cached else model.predict_proba(nbr)[:, cls]
    design = np.column_stack([np.ones(n_samples), nbr])
    sw = np.sqrt(w)
    pen = np.sqrt(lam) * np.eye(design.shape[1])
    pen[0, 0] = 0.0
    A = np.vstack([design * sw[:, None], pen])
    yv = np.concatenate([target * sw, np.zeros(design.shape[1])])
    beta = np.linalg.lstsq(A, yv, rcond=None)[0]
    return pd.DataFrame({"feature": FEATURE_NAMES, "contribution": beta[1:] * instance})


def pairwise_spearman(vectors):
    vals = [spearmanr(u, v)[0] for u, v in combinations(vectors, 2)
            if np.ptp(u) > 0 and np.ptp(v) > 0]
    return float(np.nanmean(vals)) if vals else float("nan")


# =============================================================================
# SELF-CHECKS (exact)
# =============================================================================
def exact_ceiling(encoder, n=24):
    """Bayes-optimal eps=0 accuracy by Gauss-Legendre quadrature over the 4 quadrants."""
    t, w = np.polynomial.legendre.leggauss(n)
    P = np.zeros((4, 4))
    for ci, lab in enumerate(LABELS):
        ra = (0, np.pi / 2) if lab[0] == "0" else (np.pi / 2, np.pi)
        rb = (0, np.pi / 2) if lab[1] == "0" else (np.pi / 2, np.pi)
        xa = (ra[1] - ra[0]) / 2 * t + sum(ra) / 2
        xb = (rb[1] - rb[0]) / 2 * t + sum(rb) / 2
        wn = w / w.sum()
        for i, a in enumerate(xa):
            for j, b in enumerate(xb):
                P[ci] += wn[i] * wn[j] * np.real(np.diag(circuit_density(a, b, encoder, 0.0)))
    return float(P.max(0).sum() * 0.25)


def self_check():
    say("Self-checks (exact quadrature, no sampling error)")
    rng = np.random.default_rng(0)
    for enc in ENCODERS:
        for _ in range(20):
            a, b = rng.uniform(0, np.pi, 2)
            rho = circuit_density(a, b, enc, 0.2)
            assert abs(np.trace(rho) - 1) < 1e-9 and np.allclose(rho, rho.conj().T)
            assert np.linalg.eigvalsh(rho).min() > -1e-9
    e = 0.2
    f00 = np.real(np.trace(circuit_density(.1, .1, "basis", 0) @ circuit_density(.1, .1, "basis", e)))
    f11 = np.real(np.trace(circuit_density(2, 2, "basis", 0) @ circuit_density(2, 2, "basis", e)))
    assert abs(f00 - 1) < 1e-12 and abs(f11 - (1 - e / 2) ** 2) < 1e-9
    analytic = (0.5 + 1 / np.pi) ** 2
    ceil = {enc: exact_ceiling(enc) for enc in ENCODERS}
    for k, v in ceil.items():
        say(f"eps=0 Bayes ceiling {k:<10s} = {v:.4f}", 1)
    say(f"analytic (1/2+1/pi)^2        = {analytic:.4f}", 1)
    assert abs(ceil["basis"] - 1) < 1e-9
    assert abs(ceil["angle"] - analytic) < 1e-3 and abs(ceil["entangling"] - analytic) < 1e-3
    say("self-checks passed", 1)
    return ceil


# =============================================================================
# TRIGGER TABLES (Table 8, S2, tau sweep) -- exact, from saved posteriors
# =============================================================================
def build_trigger_tables(store, taus, tau):
    t8, s2, sweep = [], [], []
    for (enc, eps), (conf, st) in store.items():
        fl = conf < tau
        if fl.all():
            margin, note = tau - conf.max(), ""
        elif not fl.any():
            margin, note = conf.min() - tau, ""
        else:
            margin, note = np.nan, "Straddles"
        t8.append([enc, eps, conf.mean(), conf.std(), np.median(conf), conf.min(), conf.max(),
                   margin, int(fl.sum()), 100 * fl.mean(), note])
        for s in range(4):
            m = st == s
            if m.any():
                assert np.ptp(conf[m]) < 1e-9, "confidence must be constant per feature state"
                s2.append([enc, eps, "".join(str(int(v)) for v in UNIQUE_STATES[s]),
                           conf[m][0], 100 * m.mean(), "Triggered" if conf[m][0] < tau else "Bypassed"])
    total = sum(len(c) for c, _ in store.values())
    for t in taus:
        k_all = [(int((c < t).sum()), len(c)) for c, _ in store.values()]
        n = sum(k for k, _ in k_all)
        sweep.append([t, sum(k == m for k, m in k_all), sum(0 < k < m for k, m in k_all), n,
                      100 * n / total, total / n if n else np.nan])
    c8 = ["encoder", "noise", "mean", "std", "median", "min", "max", "margin", "flagged", "rate_pct", "note"]
    return (pd.DataFrame(t8, columns=c8),
            pd.DataFrame(s2, columns=["encoder", "noise", "state_features", "peak_posterior",
                                      "frequency_pct", "decision"]),
            pd.DataFrame(sweep, columns=["tau", "full_configs", "partial_configs", "flagged",
                                         "rate_pct", "reduction_x"]))


# =============================================================================
# LaTeX WRITER
# =============================================================================
def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_tex(df: pd.DataFrame, path: Path, caption: str, label: str, align: str | None = None):
    align = align or "l" * len(df.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(tex_escape(c) for c in df.columns) + r" \\", r"\midrule"]
    prev = None
    for _, r in df.iterrows():
        if prev is not None and "Encoder" in df.columns and r["Encoder"] != prev:
            lines.append(r"\midrule")
        prev = r.get("Encoder", None)
        lines.append(" & ".join(str(v) for v in r.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def f(x, d=4):
    return "---" if pd.isna(x) else f"{x:.{d}f}"


def thousands(n):
    return f"{int(n):,}".replace(",", "{,}")


# =============================================================================
# FIGURES
# =============================================================================
def style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                         "axes.titlesize": 11, "axes.labelsize": 10})


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    say(f"figure -> {path.name}", 2)


def fig_confusion(cm, enc, eps, path):
    fig, ax = plt.subplots(figsize=(4.3, 3.7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set(xlabel="Predicted label", ylabel="True label", title=rf"{enc.title()} ($\epsilon$={eps:.2f})")
    save_fig(fig, path)


def draw_reliability(ax, proba, y_idx, enc, eps):
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for i, lab in enumerate(LABELS):
        yt, yp = calibration_curve((y_idx == i).astype(int), proba[:, i], n_bins=8, strategy="uniform")
        ax.plot(yp, yt, marker="o", label=f"Class {lab}")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
           title=rf"{enc.title()} ($\epsilon$={eps:.2f})")


def fig_reliability(proba, y_idx, enc, eps, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    draw_reliability(ax, proba, y_idx, enc, eps)
    ax.legend(loc="upper left", fontsize=8)
    save_fig(fig, path)


def fig_reliability_grid(cache, noise_levels, path):
    lo, hi = min(noise_levels), max(noise_levels)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for r, eps in enumerate((lo, hi)):
        for c, enc in enumerate(ENCODERS):
            proba, y_idx = cache[(enc, eps)]
            draw_reliability(axes[r, c], proba, y_idx, enc, eps)
    axes[0, 0].legend(loc="upper left", fontsize=7)
    save_fig(fig, path)


def fig_explanation(exp, enc, eps, i, pred, note, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.barh(exp["feature"], exp["contribution"],
            color=["#16A34A" if v >= 0 else "#DC2626" for v in exp["contribution"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel=f"Signed contribution toward class {pred}",
           title=rf"{enc.title()}, $\epsilon$={eps:.2f}, sample {i}{note}")
    save_fig(fig, path)


def fig_lines(df, y, ylabel, path, ylim=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x="noise", y=y, hue="encoder", marker="o", ax=ax)
    ax.set(xlabel=r"Depolarizing noise $\epsilon$", ylabel=ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    save_fig(fig, path)


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def run(cfg: dict):
    out = Path(cfg["output_dir"])
    figs, tabs, mods, arrs = out / "figures", out / "tables", out / "models", out / "arrays"
    for d in (figs, tabs, mods, arrs):
        d.mkdir(parents=True, exist_ok=True)
    style()
    say("=" * 70)
    say("STARTING BENCHMARK PIPELINE")
    say(f"output directory: {out.resolve()}")
    say("=" * 70)

    ceil = self_check()

    # ---- data
    tr_seed, te_seed, shots = int(cfg["train_seed"]), int(cfg["test_seed"]), int(cfg["shots"])
    tau = float(cfg["confidence_threshold"])
    if cfg.get("dataset_path"):
        train_in, test_in = load_dataset(cfg["dataset_path"])
    else:
        say("No dataset given: sampling i.i.d. uniform inputs")
        train_in = uniform_inputs(int(cfg["n_train_base"]), child_rng(tr_seed, 0))
        test_in = uniform_inputs(int(cfg["n_test_base"]), child_rng(te_seed, 0))
    n_test_base = len(test_in)

    metrics, calib, macro, cms, fids, timings, imps = [], [], [], [], [], [], []
    lime_t, stab, expl, preds = [], [], [], []
    store, cache = {}, {}
    n_cfg = len(ENCODERS) * len(cfg["noise_levels"])
    k_cfg = 0

    for eidx, enc in enumerate(ENCODERS):
        for nidx, eps in enumerate(cfg["noise_levels"]):
            eps = float(eps)
            k_cfg += 1
            say(f"[{k_cfg:02d}/{n_cfg}] encoder={enc} epsilon={eps:.2f}")
            say(f"simulating training circuits ({len(train_in)} inputs x {shots} shots)", 1)
            train, _ = synthesize(train_in, enc, eps, shots, child_rng(tr_seed, 2, eidx, nidx))
            say("fitting Random Forest + grouped Platt calibration", 1)
            model, rf, n_fit, n_cal = fit_calibrated_model(
                train, tr_seed + 10 * eidx + nidx, cfg["random_forest"], cfg["calibration_fraction"])
            say(f"fit instances={n_fit}  calibration instances={n_cal}", 2)
            import joblib
            joblib.dump({"model": model, "features": FEATURE_NAMES, "encoder": enc},
                        mods / f"{enc}_{eps:.2f}.joblib")
            imps += [[enc, eps, n, v] for n, v in zip(FEATURE_NAMES, rf.feature_importances_)]

            say(f"simulating test circuits ({n_test_base} inputs x {shots} shots)", 1)
            test, enc_time = synthesize(test_in, enc, eps, shots, child_rng(te_seed, 1, eidx, nidx))
            Xf, n = test[FEATURE_NAMES].to_numpy(), len(test)

            say("decoding + calibrated posteriors", 1)
            model.predict_proba(Xf)                                   # warm-up
            reps = []
            for _ in range(int(cfg["decode_repeats"])):
                t0 = time.perf_counter()
                proba = model.predict_proba(Xf)
                reps.append((time.perf_counter() - t0) / n)
            dec_mean, dec_std = float(np.mean(reps)), float(np.std(reps))
            pred = model.classes_[proba.argmax(1)]
            conf = proba.max(1)
            state = (2 * test["b0"] + test["b1"]).to_numpy().astype(int)
            store[(enc, eps)] = (conf, state)
            np.save(arrs / f"probs_{enc}_{eps:.2f}.npy", proba)

            y_idx = np.array([list(LABELS).index(l) for l in test["label"]])
            cache[(enc, eps)] = (proba, y_idx)
            cm = confusion_matrix(test["label"], pred, labels=LABELS)
            correct = pred == test["label"].to_numpy()
            acc = float(correct.mean())
            lo, hi = bootstrap_acc_ci(correct, test["base_id"].to_numpy(), n_test_base,
                                      int(cfg["bootstrap_reps"]), child_rng(te_seed, 5, eidx, nidx))
            ece, brier = compute_ece(proba, y_idx), compute_brier(proba, y_idx)
            ent = float(-np.sum(proba * np.log(proba + 1e-12), 1).mean())
            ll = float(log_loss(test["label"], proba, labels=LABELS))
            p, r, f1, _ = precision_recall_fscore_support(test["label"], pred, labels=LABELS,
                                                          average="macro", zero_division=0)
            say(f"accuracy={acc:.4f} [{lo:.4f},{hi:.4f}]  ECE={ece:.4f}  Brier={brier:.4f}  "
                f"meanPeak={conf.mean():.4f}", 1)

            if enc == "basis":                                   # noise-model sanity check
                freq = test.groupby("base_id").label.first().value_counts(normalize=True)
                exp_acc = sum(freq.get(l, 0) * (1 - eps / 2) ** l.count("1") for l in LABELS)
                flag = "" if abs(exp_acc - acc) < 0.02 else "  <-- CHECK noise model"
                say(f"expected Basis accuracy under Eq.(7): {exp_acc:.4f}{flag}", 2)

            metrics.append([enc, eps, acc, lo, hi, ent, ll, int(n - np.trace(cm)), n])
            calib.append([enc, eps, ece, brier, ll, acc])
            macro.append([enc, eps, acc, p, r, f1, test["fidelity"].mean()])
            fids.append([enc, eps, test["fidelity"].mean()])
            timings.append([enc, eps, enc_time, dec_mean, dec_std])
            cms += [[enc, eps, LABELS[i], LABELS[j], int(cm[i, j])] for i in range(4) for j in range(4)]
            preds.append(pd.DataFrame({"encoder": enc, "noise": eps, "sample": np.arange(n),
                                       "base_id": test["base_id"].to_numpy(),
                                       "true_label": test["label"].to_numpy(),
                                       "predicted_label": pred, "confidence": conf,
                                       **{c: test[c].to_numpy() for c in FEATURE_NAMES}}))

            say("figures: confusion matrix + reliability diagram", 1)
            fig_confusion(cm, enc, eps, figs / f"cm_{enc}_{eps:.2f}.png")
            fig_reliability(proba, y_idx, enc, eps, figs / f"reliability_{enc}_{eps:.2f}.png")

            # ---- confidence-gated explanations
            flagged = np.flatnonzero(conf < tau)
            say(f"gate tau={tau}: flagged {len(flagged)}/{n} ({100 * len(flagged) / n:.1f}%)", 1)
            targets, note = list(flagged[:int(cfg["lime_examples_per_config"])]), ""
            if len(flagged) == 0 and enc == "basis" and eps == max(cfg["noise_levels"]):
                targets, note = [int(conf.argmin())], " (not flagged; shown for contrast)"
                say("Basis never flagged: explaining the lowest-confidence instance for contrast", 2)
            if len(targets) == 0:
                continue
            times = []
            say(f"timing {len(targets)} naive explanations ({cfg['lime_samples']} perturbations each)", 1)
            for i in targets:
                t0 = time.perf_counter()
                ex = local_surrogate(model, Xf[i], pred[i], te_seed + int(i),
                                     int(cfg["lime_samples"]), cached=False)
                times.append(time.perf_counter() - t0)
                ex.insert(0, "sample", int(i)); ex.insert(0, "noise", eps); ex.insert(0, "encoder", enc)
                ex["note"] = note.strip()
                expl.append(ex)
                fig_explanation(ex, enc, eps, int(i), pred[i], note,
                                figs / f"explanation_{enc}_{eps:.2f}_{int(i)}.png")
            lime_t.append([enc, eps, len(flagged), len(times),
                           1000 * float(np.mean(times)), 1000 * float(np.std(times))])

            if len(flagged):
                say("surrogate stability across perturbation seeds", 1)
                pick = child_rng(te_seed, 3, eidx, nidx).choice(
                    flagged, size=min(int(cfg["stability_instances"]), len(flagged)), replace=False)
                rhos = []
                for i in pick:
                    vecs = [local_surrogate(model, Xf[i], pred[i], 10_000 + s, int(cfg["lime_samples"]),
                                            cached=True)["contribution"].to_numpy()
                            for s in range(int(cfg["stability_seeds"]))]
                    rhos.append(pairwise_spearman(vecs))
                und = int(np.isnan(rhos).sum())
                stab.append([enc, eps, len(pick), und,
                             float(np.nanmean(rhos)) if und < len(pick) else np.nan])
                say(f"mean Spearman={stab[-1][-1]:.3f}  (undefined for {und}/{len(pick)} instances)", 2)

    # =========================================================================
    say("=" * 70)
    say("EXPORTING TABLES (CSV + LaTeX)")
    md = pd.DataFrame(metrics, columns=["encoder", "noise", "accuracy", "acc_ci_lo", "acc_ci_hi",
                                        "mean_entropy", "log_loss", "errors", "n"])
    cd = pd.DataFrame(calib, columns=["encoder", "noise", "ece", "brier_score", "log_loss", "accuracy"])
    mc = pd.DataFrame(macro, columns=["encoder", "noise", "accuracy", "macro_precision",
                                      "macro_recall", "macro_f1", "mean_fidelity"])
    fd = pd.DataFrame(fids, columns=["encoder", "noise", "mean_fidelity"])
    td = pd.DataFrame(timings, columns=["encoder", "noise", "encoding_s_per_input",
                                        "decoding_s_per_instance", "decoding_std"])
    im = pd.DataFrame(imps, columns=["encoder", "noise", "feature", "importance"])
    cmd = pd.DataFrame(cms, columns=["encoder", "noise", "true_label", "predicted_label", "count"])
    t8, s2, sw = build_trigger_tables(store, list(cfg["tau_sweep"]), tau)
    lt = pd.DataFrame(lime_t, columns=["encoder", "noise", "n_flagged", "n_timed", "naive_ms_mean", "naive_ms_std"])
    st = pd.DataFrame(stab, columns=["encoder", "noise", "n_instances", "n_undefined", "mean_spearman"])

    for name, d in [("metrics", md), ("calibration_metrics", cd), ("macro_metrics", mc),
                    ("fidelity", fd), ("timings", td), ("feature_importance", im),
                    ("confusion_matrices", cmd), ("table8_trigger_stats", t8),
                    ("tableS2_state_posteriors", s2), ("tau_sweep_exact", sw),
                    ("lime_timing", lt), ("lime_stability", st)]:
        d.to_csv(tabs / f"{name}.csv", index=False)
        say(f"csv -> tables/{name}.csv", 1)
    pd.concat(preds, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    if expl:
        pd.concat(expl, ignore_index=True).to_csv(tabs / "local_explanations.csv", index=False)
    say("csv -> predictions.csv, tables/local_explanations.csv", 1)

    # projected (not measured) explanation runtime
    ms = (lt.naive_ms_mean * lt.n_timed).sum() / max(lt.n_timed.sum(), 1)
    flagged_total = int(t8.flagged.sum())
    total = int(t8.shape[0] * n_test_base * shots)
    proj = pd.DataFrame([["mean naive ms per explanation (measured)", ms],
                         ["flagged instances", flagged_total],
                         ["projected hours, flagged only", ms * flagged_total / 3.6e6],
                         ["projected hours, all instances", ms * total / 3.6e6],
                         ["workload reduction (x)", total / max(flagged_total, 1)]],
                        columns=["quantity", "value"])
    proj.to_csv(tabs / "runtime_projection.csv", index=False)
    say(f"projected explanation time: {ms * flagged_total / 3.6e6:.2f} h flagged vs "
        f"{ms * total / 3.6e6:.2f} h all (projections, not measurements)", 1)

    # ---- LaTeX tables
    E = lambda s: s.title()
    d = md.copy()
    fid_map = fd.set_index(["encoder", "noise"]).mean_fidelity
    T3 = pd.DataFrame({
        "Encoder": d.encoder.map(E), "$\\epsilon$": d.noise.map(lambda x: f"{x:.2f}"),
        "Accuracy": d.accuracy.map(f),
        "95\\% CI": [f"[{f(l)}, {f(h)}]" for l, h in zip(d.acc_ci_lo, d.acc_ci_hi)],
        "Fidelity": [f(fid_map[(e, n)]) for e, n in zip(d.encoder, d.noise)],
        "Entropy": d.mean_entropy.map(f), "Log loss": d.log_loss.map(f),
        "Errors / N": [f"{e} / {thousands(n)}" for e, n in zip(d.errors, d.n)]})
    write_tex(T3, tabs / "table3_accuracy.tex",
              "Accuracy (95\\% CI by base-input bootstrap), mean fidelity, entropy and log loss.",
              "tab:accuracy", "llcccccc")

    T4 = []
    for enc in ENCODERS:
        for eps in cfg["noise_levels"]:
            sub = cmd[(cmd.encoder == enc) & (cmd.noise == eps)]
            for lab in LABELS:
                row = sub[sub.true_label == lab].set_index("predicted_label")["count"]
                T4.append([E(enc), f"{eps:.2f}", lab, *[str(int(row[c])) for c in LABELS]])
    write_tex(pd.DataFrame(T4, columns=["Encoder", "$\\epsilon$", "True", "00", "01", "10", "11"]),
              tabs / "table4_confusion.tex", "Confusion-matrix counts (rows: true, columns: predicted).",
              "tab:confusion", "llcrrrr")

    T5 = pd.DataFrame({"Encoder": cd.encoder.map(E), "$\\epsilon$": cd.noise.map(lambda x: f"{x:.2f}"),
                       "ECE": cd.ece.map(f), "Brier": cd.brier_score.map(f),
                       "Log loss": cd.log_loss.map(f), "Accuracy": cd.accuracy.map(f)})
    write_tex(T5, tabs / "table5_calibration.tex", "Top-label ECE, Brier score, log loss and accuracy.",
              "tab:calibration", "llccc c".replace(" ", ""))

    T6 = pd.DataFrame({"Encoder": mc.encoder.map(E), "$\\epsilon$": mc.noise.map(lambda x: f"{x:.2f}"),
                       "Accuracy": mc.accuracy.map(f), "Macro P": mc.macro_precision.map(f),
                       "Macro R": mc.macro_recall.map(f), "Macro F1": mc.macro_f1.map(f),
                       "Fidelity": mc.mean_fidelity.map(f)})
    write_tex(T6, tabs / "table6_macro.tex", "Macro-averaged metrics and mean state fidelity.",
              "tab:macro", "llcccccc"[:7])

    g = im.groupby(["encoder", "feature"]).importance.mean().unstack()[FEATURE_NAMES]
    tt = td.groupby("encoder")[["encoding_s_per_input", "decoding_s_per_instance"]].mean()
    T7 = pd.DataFrame({"Encoder": [E(e) for e in ENCODERS],
                       "Enc. ($\\mu$s/input, NumPy)": [f"{tt.loc[e, 'encoding_s_per_input'] * 1e6:.1f}" for e in ENCODERS],
                       "Dec. ($\\mu$s/inst.)": [f"{tt.loc[e, 'decoding_s_per_instance'] * 1e6:.2f}" for e in ENCODERS],
                       **{n: [f"{g.loc[e, n]:.3f}" for e in ENCODERS] for n in FEATURE_NAMES}})
    write_tex(T7, tabs / "table7_timing_gini.tex",
              "Simulator wall-clock latency and Gini importance (mean over noise levels). "
              "Importances among collinear features are not identifiable.", "tab:timing", "lcccccccc"[:8])

    T8 = pd.DataFrame({
        "Encoder": t8.encoder.map(E), "$\\epsilon$": t8.noise.map(lambda x: f"{x:.2f}"),
        "Mean $\\pm$ Std": [f"{m:.3f}$\\pm${s:.3f}" for m, s in zip(t8["mean"], t8["std"])],
        "Median": t8["median"].map(lambda x: f"{x:.3f}"),
        "[Min, Max]": [f"[{a:.3f}, {b:.3f}]" for a, b in zip(t8["min"], t8["max"])],
        "$\\Delta$": [("Straddles" if pd.isna(m) else f"{m:+.3f}") for m in t8.margin],
        "Triggered": t8.flagged.map(thousands), "Rate (\\%)": t8.rate_pct.map(lambda x: f"{x:.1f}")})
    write_tex(T8, tabs / "table8_trigger.tex",
              f"Peak posteriors and trigger rates at $\\tau_{{\\mathrm{{conf}}}}={tau}$ "
              f"(single train/test seed pair {tr_seed}/{te_seed}).", "tab:lime_trigger_distribution", "llcccccc")

    S2 = []
    for (enc, eps), grp in s2.groupby(["encoder", "noise"], sort=False):
        row = {r.state_features: f"{r.peak_posterior:.3f} ({r.frequency_pct:.1f})" for r in grp.itertuples()}
        S2.append([E(enc), f"{eps:.2f}", *[row.get(k, "---") for k in
                   ("00000", "01110", "10110", "11021")],
                   "/".join(sorted(set(grp.decision)))])
    write_tex(pd.DataFrame(S2, columns=["Encoder", "$\\epsilon$", "00000", "01110", "10110", "11021", "Decision"]),
              tabs / "tableS2_states.tex", "Peak posterior (occurrence \\%) per feature state.", "tab:S2",
              "llccccc")

    SW = pd.DataFrame({"$\\tau$": sw.tau.map(lambda x: f"{x:g}"), "Full configs": sw.full_configs,
                       "Partial": sw.partial_configs, "Flagged": sw.flagged.map(thousands),
                       "Rate (\\%)": sw.rate_pct.map(lambda x: f"{x:.1f}"),
                       "Reduction": sw.reduction_x.map(lambda x: "---" if pd.isna(x) else f"{x:.2f}$\\times$")})
    write_tex(SW, tabs / "table_tau_sweep.tex", "Sensitivity to $\\tau_{\\mathrm{conf}}$ (exact counts).",
              "tab:tau_sweep", "cccccc")
    say("latex -> tables/table3..8, S2, tau_sweep (.tex)", 1)

    # =========================================================================
    say("=" * 70)
    say("GENERATING SUMMARY FIGURES")
    fig_lines(md, "accuracy", "Accuracy", figs / "accuracy_vs_noise.png", (-0.02, 1.03))
    fig_lines(fd, "mean_fidelity", "Mean state fidelity", figs / "fidelity_vs_noise.png", (-0.02, 1.03))
    fig_lines(cd, "ece", "ECE", figs / "ece_vs_noise.png")
    fig_lines(cd, "brier_score", "Brier score", figs / "brier_vs_noise.png")
    fig_reliability_grid(cache, list(cfg["noise_levels"]), figs / "reliability_grid.png")

    avg = td.groupby("encoder", as_index=False)[["encoding_s_per_input", "decoding_s_per_instance"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    sns.barplot(data=avg, x="encoder", y="encoding_s_per_input", hue="encoder", ax=axes[0],
                palette="Blues_d", legend=False)
    sns.barplot(data=avg, x="encoder", y="decoding_s_per_instance", hue="encoder", ax=axes[1],
                palette="Oranges_d", legend=False)
    axes[0].set(title="Encoding (NumPy density matrix)", ylabel="Seconds / input")
    axes[1].set(title="Decoding", ylabel="Seconds / instance")
    save_fig(fig, figs / "timings.png")

    gm = im.groupby(["encoder", "feature"], as_index=False).importance.mean()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    sns.barplot(data=gm, x="feature", y="importance", hue="encoder", ax=ax)
    ax.set(xlabel="Feature", ylabel="Gini importance (mean over noise levels)")
    save_fig(fig, figs / "feature_importance.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for enc in ENCODERS:
        s = t8[t8.encoder == enc]
        ax.plot(s.noise, s["mean"], marker="o", label=enc)
        ax.fill_between(s.noise, s["min"], s["max"], alpha=0.15)
    ax.axhline(tau, color="k", ls="--", lw=1, label=rf"$\tau$={tau}")
    ax.set(xlabel=r"Depolarizing noise $\epsilon$", ylabel="Peak posterior (mean, min-max)")
    ax.legend(fontsize=8)
    save_fig(fig, figs / "peak_posterior_vs_noise.png")

    # ---- manifest
    manifest = {"config": cfg, "exact_eps0_ceilings": ceil,
                "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "pandas": pd.__version__, "sklearn": sklearn.__version__,
                             "platform": platform.platform()},
                "total_flagged": flagged_total, "total_instances": total}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    say("=" * 70)
    say(f"DONE in {time.perf_counter() - T0:.1f}s. Flagged {flagged_total}/{total} "
        f"({100 * flagged_total / total:.1f}%).")
    say(f"Outputs: {out.resolve()}  (figures/, tables/, arrays/, models/, predictions.csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--quick", action="store_true", help="fast smoke test (fewer perturbations/bootstraps)")
    a = ap.parse_args()
    config = dict(DEFAULT_CONFIG)
    if a.dataset_path:
        config["dataset_path"] = a.dataset_path
    if a.output_dir:
        config["output_dir"] = a.output_dir
    if a.quick:
        config.update(lime_samples=1000, bootstrap_reps=100, stability_instances=3,
                      stability_seeds=3, lime_examples_per_config=1)
    run(config)
