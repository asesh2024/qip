#!/usr/bin/env python3
"""
Reproducibility suite: noise resilience and probabilistic calibration of
Basis, Angle and Entangling (CNOT-correlated) encodings under gate-wise
depolarizing noise, with Platt-calibrated Random Forests, ECE, Brier score,
reliability diagrams and confidence-gated LIME-style surrogates.

NOTE: pure NumPy density-matrix simulation (no Qiskit).
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import joblib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # scikit-learn < 1.6
    FrozenEstimator = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import GroupShuffleSplit

# =============================================================================
# CONFIGURATION
# =============================================================================
LABELS = np.array(["00", "01", "10", "11"])
FEATURE_NAMES = ["b0", "b1", "parity", "weight", "correlation"]
I2 = np.eye(2, dtype=complex)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

DEFAULT_CONFIG = {
    "output_dir": "outputs/run3",
    "train_seed": 42,
    "test_seed": 100,
    "input_sampling": "uniform",          # "uniform" (i.i.d.) or "balanced"
    "n_train_base": 2000,                 # base inputs; x shots = instances
    "n_test_base": 2000,
    "shots": 5,
    "calibration_fraction": 0.25,         # grouped by base input
    "confidence_threshold": 0.65,
    "tau_sweep": [0.55, 0.58, 0.60, 0.62, 0.64, 0.65, 0.66, 0.68, 0.70, 0.705, 0.75],
    "local_explanation_samples": 10000,
    "explanations_per_config": 5,         # timed with the naive (uncached) path
    "stability_instances": 20,            # flagged instances per config
    "stability_seeds": 10,
    "decode_repeats": 5,
    "encoders": ["basis", "angle", "entangling"],
    "noise_levels": [0.00, 0.05, 0.10, 0.20],
    "random_forest": {"n_estimators": 100, "max_depth": 5, "n_jobs": -1},
}


def child_rng(*key: int) -> np.random.Generator:
    """Independent, collision-free stream per (seed, stage, encoder, noise)."""
    return np.random.default_rng(np.random.SeedSequence(list(key)))


# =============================================================================
# QUANTUM SIMULATION CORE
# =============================================================================
def ry(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta / 2), -np.sin(theta / 2)],
                     [np.sin(theta / 2),  np.cos(theta / 2)]], dtype=complex)


def expand_1q(gate: np.ndarray, qubit: int) -> np.ndarray:
    return np.kron(gate, I2) if qubit == 0 else np.kron(I2, gate)


def apply_unitary(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    return u @ rho @ u.conj().T


def depolarize(rho: np.ndarray, epsilon: float, qubits: tuple[int, ...]) -> np.ndarray:
    """Eq. (7): (1-e) rho + e (I/2 (x) Tr_q rho); Eq. (8) for two qubits: (1-e) rho + e I/4."""
    if epsilon <= 0.0:
        return rho
    if len(qubits) == 2:
        return (1.0 - epsilon) * rho + epsilon * np.eye(4, dtype=complex) / 4.0
    q = qubits[0]
    tensor = rho.reshape(2, 2, 2, 2)  # q0, q1, q0', q1'
    if q == 0:
        mixed = np.kron(I2 / 2.0, np.trace(tensor, axis1=0, axis2=2))
    else:
        mixed = np.kron(np.trace(tensor, axis1=1, axis2=3), I2 / 2.0)
    return (1.0 - epsilon) * rho + epsilon * mixed


def circuit_density(a: float, b: float, encoder: str, epsilon: float = 0.0) -> np.ndarray:
    """Evolve |00><00| through the encoder circuit with gate-wise noise."""
    psi = np.array([1, 0, 0, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())

    def one(gate, q):
        nonlocal rho
        rho = apply_unitary(rho, expand_1q(gate, q))
        rho = depolarize(rho, epsilon, (q,))

    if encoder == "basis":
        # Noise only follows an executed X gate; idle qubits are noiseless.
        if a >= np.pi / 2.0:
            one(X_GATE, 0)
        if b >= np.pi / 2.0:
            one(X_GATE, 1)
    elif encoder == "angle":
        one(ry(a), 0)
        one(ry(b), 1)
    elif encoder == "entangling":          # Eq. (6): Ry(a), CNOT(0->1), Ry(b)
        one(ry(a), 0)
        rho = apply_unitary(rho, CNOT)
        rho = depolarize(rho, epsilon, (0, 1))
        one(ry(b), 1)
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    rho = (rho + rho.conj().T) / 2.0
    return rho / np.trace(rho)


def extract_features(bits) -> np.ndarray:
    b0, b1 = int(bits[0]), int(bits[1])
    return np.array([b0, b1, b0 ^ b1, b0 + b1, b0 * b1], dtype=float)


# index = 2*b0 + b1  ->  00000, 01110, 10110, 11021
UNIQUE_STATES = np.vstack([extract_features((i // 2, i % 2)) for i in range(4)])


def quadrant_label(a: float, b: float) -> str:
    return f"{int(a >= np.pi / 2.0)}{int(b >= np.pi / 2.0)}"


def uniform_inputs(n: int, rng: np.random.Generator) -> pd.DataFrame:
    a, b = rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n)
    return pd.DataFrame({"a": a, "b": b,
                         "label": [quadrant_label(x, y) for x, y in zip(a, b)],
                         "base_id": np.arange(n)})


def balanced_inputs(per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        lo_a, hi_a = (0.0, np.pi / 2) if label[0] == "0" else (np.pi / 2, np.pi)
        lo_b, hi_b = (0.0, np.pi / 2) if label[1] == "0" else (np.pi / 2, np.pi)
        for _ in range(per_class):
            a, b = rng.uniform(lo_a, hi_a), rng.uniform(lo_b, hi_b)
            rows.append((a, b, quadrant_label(a, b)))
    df = pd.DataFrame(rows, columns=["a", "b", "label"])
    df["base_id"] = np.arange(len(df))
    return df


def make_inputs(config: dict, n_base: int, rng) -> pd.DataFrame:
    if config["input_sampling"] == "balanced":
        return balanced_inputs(n_base // 4, rng)
    return uniform_inputs(n_base, rng)


def synthesize(inputs: pd.DataFrame, encoder: str, epsilon: float, shots: int,
               rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    """Each shot is kept as its own instance (no majority vote)."""
    rows, elapsed = [], 0.0
    for rec in inputs.itertuples(index=False):
        t0 = time.perf_counter()
        rho = circuit_density(rec.a, rec.b, encoder, epsilon)
        elapsed += time.perf_counter() - t0
        ideal = circuit_density(rec.a, rec.b, encoder, 0.0)
        fidelity = float(np.real(np.trace(ideal @ rho)))
        probs = np.clip(np.real(np.diag(rho)), 0.0, None)
        probs /= probs.sum()
        for shot, s in enumerate(rng.choice(4, size=shots, p=probs)):
            f = UNIQUE_STATES[s]
            rows.append([rec.base_id, rec.a, rec.b, rec.label, shot, *f, fidelity])
    cols = ["base_id", "a", "b", "label", "shot", *FEATURE_NAMES, "fidelity"]
    return pd.DataFrame(rows, columns=cols), elapsed / len(inputs)


# =============================================================================
# CALIBRATION & METRICS
# =============================================================================
def compute_ece(probs, y_idx, n_bins: int = 10) -> float:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    acc = pred == y_idx
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            ece += abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return float(ece)


def compute_multiclass_brier(probs, y_idx) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), y_idx] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def fit_calibrated_model(frame: pd.DataFrame, seed: int, rf_cfg: dict, cal_fraction: float):
    """RF on fit split, one-vs-rest sigmoid calibration on a disjoint split
    grouped by base input (no coordinate leakage across shots)."""
    X, y = frame[FEATURE_NAMES].to_numpy(), frame["label"].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=cal_fraction, random_state=seed)
    fit_idx, cal_idx = next(gss.split(X, y, groups=frame["base_id"].to_numpy()))
    assert set(y[fit_idx]) == set(LABELS) == set(y[cal_idx]), "a class is missing in a split"
    rf = RandomForestClassifier(random_state=seed, **rf_cfg).fit(X[fit_idx], y[fit_idx])
    if FrozenEstimator is not None:
        model = CalibratedClassifierCV(FrozenEstimator(rf), method="sigmoid")
    else:
        model = CalibratedClassifierCV(rf, method="sigmoid", cv="prefit")
    model.fit(X[cal_idx], y[cal_idx])
    assert list(model.classes_) == list(LABELS)
    return model, rf


# =============================================================================
# LOCAL SURROGATE (LIME-style)
# =============================================================================
def local_surrogate_explanation(model, instance, predicted, seed, n_samples=10000,
                                cached=True, lam=1e-6) -> pd.DataFrame:
    """Binary perturbation of (b0, b1) with p, w, c recomputed (on-manifold).
    Only 4 distinct neighbourhood points exist, so the 6-column design has rank <= 4:
    the ridge solution is the minimum-norm fit and per-feature weights are not identifiable.
    cached=True evaluates the model once on the 4 unique states (identical result, faster);
    cached=False evaluates all n_samples rows (use for timing the naive cost)."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(n_samples, 2))
    idx = 2 * raw[:, 0] + raw[:, 1]
    idx[0] = int(2 * instance[0] + instance[1])
    nbr = UNIQUE_STATES[idx]
    dist = np.linalg.norm(nbr - instance, axis=1)
    w = np.exp(-(dist ** 2) / (0.75 * np.sqrt(len(instance))) ** 2)
    cls = list(model.classes_).index(predicted)
    target = (model.predict_proba(UNIQUE_STATES)[idx, cls] if cached
              else model.predict_proba(nbr)[:, cls])
    design = np.column_stack([np.ones(n_samples), nbr])
    sw = np.sqrt(w)
    pen = np.sqrt(lam) * np.eye(design.shape[1])
    pen[0, 0] = 0.0                                   # intercept unpenalised
    A = np.vstack([design * sw[:, None], pen])
    y = np.concatenate([target * sw, np.zeros(design.shape[1])])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]       # stable ridge solve
    return pd.DataFrame({"feature": FEATURE_NAMES, "contribution": beta[1:] * instance})


def pairwise_spearman(vectors) -> float:
    vals = []
    for u, v in combinations(vectors, 2):
        if np.ptp(u) == 0 or np.ptp(v) == 0:          # constant vector: undefined
            continue
        vals.append(spearmanr(u, v)[0])
    return float(np.nanmean(vals)) if vals else float("nan")


# =============================================================================
# TRIGGER STATISTICS (Table 8, Table S2, tau sweep) -- computed exactly
# =============================================================================
def export_trigger_tables(store: dict, taus: list[float], tau: float, out: Path):
    t8, s2, sweep = [], [], []
    for (enc, eps), (conf, st) in store.items():
        flagged = conf < tau
        if flagged.all():
            margin = tau - conf.max()
        elif not flagged.any():
            margin = conf.min() - tau
        else:
            margin = np.nan                           # straddles
        t8.append([enc, eps, conf.mean(), conf.std(), np.median(conf), conf.min(), conf.max(),
                   margin, int(flagged.sum()), 100.0 * flagged.mean(),
                   "Straddles" if np.isnan(margin) else ""])
        for s in range(4):
            m = st == s
            if m.any():
                assert np.ptp(conf[m]) < 1e-9, "confidence must be constant per feature state"
            s2.append([enc, eps, "".join(str(int(v)) for v in UNIQUE_STATES[s]),
                       conf[m][0] if m.any() else np.nan, 100.0 * m.mean(),
                       ("Triggered" if conf[m][0] < tau else "Bypassed") if m.any() else ""])
    for t in taus:
        n_flag, full, part = 0, 0, 0
        for conf, _ in store.values():
            k = int((conf < t).sum())
            n_flag += k
            full += k == len(conf)
            part += 0 < k < len(conf)
        total = sum(len(c) for c, _ in store.values())
        sweep.append([t, full, part, n_flag, 100.0 * n_flag / total,
                      total / n_flag if n_flag else np.nan])
    pd.DataFrame(t8, columns=["encoder", "noise", "mean", "std", "median", "min", "max",
                              "margin", "flagged", "rate_pct", "note"]
                 ).to_csv(out / "table8_trigger_stats.csv", index=False)
    pd.DataFrame(s2, columns=["encoder", "noise", "state_features", "peak_posterior",
                              "frequency_pct", "decision"]
                 ).to_csv(out / "tableS2_state_posteriors.csv", index=False)
    pd.DataFrame(sweep, columns=["tau", "full_configs", "partial_configs", "flagged",
                                 "rate_pct", "reduction_x"]
                 ).to_csv(out / "tau_sweep_exact.csv", index=False)
    return pd.DataFrame(t8, columns=["encoder", "noise", "mean", "std", "median", "min",
                                     "max", "margin", "flagged", "rate_pct", "note"])


# =============================================================================
# SELF-CHECKS
# =============================================================================
def bayes_ceiling(encoder: str, per_class: int = 300, seed: int = 0) -> float:
    """Best accuracy any decoder of the measured state can reach at eps=0."""
    inputs = balanced_inputs(per_class, np.random.default_rng(seed))
    P = np.zeros((4, 4))
    for rec in inputs.itertuples(index=False):
        rho = circuit_density(rec.a, rec.b, encoder, 0.0)
        P[list(LABELS).index(rec.label)] += np.real(np.diag(rho))
    return float((P / per_class).max(axis=0).sum() * 0.25)


def self_check():
    print("[*] Self-checks")
    rng = np.random.default_rng(0)
    for enc in ("basis", "angle", "entangling"):
        for _ in range(20):
            a, b = rng.uniform(0, np.pi, 2)
            rho = circuit_density(a, b, enc, 0.2)
            assert abs(np.trace(rho) - 1) < 1e-9 and np.allclose(rho, rho.conj().T)
            assert np.linalg.eigvalsh(rho).min() > -1e-9
    # Basis: |00> untouched; |11> fidelity (1 - e/2)^2
    e = 0.2
    f00 = np.real(np.trace(circuit_density(0.1, 0.1, "basis", 0) @ circuit_density(0.1, 0.1, "basis", e)))
    f11 = np.real(np.trace(circuit_density(2, 2, "basis", 0) @ circuit_density(2, 2, "basis", e)))
    assert abs(f00 - 1) < 1e-12 and abs(f11 - (1 - e / 2) ** 2) < 1e-9
    ceil = {enc: bayes_ceiling(enc) for enc in ("basis", "angle", "entangling")}
    print("    eps=0 decoder ceilings:", {k: round(v, 3) for k, v in ceil.items()},
          "(analytic Angle: (1/2+1/pi)^2 = 0.670)")
    assert abs(ceil["basis"] - 1) < 1e-9 and abs(ceil["angle"] - 0.670) < 0.02
    print("    OK")


# =============================================================================
# FIGURES
# =============================================================================
def style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300,
                         "axes.titlesize": 11, "axes.labelsize": 10})


def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, encoder, epsilon, path):
    fig, ax = plt.subplots(figsize=(4.3, 3.7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set(xlabel="Predicted Label", ylabel="True Label",
           title=rf"{encoder.title()} ($\epsilon$={epsilon:.2f})")
    save_figure(fig, path)


def plot_reliability(proba, y_idx, encoder, epsilon, path):
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    for i, label in enumerate(LABELS):
        yt, yp = calibration_curve((y_idx == i).astype(int), proba[:, i], n_bins=8, strategy="uniform")
        ax.plot(yp, yt, marker="o", label=f"Class {label}")
    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
           title=rf"Reliability: {encoder.title()} ($\epsilon$={epsilon:.2f})")
    ax.legend(loc="upper left", fontsize=8)
    save_figure(fig, path)


def plot_summaries(out: Path, calib_df: pd.DataFrame):
    figs = out / "figures"
    metrics = pd.read_csv(out / "metrics.csv")
    fid = pd.read_csv(out / "fidelity.csv")
    timings = pd.read_csv(out / "timings.csv")
    for df, y, yl, name in [(metrics, "accuracy", "Accuracy", "accuracy_vs_noise"),
                            (fid, "mean_fidelity", "Mean State Fidelity", "fidelity_vs_noise"),
                            (calib_df, "ece", "ECE", "ece_vs_noise"),
                            (calib_df, "brier_score", "Brier Score", "brier_vs_noise")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.lineplot(data=df, x="noise", y=y, hue="encoder", marker="o", ax=ax)
        ax.set(xlabel=r"Depolarizing Noise $\epsilon$", ylabel=yl)
        save_figure(fig, figs / f"{name}.png")
    avg = timings.groupby("encoder", as_index=False)[
        ["encoding_seconds_per_input", "decoding_seconds_per_instance"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    sns.barplot(data=avg, x="encoder", y="encoding_seconds_per_input", hue="encoder",
                ax=axes[0], palette="Blues_d", legend=False)
    sns.barplot(data=avg, x="encoder", y="decoding_seconds_per_instance", hue="encoder",
                ax=axes[1], palette="Oranges_d", legend=False)
    axes[0].set(title="Encoding (NumPy density matrix, per input)", ylabel="Seconds")
    axes[1].set(title="Decoding (per instance)", ylabel="Seconds")
    save_figure(fig, figs / "timings.png")


# =============================================================================
# PIPELINE
# =============================================================================
@dataclass
class RunResult:
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    triggers: pd.DataFrame


def run(config: dict) -> RunResult:
    print("=" * 78 + "\nSTARTING BENCHMARK PIPELINE\n" + "=" * 78)
    t_start = time.perf_counter()
    out = Path(config["output_dir"])
    figs, models = out / "figures", out / "models"
    figs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    style()
    self_check()

    tr_seed, te_seed = int(config["train_seed"]), int(config["test_seed"])
    shots = int(config["shots"])
    tau = float(config["confidence_threshold"])
    train_inputs = make_inputs(config, int(config["n_train_base"]), child_rng(tr_seed, 0))
    test_inputs = make_inputs(config, int(config["n_test_base"]), child_rng(te_seed, 0))

    metrics, calib, cms, fids, timings, imps = [], [], [], [], [], []
    explanations, lime_timing, stability, preds = [], [], [], []
    store = {}

    for eidx, enc in enumerate(config["encoders"]):
        print(f"\n[*] Encoder: {enc.upper()}")
        for nidx, eps in enumerate(config["noise_levels"]):
            eps = float(eps)
            print(f"  -> epsilon = {eps:.2f}")
            train, _ = synthesize(train_inputs, enc, eps, shots, child_rng(tr_seed, 2, eidx, nidx))
            model, rf = fit_calibrated_model(train, tr_seed + 10 * eidx + nidx,
                                             config["random_forest"], config["calibration_fraction"])
            joblib.dump({"model": model, "features": FEATURE_NAMES, "encoder": enc},
                        models / f"{enc}_{eps:.2f}.joblib")
            imps += [[enc, eps, n, v] for n, v in zip(FEATURE_NAMES, rf.feature_importances_)]

            test, enc_time = synthesize(test_inputs, enc, eps, shots, child_rng(te_seed, 1, eidx, nidx))
            Xf = test[FEATURE_NAMES].to_numpy()
            n_test = len(test)

            model.predict_proba(Xf)                                   # warm-up
            reps = []
            for _ in range(int(config["decode_repeats"])):
                t0 = time.perf_counter()
                proba = model.predict_proba(Xf)
                reps.append((time.perf_counter() - t0) / n_test)
            dec_time = float(np.mean(reps))
            pred = model.classes_[proba.argmax(axis=1)]

            conf = proba.max(axis=1)
            state = (2 * test["b0"] + test["b1"]).to_numpy().astype(int)
            store[(enc, eps)] = (conf, state)
            np.save(out / f"probs_{enc}_{eps:.2f}.npy", proba)

            y_idx = np.array([list(LABELS).index(l) for l in test["label"]])
            cm = confusion_matrix(test["label"], pred, labels=LABELS)
            acc = accuracy_score(test["label"], pred)
            ece = compute_ece(proba, y_idx)
            brier = compute_multiclass_brier(proba, y_idx)
            ent = float(-np.sum(proba * np.log(proba + 1e-12), axis=1).mean())
            ll = log_loss(test["label"], proba, labels=LABELS)

            metrics.append([enc, eps, acc, ent, ll, int(n_test - np.trace(cm)), n_test])
            calib.append([enc, eps, ece, brier, ll, acc])
            timings.append([enc, eps, enc_time, dec_time])
            fids.append([enc, eps, test["fidelity"].mean()])
            preds.append(pd.DataFrame({"encoder": enc, "noise": eps, "sample": np.arange(n_test),
                                       "base_id": test["base_id"], "true_label": test["label"],
                                       "predicted_label": pred, "confidence": conf,
                                       **{n: test[n] for n in FEATURE_NAMES}}))
            cms += [[enc, eps, LABELS[i], LABELS[j], int(cm[i, j])] for i in range(4) for j in range(4)]

            plot_reliability(proba, y_idx, enc, eps, figs / f"reliability_{enc}_{eps:.2f}.png")
            plot_confusion(cm, enc, eps, figs / f"cm_{enc}_{eps:.2f}.png")

            flagged = np.flatnonzero(conf < tau)
            print(f"    flagged {len(flagged)}/{n_test} ({100 * len(flagged) / n_test:.1f}%)")
            if len(flagged) == 0:
                continue

            # (a) explanations + naive-cost timing
            k = min(int(config["explanations_per_config"]), len(flagged))
            times = []
            for i in flagged[:k]:
                t0 = time.perf_counter()
                exp = local_surrogate_explanation(model, Xf[i], pred[i], te_seed + int(i),
                                                  int(config["local_explanation_samples"]), cached=False)
                times.append(time.perf_counter() - t0)
                exp.insert(0, "sample", int(i)); exp.insert(0, "noise", eps); exp.insert(0, "encoder", enc)
                explanations.append(exp)
                fig, ax = plt.subplots(figsize=(5.2, 3.2))
                ax.barh(exp["feature"], exp["contribution"],
                        color=["#16A34A" if v >= 0 else "#DC2626" for v in exp["contribution"]])
                ax.axvline(0, color="black", lw=0.8)
                ax.set(xlabel=f"Signed contribution toward class {pred[i]}",
                       title=rf"{enc.title()}, $\epsilon$={eps:.2f}, sample {i}")
                save_figure(fig, figs / f"explanation_{enc}_{eps:.2f}_{i}.png")
            lime_timing.append([enc, eps, len(flagged), k, 1000 * float(np.mean(times)),
                                1000 * float(np.std(times))])

            # (b) seed stability on a random flagged subset
            pick = child_rng(te_seed, 3, eidx, nidx).choice(
                flagged, size=min(int(config["stability_instances"]), len(flagged)), replace=False)
            rhos, undefined = [], 0
            for i in pick:
                vecs = [local_surrogate_explanation(model, Xf[i], pred[i], 10_000 + s,
                                                    int(config["local_explanation_samples"]),
                                                    cached=True)["contribution"].to_numpy()
                        for s in range(int(config["stability_seeds"]))]
                r = pairwise_spearman(vecs)
                undefined += np.isnan(r)
                rhos.append(r)
            stability.append([enc, eps, len(pick), int(undefined),
                              float(np.nanmean(rhos)) if undefined < len(pick) else np.nan])

    print("\n[*] Exporting tables")
    metric_df = pd.DataFrame(metrics, columns=["encoder", "noise", "accuracy", "mean_entropy",
                                               "log_loss", "misclassifications", "n"])
    calib_df = pd.DataFrame(calib, columns=["encoder", "noise", "ece", "brier_score", "log_loss", "accuracy"])
    metric_df.to_csv(out / "metrics.csv", index=False)
    calib_df.to_csv(out / "calibration_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(cms, columns=["encoder", "noise", "true_label", "predicted_label", "count"]
                 ).to_csv(out / "confusion_matrices.csv", index=False)
    pd.DataFrame(fids, columns=["encoder", "noise", "mean_fidelity"]).to_csv(out / "fidelity.csv", index=False)
    pd.DataFrame(timings, columns=["encoder", "noise", "encoding_seconds_per_input",
                                   "decoding_seconds_per_instance"]).to_csv(out / "timings.csv", index=False)
    pd.DataFrame(imps, columns=["encoder", "noise", "feature", "importance"]
                 ).to_csv(out / "feature_importance.csv", index=False)
    pd.DataFrame(lime_timing, columns=["encoder", "noise", "n_flagged", "n_timed",
                                       "naive_ms_mean", "naive_ms_std"]
                 ).to_csv(out / "lime_timing.csv", index=False)
    pd.DataFrame(stability, columns=["encoder", "noise", "n_instances", "n_undefined", "mean_spearman"]
                 ).to_csv(out / "lime_stability.csv", index=False)
    if explanations:
        pd.concat(explanations, ignore_index=True).to_csv(out / "local_explanations.csv", index=False)

    trig = export_trigger_tables(store, list(config["tau_sweep"]), tau, out)
    total_flagged = int(trig["flagged"].sum())
    total = sum(len(c) for c, _ in store.values())
    print(f"    flagged {total_flagged}/{total} ({100 * total_flagged / total:.1f}%)")

    (out / "run_manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    plot_summaries(out, calib_df)
    print(f"\n[done] {time.perf_counter() - t_start:.1f}s; artifacts in {out.resolve()}")
    return RunResult(metric_df, calib_df, trig)


if __name__ == "__main__":
    run(DEFAULT_CONFIG)
