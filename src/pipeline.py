#!/usr/bin/env python3
"""
Reproducibility Suite: Noise Resilience and Probabilistic Calibration in
Hybrid Quantum Machine Learning Encodings under Depolarizing Noise.

Evaluates Basis, Angle, and Amplitude-inspired encodings with Platt-calibrated
Random Forest models, Expected Calibration Error (ECE), Multi-class Brier score,
Reliability Diagrams, and LIME local surrogate explanations.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:  # scikit-learn < 1.6
    FrozenEstimator = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
LABELS = np.array(["00", "01", "10", "11"])
FEATURE_NAMES = ["b0", "b1", "parity", "weight", "correlation"]
I2 = np.eye(2, dtype=complex)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)

DEFAULT_CONFIG = {
    "output_dir": "outputs/run2",
    "seed": 42,
    "train_per_class": 500,
    "test_per_class": 500,
    "shots": 5,
    "calibration_fraction": 0.25,
    "confidence_threshold": 0.65,
    "local_explanation_samples": 10000,
    "explanations_per_encoder_noise": 5,
    "encoders": ["basis", "angle", "amplitude"],
    "noise_levels": [0.00, 0.05, 0.10, 0.20],
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 5,
        "n_jobs": -1
    }
}

# ==============================================================================
# QUANTUM SIMULATION CORE
# ==============================================================================
def ry(theta: float) -> np.ndarray:
    """Single-qubit Pauli-Y rotation operator."""
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2),  np.cos(theta / 2)]
    ], dtype=complex)


def expand_1q(gate: np.ndarray, qubit: int) -> np.ndarray:
    """Expands a single-qubit gate into a 2-qubit Hilbert space."""
    return np.kron(gate, I2) if qubit == 0 else np.kron(I2, gate)


def apply_unitary(rho: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    """Applies unitary transformation rho -> U @ rho @ U^dagger."""
    return unitary @ rho @ unitary.conj().T


def depolarize(rho: np.ndarray, epsilon: float, qubits: tuple[int, ...]) -> np.ndarray:
    """Applies local depolarizing channel noise to single or multi-qubit subsystems."""
    if epsilon <= 0.0:
        return rho
    if len(qubits) == 2:
        return (1.0 - epsilon) * rho + epsilon * np.eye(4, dtype=complex) / 4.0
    
    q = qubits[0]
    tensor = rho.reshape(2, 2, 2, 2)  # q0, q1, q0', q1'
    if q == 0:
        reduced = np.trace(tensor, axis1=0, axis2=2)
        mixed = np.kron(I2 / 2.0, reduced)
    else:
        reduced = np.trace(tensor, axis1=1, axis2=3)
        mixed = np.kron(reduced, I2 / 2.0)
    return (1.0 - epsilon) * rho + epsilon * mixed


def circuit_density(a: float, b: float, encoder: str, epsilon: float = 0.0) -> np.ndarray:
    """Evolves pure ground state |00> through the specified parameterized circuit."""
    psi = np.array([1, 0, 0, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())

    def one(gate, q):
        nonlocal rho
        rho = apply_unitary(rho, expand_1q(gate, q))
        rho = depolarize(rho, epsilon, (q,))

    if encoder == "basis":
        if a >= np.pi / 2.0:
            one(X_GATE, 0)
        if b >= np.pi / 2.0:
            one(X_GATE, 1)
    elif encoder == "angle":
        one(ry(a), 0)
        one(ry(b), 1)
    elif encoder == "amplitude":
        one(ry(a), 0)
        rho = apply_unitary(rho, CNOT)
        rho = depolarize(rho, epsilon, (0, 1))
        one(ry(b), 1)
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    rho = (rho + rho.conj().T) / 2.0
    return rho / np.trace(rho)


def sample_bits(rho: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Samples computational-basis measurement bitstrings from the state density matrix."""
    probs = np.clip(np.real(np.diag(rho)), 0.0, None)
    probs /= probs.sum()
    state = int(rng.choice(4, p=probs))
    return state // 2, state % 2


def extract_features(bits: tuple[int, int]) -> np.ndarray:
    """Extracts the structured 5D feature vector: [b0, b1, parity, weight, correlation]."""
    b0, b1 = bits
    return np.array([b0, b1, b0 ^ b1, b0 + b1, b0 * b1], dtype=float)


def quadrant_label(a: float, b: float) -> str:
    """Deterministic ground-truth quadrant assignment."""
    return f"{int(a >= np.pi / 2.0)}{int(b >= np.pi / 2.0)}"


def balanced_inputs(per_class: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generates a balanced dataset of continuous parameters (a, b) across the 4 quadrants."""
    rows = []
    for label in LABELS:
        lo_a, hi_a = (0.0, np.pi / 2.0) if label[0] == "0" else (np.pi / 2.0, np.pi)
        lo_b, hi_b = (0.0, np.pi / 2.0) if label[1] == "0" else (np.pi / 2.0, np.pi)
        for _ in range(per_class):
            a = rng.uniform(lo_a, hi_a)
            b = rng.uniform(lo_b, hi_b)
            rows.append((a, b, quadrant_label(a, b)))
    return pd.DataFrame(rows, columns=["a", "b", "label"])


def synthesize(inputs: pd.DataFrame, encoder: str, epsilon: float, shots: int,
               rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    """Simulates quantum execution, extracts measurement shots, and computes state overlap."""
    rows, elapsed = [], 0.0
    for rec in inputs.itertuples(index=False):
        t0 = time.perf_counter()
        rho = circuit_density(rec.a, rec.b, encoder, epsilon)
        elapsed += time.perf_counter() - t0
        
        ideal = circuit_density(rec.a, rec.b, encoder, 0.0)
        fidelity = float(np.real(np.trace(ideal @ rho)))
        
        for shot in range(shots):
            bits = sample_bits(rho, rng)
            feat = extract_features(bits)
            rows.append([rec.a, rec.b, rec.label, shot, *bits, *feat[2:], fidelity])
            
    columns = ["a", "b", "label", "shot", "b0", "b1", "parity", "weight", "correlation", "fidelity"]
    return pd.DataFrame(rows, columns=columns), elapsed / len(inputs)

# ==============================================================================
# CALIBRATION & EVALUATION METRICS
# ==============================================================================
def compute_ece(probs: np.ndarray, y_true_idx: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across M equal-width confidence bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == y_true_idx
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return float(ece)


def compute_multiclass_brier(probs: np.ndarray, y_true_idx: np.ndarray) -> float:
    """Computes Multi-class Brier Score (Mean Squared Error on Probability Vectors)."""
    n_samples, n_classes = probs.shape
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), y_true_idx] = 1.0
    return float(np.mean(np.sum((probs - y_one_hot) ** 2, axis=1)))


def fit_calibrated_model(frame: pd.DataFrame, seed: int, rf_cfg: dict, cal_fraction: float):
    """Trains a Random Forest and fits Platt Scaling (Sigmoid) on a disjoint calibration set."""
    Xf, y = frame[FEATURE_NAMES].to_numpy(), frame["label"].to_numpy()
    x_fit, x_cal, y_fit, y_cal = train_test_split(
        Xf, y, test_size=cal_fraction, random_state=seed, stratify=y
    )
    rf = RandomForestClassifier(random_state=seed, **rf_cfg)
    rf.fit(x_fit, y_fit)
    
    if FrozenEstimator is not None:
        model = CalibratedClassifierCV(FrozenEstimator(rf), method="sigmoid")
    else:
        model = CalibratedClassifierCV(rf, method="sigmoid", cv="prefit")
    model.fit(x_cal, y_cal)
    return model, rf


def local_surrogate_explanation(model, instance: np.ndarray, predicted: str, seed: int,
                                n_samples: int = 10000) -> pd.DataFrame:
    """LIME-style local binary perturbation strictly respecting algebraic dependencies."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(0, 2, size=(n_samples, 2))
    neighborhood = np.vstack([extract_features(tuple(x)) for x in raw])
    neighborhood[0] = instance
    
    distances = np.linalg.norm(neighborhood - instance, axis=1)
    weights = np.exp(-(distances ** 2) / (0.75 * np.sqrt(len(instance))) ** 2)
    class_idx = list(model.classes_).index(predicted)
    target = model.predict_proba(neighborhood)[:, class_idx]
    
    design = np.column_stack([np.ones(n_samples), neighborhood])
    W = np.sqrt(weights)[:, None]
    ridge = 1e-6 * np.eye(design.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.solve((design * W).T @ (design * W) + ridge, (design * W).T @ (target * W[:, 0]))
    return pd.DataFrame({"feature": FEATURE_NAMES, "contribution": beta[1:] * instance})

# ==============================================================================
# VISUALIZATION SUITE (LaTeX Compliant Raw Strings)
# ==============================================================================
def style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 11,
        "axes.labelsize": 10
    })


def save_figure(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(cm, encoder: str, epsilon: float, path: Path):
    fig, ax = plt.subplots(figsize=(4.3, 3.7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set(xlabel="Predicted Label", ylabel="True Label",
           title=rf"{encoder.title()} Encoding ($\epsilon$={epsilon:.2f})")
    save_figure(fig, path)


def plot_reliability_diagrams(proba: np.ndarray, y_true_idx: np.ndarray, encoder: str, epsilon: float, path: Path):
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    for idx, label in enumerate(LABELS):
        y_binary = (y_true_idx == idx).astype(int)
        prob_pred = proba[:, idx]
        prob_true, prob_pred_binned = calibration_curve(y_binary, prob_pred, n_bins=8, strategy="uniform")
        ax.plot(prob_pred_binned, prob_true, marker="o", label=f"Class {label}")

    ax.set(xlabel="Mean Predicted Probability", ylabel="Fraction of Positives",
           title=rf"Reliability Diagram: {encoder.title()} ($\epsilon$={epsilon:.2f})")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    save_figure(fig, path)


def plot_calibration_vs_noise(calib_df: pd.DataFrame, path_dir: Path):
    # ECE vs Noise
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.lineplot(data=calib_df, x="noise", y="ece", hue="encoder", marker="s", ax=ax)
    ax.set(xlabel=r"Depolarizing Noise $\epsilon$", ylabel="Expected Calibration Error (ECE)",
           title="Calibration Error vs. Quantum Noise")
    save_figure(fig, path_dir / "ece_vs_noise.png")

    # Brier Score vs Noise
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.lineplot(data=calib_df, x="noise", y="brier_score", hue="encoder", marker="D", ax=ax)
    ax.set(xlabel=r"Depolarizing Noise $\epsilon$", ylabel="Brier Score (Lower is Better)",
           title="Brier Score vs. Quantum Noise")
    save_figure(fig, path_dir / "brier_vs_noise.png")


def plot_summaries(out: Path):
    metrics = pd.read_csv(out / "metrics.csv")
    fidelity = pd.read_csv(out / "fidelity.csv")
    timings = pd.read_csv(out / "timings.csv")
    
    # Accuracy vs Noise
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.lineplot(data=metrics, x="noise", y="accuracy", hue="encoder", marker="o", ax=ax)
    ax.set(xlabel=r"Depolarizing Noise $\epsilon$", ylabel="Accuracy", ylim=(-0.02, 1.03))
    save_figure(fig, out / "figures" / "accuracy_vs_noise.png")
    
    # Fidelity vs Noise
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sns.lineplot(data=fidelity, x="noise", y="mean_fidelity", hue="encoder", marker="o", ax=ax)
    ax.set(xlabel=r"Depolarizing Noise $\epsilon$", ylabel="Mean State Fidelity", ylim=(-0.02, 1.03))
    save_figure(fig, out / "figures" / "fidelity_vs_noise.png")

    # Timings
    avg = timings.groupby("encoder", as_index=False)[["encoding_seconds_per_input", "decoding_seconds_per_instance"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7))
    sns.barplot(data=avg, x="encoder", y="encoding_seconds_per_input", ax=axes[0], palette="Blues_d")
    sns.barplot(data=avg, x="encoder", y="decoding_seconds_per_instance", ax=axes[1], palette="Oranges_d")
    axes[0].set(xlabel="Encoder", ylabel="Seconds/Input", title="Encoding Overhead")
    axes[1].set(xlabel="Encoder", ylabel="Seconds/Instance", title="Decoding Overhead")
    save_figure(fig, out / "figures" / "timings.png")

# ==============================================================================
# PIPELINE EXECUTION ENGINE
# ==============================================================================
@dataclass
class RunResult:
    metrics: pd.DataFrame
    calibration: pd.DataFrame
    predictions: pd.DataFrame


def run(config: dict) -> RunResult:
    out = Path(config["output_dir"])
    figs = out / "figures"
    models = out / "models"
    figs.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    
    style()
    seed = int(config["seed"])
    rng_train = np.random.default_rng(seed)
    
    train_inputs = balanced_inputs(int(config["train_per_class"]), rng_train)
    test_inputs = balanced_inputs(int(config["test_per_class"]), np.random.default_rng(seed + 1))
    
    metrics, calib_metrics, predictions, cms, fidelities, timings, importances, explanations = [], [], [], [], [], [], [], []

    for eidx, encoder in enumerate(config["encoders"]):
        train, _ = synthesize(train_inputs, encoder, 0.0, int(config["shots"]),
                             np.random.default_rng(seed + 100 + eidx))
        
        model, rf = fit_calibrated_model(train, seed + eidx,
                                          config["random_forest"], config["calibration_fraction"])
        
        joblib.dump({"model": model, "features": FEATURE_NAMES, "encoder": encoder},
                    models / f"{encoder}.joblib")
        
        for name, value in zip(FEATURE_NAMES, rf.feature_importances_):
            importances.append([encoder, name, value])

        for nidx, epsilon in enumerate(config["noise_levels"]):
            test, test_enc_time = synthesize(test_inputs, encoder, float(epsilon),
                                             int(config["shots"]),
                                             np.random.default_rng(seed + 1000 * eidx + nidx))
            Xf = test[FEATURE_NAMES].to_numpy()
            
            t0 = time.perf_counter()
            pred = model.predict(Xf)
            proba = model.predict_proba(Xf)
            dec_time = (time.perf_counter() - t0) / len(test)
            
            conf = proba.max(axis=1)
            cm = confusion_matrix(test["label"], pred, labels=LABELS)
            acc = accuracy_score(test["label"], pred)
            
            label_map = {lbl: idx for idx, lbl in enumerate(LABELS)}
            y_true_idx = np.array([label_map[lbl] for lbl in test["label"]])
            
            # Statistical Evaluation Metrics
            ece = compute_ece(proba, y_true_idx)
            brier = compute_multiclass_brier(proba, y_true_idx)
            entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1).mean()
            loss = log_loss(test["label"], proba, labels=LABELS)
            
            metrics.append([encoder, epsilon, acc, entropy, loss, int(len(test) - np.trace(cm)), len(test)])
            calib_metrics.append([encoder, epsilon, ece, brier, loss, acc])
            timings.append([encoder, epsilon, test_enc_time, dec_time])
            fidelities.append([encoder, epsilon, test["fidelity"].mean()])
            
            for i, row in test.reset_index(drop=True).iterrows():
                predictions.append([encoder, epsilon, i, row.label, pred[i], conf[i],
                                    *row[FEATURE_NAMES].tolist()])
            
            for i, true in enumerate(LABELS):
                for j, guessed in enumerate(LABELS):
                    cms.append([encoder, epsilon, true, guessed, int(cm[i, j])])
                    
            # Generate Reliability Diagrams & Confusion Matrices
            plot_reliability_diagrams(proba, y_true_idx, encoder, epsilon, figs / f"reliability_{encoder}_{epsilon:.2f}.png")
            plot_confusion(cm, encoder, epsilon, figs / f"cm_{encoder}_{epsilon:.2f}.png")

            # Local Surrogate Explanations for Uncertain Predictions
            flagged = np.flatnonzero(conf < config["confidence_threshold"])
            for idx in flagged[:int(config["explanations_per_encoder_noise"])]:
                exp = local_surrogate_explanation(model, Xf[idx], pred[idx],
                                                  seed + idx, int(config["local_explanation_samples"]))
                exp.insert(0, "sample", int(idx))
                exp.insert(0, "noise", epsilon)
                exp.insert(0, "encoder", encoder)
                explanations.append(exp)
                
                fig, ax = plt.subplots(figsize=(5.2, 3.2))
                colors = ["#2563EB" if v >= 0 else "#DC2626" for v in exp["contribution"]]
                ax.barh(exp["feature"], exp["contribution"], color=colors)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set(xlabel=rf"Contribution toward class {pred[idx]}",
                       title=rf"{encoder.title()}, $\epsilon$={epsilon:.2f}, sample {idx}")
                save_figure(fig, figs / f"explanation_{encoder}_{epsilon:.2f}_{idx}.png")

    # Data Export
    metric_df = pd.DataFrame(metrics, columns=["encoder", "noise", "accuracy", "mean_entropy", "log_loss", "misclassifications", "n"])
    calib_df = pd.DataFrame(calib_metrics, columns=["encoder", "noise", "ece", "brier_score", "log_loss", "accuracy"])
    pred_df = pd.DataFrame(predictions, columns=["encoder", "noise", "sample", "true_label",
                                                  "predicted_label", "confidence", *FEATURE_NAMES])
    
    metric_df.to_csv(out / "metrics.csv", index=False)
    calib_df.to_csv(out / "calibration_metrics.csv", index=False)
    pred_df.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(cms, columns=["encoder", "noise", "true_label", "predicted_label", "count"]).to_csv(out / "confusion_matrices.csv", index=False)
    pd.DataFrame(fidelities, columns=["encoder", "noise", "mean_fidelity"]).to_csv(out / "fidelity.csv", index=False)
    pd.DataFrame(timings, columns=["encoder", "noise", "encoding_seconds_per_input", "decoding_seconds_per_instance"]).to_csv(out / "timings.csv", index=False)
    pd.DataFrame(importances, columns=["encoder", "feature", "importance"]).to_csv(out / "feature_importance.csv", index=False)
    
    if explanations:
        pd.concat(explanations, ignore_index=True).to_csv(out / "local_explanations.csv", index=False)
        
    (out / "run_manifest.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    
    # Generate Visual Summaries
    plot_summaries(out)
    plot_calibration_vs_noise(calib_df, figs)
    
    print(f"\nExecution Complete. All artifacts saved to: '{out.resolve()}'")
    return RunResult(metric_df, calib_df, pred_df)


if __name__ == "__main__":
    results = run(DEFAULT_CONFIG)
