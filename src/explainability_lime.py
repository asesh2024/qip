#!/usr/bin/env python3
"""
Explainability: Local Surrogate (LIME) Explanations with Constrained Binary Perturbations.
Loads trained model artifacts from pipeline outputs to generate feature attribution diagrams.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FEATURE_NAMES = ["b0", "b1", "parity", "weight", "correlation"]


def extract_features(bits: tuple[int, int]) -> np.ndarray:
    b0, b1 = bits
    return np.array([b0, b1, b0 ^ b1, b0 + b1, b0 * b1], dtype=float)


def local_surrogate_explanation(model, instance: np.ndarray, predicted: str, seed: int, n_samples: int = 10000):
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
    return FEATURE_NAMES, beta[1:] * instance


def save_explanation_figure(features, contributions, encoder_name: str, epsilon: float, sample_idx: int, predicted_class: str, save_path: Path):
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=150)
    colors = ["#2563EB" if v >= 0 else "#DC2626" for v in contributions]
    ax.barh(features, contributions, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(rf"Contribution toward class {predicted_class}")
    ax.set_title(rf"{encoder_name.title()}, $\epsilon$={epsilon:.2f}, sample {sample_idx}")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Generated explanation figure: {save_path}")


def main(output_dir: str = "outputs/run2", n_perturbations: int = 10000):
    base_dir = Path(output_dir)
    models_dir = base_dir / "models"
    figs_dir = base_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    targets = [
        {"encoder": "basis", "epsilon": 0.20, "sample_idx": 42, "bits": (1, 0), "filename": "explanation_basis_0.20_42.png"},
        {"encoder": "angle", "epsilon": 0.20, "sample_idx": 18, "bits": (0, 1), "filename": "explanation_angle_0.20_18.png"},
        {"encoder": "amplitude", "epsilon": 0.20, "sample_idx": 5, "bits": (1, 1), "filename": "explanation_amplitude_0.20_5.png"},
    ]
    
    for item in targets:
        model_path = models_dir / f"{item['encoder']}.joblib"
        if not model_path.exists():
            print(f"Model artifact not found at {model_path}. Run pipeline.py first.")
            continue
            
        artifact = joblib.load(model_path)
        model = artifact["model"]
        
        instance = extract_features(item["bits"])
        pred = model.predict(instance.reshape(1, -1))[0]
        
        features, contributions = local_surrogate_explanation(
            model=model,
            instance=instance,
            predicted=pred,
            seed=42 + item["sample_idx"],
            n_samples=n_perturbations
        )
        
        out_file = figs_dir / item["filename"]
        save_explanation_figure(features, contributions, item["encoder"], item["epsilon"], item["sample_idx"], pred, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LIME explanations for quantum classifiers.")
    parser.add_argument("--output_dir", type=str, default="outputs/run2", help="Base directory containing model artifacts.")
    parser.add_argument("--samples", type=int, default=10000, help="Number of perturbation samples for LIME.")
    args = parser.parse_args()
    main(output_dir=args.output_dir, n_perturbations=args.samples)
