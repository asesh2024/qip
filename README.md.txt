# Quantum Classification Under Noise: A Comparative Study of Encoding Strategies

This repository provides the official implementation, datasets, and reproducibility suite for evaluating the noise resilience, probabilistic calibration, and interpretability of Basis, Angle, and Amplitude quantum encodings under depolarizing noise.

##  Repository Structure
- `src/pipeline.py`: Runs end-to-end quantum state synthesis, depolarizing channel simulation, Platt-calibrated Random Forest classification, and metric evaluation (ECE, Brier score, state fidelity).
- `src/explainability_lime.py`: Generates constrained local surrogate (LIME) feature attributions.
- `data/`: Contains base synthetic parameter datasets and train/calibration/test splits.
- `outputs/`: Evaluation metrics (CSV), trained model artifacts, and high-resolution figures.

##  Quickstart & Reproduction

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt


### 2. Run main pipeline

python src/pipeline.py


### 3. Generate LIME Interpretability Figures

python src/explainability_lime.py --output_dir outputs --samples 10000


